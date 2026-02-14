
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from torch.optim.lr_scheduler import StepLR
from ecg_risk_prediction.models.ecg_cnn import ECGCNN


# CONFIGURATION

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
NUM_CLASSES = 4
RANDOM_STATE = 42

DATA_DIR = "../data"
OUTPUT_DIR = "../outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# DATASET CLASS
class ECGDataset(Dataset):
    """
    ECG Dataset wrapper.
    Each sample is reshaped to (1, H, W) for CNN input.
    """

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# LOAD ECG CSV DATA
def load_ecg_data():
    datasets = [
        ("Normal Person ECG Images (284x12=3408)_flattened.csv", 0),
        ("ECG Images of Patient that have abnormal heartbeat (233x12=2796)_flattened.csv", 1),
        ("ECG Images of Myocardial Infarction Patients (240x12=2880)_flattened.csv", 2),
        ("ECG Images of Patient that have History of MI (172x12=2064)_flattened.csv", 3),
    ]

    X_all, y_all = [], []

    for file_name, label in datasets:
        file_path = os.path.join(DATA_DIR, file_name)
        print(f"Loading: {file_path}")

        df = pd.read_csv(file_path)

        # 1) Drop non-numeric column if it exists
        if "filename" in df.columns:
            df = df.drop(columns=["filename"])

        # 2) Keep only numeric columns
        df = df.apply(pd.to_numeric, errors="coerce").dropna()

        if len(df) == 0:
            print(f"  WARNING: {file_name} became empty after cleaning.")
            continue

        X_all.append(df.values.astype(np.float32))
        y_all.extend([label] * len(df))

        print(f"  Loaded {len(df)} samples.")

    if not X_all:
        raise ValueError(f"No data was loaded. Check CSV format + file names in: {os.path.abspath(DATA_DIR)}")

    X_all = np.vstack(X_all)
    y_all = np.array(y_all)

    # normalize
    X_all = (X_all - X_all.mean()) / (X_all.std() + 1e-8)

    # reshape into (N, 1, 12, T)
    if X_all.shape[1] % 12 != 0:
        raise ValueError(f"Feature count {X_all.shape[1]} is not divisible by 12. Cannot reshape into 12 leads.")

    T = X_all.shape[1] // 12
    X_all = X_all.reshape(X_all.shape[0], 1, 12, T)

    print(f"Final Dataset Shape: {X_all.shape}")
    return X_all, y_all

# TRAINING PIPELINE
def train():
    print(f"Using device: {DEVICE}")
    X, y = load_ecg_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    train_dataset = ECGDataset(X_train, y_train)
    val_dataset = ECGDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ECGCNN(num_classes=NUM_CLASSES).to(DEVICE)

    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    # History tracking lists
    history = {'train_loss': [], 'val_acc': []}

    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validation with Metric Collection
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        avg_train_loss = running_loss / len(train_loader)
        val_accuracy = np.mean(np.array(all_preds) == np.array(all_targets))

        # Save to history for plotting
        history['train_loss'].append(avg_train_loss)
        history['val_acc'].append(val_accuracy)

        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_ecg_cnn.pth"))
            print(f"*** Epoch [{epoch + 1}] | New Best Acc: {val_accuracy:.4f} - Saved! ***")
        else:
            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] | Loss: {train_loss / len(train_loader):.4f} | Acc: {val_accuracy:.4f}")

    # FINAL EVALUATION UPGRADE
    print("\n" + "=" * 40)
    print("FINAL CLINICAL REPORT")
    print("=" * 40)
    target_names = ['Normal', 'Abnormal', 'Active MI', 'History of MI']
    print(classification_report(all_targets, all_preds, target_names=target_names))

    # Confusion Matrix Visualization
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Cardiac Risk Classification Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(OUTPUT_DIR, "evaluation_matrix.png"))
    print(f"Evaluation visualization saved to {OUTPUT_DIR}/evaluation_matrix.png")

    #  Learning Curves Plotting ---
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS + 1), history['train_loss'], color='red', label='Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS + 1), history['val_acc'], color='blue', label='Accuracy')
    plt.title('Validation Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "learning_curves.png"))
    print(f"\nLearning curves saved to {OUTPUT_DIR}/learning_curves.png")

if __name__ == "__main__":
    train()