
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from ecg_risk_prediction.models.ecg_cnn import ECGCNN


# CONFIGURATION

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 20
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

    # MODEL
    model = ECGCNN(num_classes=NUM_CLASSES).to(DEVICE)

    # CLASS-WEIGHTED LOSS
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # TRAIN LOOP
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        val_accuracy = correct / total

        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Accuracy: {val_accuracy:.4f}")

    # SAVE MODEL
    model_path = os.path.join(OUTPUT_DIR, "ecg_cnn_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train()
