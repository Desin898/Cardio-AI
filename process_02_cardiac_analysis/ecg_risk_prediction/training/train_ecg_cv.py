import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize
from torch.optim.lr_scheduler import StepLR
from ecg_risk_prediction.models.ecg_cnn import ECGCNN

# CONFIGURATION
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 50
K_FOLDS = 5
RANDOM_STATE = 42

DATA_DIR = "../data"
OUTPUT_DIR = "../outputs/cv_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self): return len(self.y)

    def __getitem__(self, idx): return self.X[idx], self.y[idx]


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
        df = pd.read_csv(file_path)
        if "filename" in df.columns: df = df.drop(columns=["filename"])
        df = df.apply(pd.to_numeric, errors="coerce").dropna()
        X_all.append(df.values.astype(np.float32))
        y_all.extend([label] * len(df))
    X_all = np.vstack(X_all)
    y_all = np.array(y_all)
    X_all = (X_all - X_all.mean()) / (X_all.std() + 1e-8)
    T = X_all.shape[1] // 12
    X_all = X_all.reshape(X_all.shape[0], 1, 12, T)
    return X_all, y_all


def train_cv():
    X, y = load_ecg_data()
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    all_fold_history = []
    fold_accuracies = []
    all_targets, all_probs = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- FOLD {fold + 1}/{K_FOLDS} ---")
        train_loader = DataLoader(ECGDataset(X[train_idx], y[train_idx]), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(ECGDataset(X[val_idx], y[val_idx]), batch_size=BATCH_SIZE, shuffle=False)

        model = ECGCNN(num_classes=4).to(DEVICE)
        weights = torch.tensor(compute_class_weight("balanced", classes=np.unique(y[train_idx]), y=y[train_idx]),
                               dtype=torch.float32).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
        criterion = nn.CrossEntropyLoss(weight=weights)

        history = {'t_loss': [], 'v_acc': []}
        best_acc = 0

        for epoch in range(EPOCHS):
            model.train()
            loss_acc = 0
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                optimizer.zero_grad()
                l = criterion(model(X_b), y_b)
                l.backward();
                optimizer.step()
                loss_acc += l.item()

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    out = model(X_b.to(DEVICE))
                    _, p = torch.max(out, 1)
                    total += y_b.size(0);
                    correct += (p.cpu() == y_b).sum().item()

            acc = correct / total
            history['t_loss'].append(loss_acc / len(train_loader))
            history['v_acc'].append(acc)
            if acc > best_acc: best_acc = acc

        fold_accuracies.append(best_acc)
        all_fold_history.append(history)
        print(f"Fold {fold + 1} Accuracy: {best_acc:.4f}")

        # Collect data for aggregate ROC from final fold
        if fold == K_FOLDS - 1:
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    out = model(X_b.to(DEVICE))
                    all_probs.extend(torch.softmax(out, dim=1).cpu().numpy())
                    all_targets.extend(y_b.numpy())

    # SAVE CV RESULTS
    print(f"\nCV MEAN ACCURACY: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")

    # Plot Mean Learning Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.mean([h['t_loss'] for h in all_fold_history], axis=0), color='blue', label='Mean Loss')
    plt.title('CV Mean Loss');
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(np.mean([h['v_acc'] for h in all_fold_history], axis=0), color='#FFD700', label='Mean Acc')
    plt.title('CV Mean Accuracy');
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "cv_learning_curves.png"))

    # Plot ROC
    y_true_bin = label_binarize(all_targets, classes=[0, 1, 2, 3])
    plt.figure(figsize=(8, 6))
    for i, name in enumerate(['Normal', 'Abnormal', 'MI', 'History']):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], np.array(all_probs)[:, i])
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], 'k--');
    plt.title('CV Aggregate ROC Curve');
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "cv_roc_curve.png"))


if __name__ == "__main__":
    train_cv()