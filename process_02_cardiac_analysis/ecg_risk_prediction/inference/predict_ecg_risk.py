import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from ecg_risk_prediction.models.ecg_cnn import ECGCNN

# CONFIG
MODEL_PATH = "../outputs/best_ecg_cnn.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "../outputs"


class CardiacPredictor:
    def __init__(self):
        self.model = ECGCNN(num_classes=4).to(DEVICE)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        self.model.eval()
        self.classes = ['Normal', 'Abnormal Heartbeat', 'Active Myocardial Infarction', 'History of MI']

    def plot_lead_activity(self, lead_variance, suspected_vessel):
        """Generates a visual heatmap of lead activity for the report."""
        plt.figure(figsize=(12, 6))
        leads = [f"Lead {i + 1}" for i in range(12)]
        sns.barplot(x=leads, y=lead_variance, hue=leads, palette="Reds_r", legend=False)
        plt.title(f"Lead Activity Profile - Suspected: {suspected_vessel}")
        plt.ylabel("Signal Variance (Intensity)")
        plt.savefig(os.path.join(OUTPUT_DIR, "lead_activity_report.png"))
        print(f"Visual activity report saved to {OUTPUT_DIR}/lead_activity_report.png")

    def get_artery_localization(self, ecg_tensor):
        # Calculate variance across time for each lead (12 rows)
        lead_activity = torch.var(ecg_tensor[0], dim=1).cpu().numpy()

        # Clinical Mapping (Indices 0-11)
        # LAD: V1-V4 (Rows 6-9) | RCA: II, III, aVF (Rows 1, 2, 4) | LCX: I, aVL, V5, V6 (Rows 0, 5, 10, 11)
        lad_score = np.mean(lead_activity[6:10])
        rca_score = np.mean([lead_activity[1], lead_activity[2], lead_activity[4]])
        lcx_score = np.mean([lead_activity[0], lead_activity[5], lead_activity[10], lead_activity[11]])

        scores = {"LAD (Anterior)": lad_score, "RCA (Inferior)": rca_score, "LCX (Lateral)": lcx_score}
        suspected = max(scores, key=scores.get)

        # Plot the activity for the tutor
        self.plot_lead_activity(lead_activity, suspected)
        return suspected

    def predict(self, processed_csv_path):
        df = pd.read_csv(processed_csv_path)
        if "filename" in df.columns: df = df.drop(columns=["filename"])
        data = df.values.astype(np.float32)[0]  # Take first patient in file

        # Preprocess exactly like training
        data = (data - data.mean()) / (data.std() + 1e-8)
        tensor_data = torch.tensor(data).reshape(1, 1, 12, -1).to(DEVICE)

        with torch.no_grad():
            outputs = self.model(tensor_data)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        res = {
            "prediction": self.classes[pred.item()],
            "confidence": f"{conf.item() * 100:.2f}%",
            "risk_level": "HIGH" if pred.item() == 2 else "MEDIUM" if pred.item() in [1, 3] else "LOW"
        }

        if pred.item() == 2:  # Only localize for active heart attacks
            res["suspected_vessel"] = self.get_artery_localization(tensor_data[0])
        else:
            res["suspected_vessel"] = "N/A - No acute blockage detected"

        return res


if __name__ == "__main__":
    predictor = CardiacPredictor()
    # Change this path to test different files!
    sample_file = "../data/ECG Images of Myocardial Infarction Patients (240x12=2880)_flattened.csv"
    report = predictor.predict(sample_file)

    print("\n--- PATIENT CARDIAC RISK REPORT ---")
    print(f"Status: {report['prediction']}")
    print(f"Confidence: {report['confidence']}")
    print(f"Emergency Priority: {report['risk_level']}")
    print(f"Artery Localization: {report['suspected_vessel']}")
    print("------------------------------------\n")