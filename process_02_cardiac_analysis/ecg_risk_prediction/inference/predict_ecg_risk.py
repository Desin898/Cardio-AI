import sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from process_02_cardiac_analysis.ecg_risk_prediction.models.ecg_cnn import ECGCNN
from Preprocessing.ECG_Preprocessing.ECG_Preprocessing_Functions import process_and_save_csv

# CONFIG
# Absolute paths so the script works regardless of where it's launched from
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "best_ecg_cnn.pth")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CardiacPredictor:
    def __init__(self):
        self.model = ECGCNN(num_classes=4).to(DEVICE)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        self.model.eval()
        self.classes = [
            'Normal',
            'Abnormal Heartbeat',
            'Active Myocardial Infarction',
            'History of MI'
        ]

    def plot_lead_activity(self, lead_variance, suspected_vessel):
        """Generates a visual heatmap of lead activity for the report."""
        plt.figure(figsize=(12, 6))
        leads = [f"Lead {i + 1}" for i in range(12)]
        sns.barplot(x=leads, y=lead_variance, hue=leads, palette="Reds_r", legend=False)
        plt.title(f"Lead Activity Profile - Suspected: {suspected_vessel}")
        plt.ylabel("Signal Variance (Intensity)")

        output_path = os.path.join(OUTPUT_DIR, "lead_activity_report.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Visual activity report saved to {output_path}")

    def get_artery_localization(self, ecg_tensor):
        # Calculate variance across time for each lead (12 rows)
        lead_activity = torch.var(ecg_tensor[0], dim=1).cpu().numpy()

        # Clinical Mapping (Indices 0-11)
        # LAD: V1-V4 (Rows 6-9) | RCA: II, III, aVF (Rows 1, 2, 4) | LCX: I, aVL, V5, V6 (Rows 0, 5, 10, 11)
        lad_score = np.mean(lead_activity[6:10])
        rca_score = np.mean([lead_activity[1], lead_activity[2], lead_activity[4]])
        lcx_score = np.mean([lead_activity[0], lead_activity[5], lead_activity[10], lead_activity[11]])

        scores = {
            "LAD (Anterior)": lad_score,
            "RCA (Inferior)": rca_score,
            "LCX (Lateral)": lcx_score
        }
        suspected = max(scores, key=scores.get)

        # Plot the activity
        self.plot_lead_activity(lead_activity, suspected)
        return suspected

    def predict(self, processed_csv_path):
        df = pd.read_csv(processed_csv_path)

        if "filename" in df.columns:
            df = df.drop(columns=["filename"])

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
    # Determine the project root (where Pipeline_Management is)
    script_dir = Path(__file__).resolve().parent
    # This script is in:
    # coronary-ai-system/process_02_cardiac_analysis/ecg_risk_prediction/inference/
    # So go up three levels to reach coronary-ai-system
    project_root = script_dir.parent.parent.parent

    # Define the Patients directory
    patients_dir = project_root / "Pipeline_Management" / "Patients"

    predictor = CardiacPredictor()

    sample_file = None

    # If an input path is provided, use it
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1]).resolve()

        if not input_path.exists():
            print(f"Input file not found: {input_path}")
            sys.exit(1)

        # If user passed an image, preprocess it and get the CSV path back
        if input_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
            _, sample_file = process_and_save_csv(str(input_path))
            print(f"Preprocessed CSV saved to: {sample_file}")

        # If user passed a CSV directly, use it as-is
        elif input_path.suffix.lower() == ".csv":
            sample_file = str(input_path)
            print(f"Using CSV file: {sample_file}")

        else:
            print("Unsupported file type. Please provide an ECG image or CSV file.")
            sys.exit(1)

    # Fallback: find the most recently modified CSV under Patients
    else:
        csv_files = list(patients_dir.rglob("*.csv"))

        if not csv_files:
            print(f"No preprocessed CSV found in {patients_dir}. Please ensure the ECG has been processed.")
            sys.exit(1)

        latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
        sample_file = str(latest_csv)
        print(f"Using latest CSV: {sample_file}")

    # Run prediction
    report = predictor.predict(sample_file)

    print("\n--- PATIENT CARDIAC RISK REPORT ---")
    print(f"Status: {report['prediction']}")
    print(f"Confidence: {report['confidence']}")
    print(f"Emergency Priority: {report['risk_level']}")
    print(f"Artery Localization: {report['suspected_vessel']}")
    print("------------------------------------\n")

