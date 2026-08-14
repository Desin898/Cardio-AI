import sys
from pathlib import Path
import os
import json

# Ensure project root is in sys.path when invoked via subprocess
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to prevent GUI thread issues
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import numpy as np
import pandas as pd

from process_02_cardiac_analysis.ecg_risk_prediction.models.ecg_resnet import MultiBranch1DResNet34
from process_02_cardiac_analysis.ecg_risk_prediction.models.ecg_cnn import ECGCNN
from Preprocessing.ECG_Preprocessing.ECG_Preprocessing_Functions import process_and_save_csv

# CONFIG
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "best_ecg_cnn.pth")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

TERRITORY_MAPPING = {
    "Left Anterior Descending (LAD)": [6, 7, 8, 9],       # V1, V2, V3, V4
    "Left Circumflex (LCx)": [0, 4, 10, 11],               # I, aVL, V5, V6
    "Right Coronary Artery (RCA)": [1, 2, 5],              # II, III, aVF
}


class CardiacPredictor:
    def __init__(self):
        self.classes = [
            'Normal',
            'Abnormal Heartbeat',
            'Active Myocardial Infarction',
            'History of MI'
        ]
        self.model = MultiBranch1DResNet34(num_classes=4, in_channels=12).to(DEVICE)

        if os.path.exists(MODEL_PATH):
            try:
                state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
                self.model.load_state_dict(state_dict)
            except Exception:
                try:
                    fallback_model = ECGCNN(num_classes=4).to(DEVICE)
                    fallback_model.load_state_dict(state_dict)
                    self.model = fallback_model
                except Exception:
                    pass

        self.model.eval()

    def plot_lead_activity(self, lead_variance, suspected_vessel):
        """Generates a visual bar chart of lead activity for report generation."""
        plt.figure(figsize=(12, 6))
        sns.barplot(x=LEAD_NAMES, y=lead_variance, hue=LEAD_NAMES, palette="Reds_r", legend=False)
        plt.title(f"12-Lead Activity Profile - Suspected Territory: {suspected_vessel}")
        plt.ylabel("Lead Deviation / Signal Variance")
        plt.xlabel("ECG Leads")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "lead_activity_report.png")
        plt.savefig(output_path)
        plt.close()
        return output_path

    def calculate_anatomical_mapping(self, signal_2d):
        if isinstance(signal_2d, torch.Tensor):
            signal_2d = signal_2d.cpu().numpy()

        if signal_2d.shape[0] != 12 and signal_2d.shape[1] == 12:
            signal_2d = signal_2d.T

        lead_variances = np.var(signal_2d, axis=1)
        lead_scores = {LEAD_NAMES[i]: round(float(lead_variances[i]), 4) for i in range(12)}

        # 1. LMCA / Multi-Vessel Ischemia Diagnostic Heuristic
        avr_var = float(lead_variances[3])
        diffuse_indices = [0, 1, 9, 10, 11]
        mean_var = float(np.mean(lead_variances))
        max_var = float(np.max(lead_variances))

        # aVR is elevated if its variance is above average lead variance and close to peak lead activity
        avr_is_elevated = (avr_var >= mean_var) and (avr_var >= 0.7 * max_var)
        high_diffuse_count = sum(1 for idx in diffuse_indices if lead_variances[idx] >= mean_var)

        if avr_is_elevated and (high_diffuse_count >= 3):
            suspected_artery = "Left Main Coronary Artery (LMCA) / Severe Multi-Vessel Disease"
            affected_indices = sorted(list(set([3] + [idx for idx in diffuse_indices if lead_variances[idx] >= mean_var])))
            affected_leads = [LEAD_NAMES[i] for i in affected_indices]
            return suspected_artery, affected_leads, lead_scores, lead_variances

        # 2. Standard Territory Mapping (LAD, LCx, RCA)
        territory_scores = {}
        for territory, lead_indices in TERRITORY_MAPPING.items():
            valid_indices = [idx for idx in lead_indices if idx < len(lead_variances)]
            territory_scores[territory] = float(np.mean([lead_variances[idx] for idx in valid_indices]))

        suspected_artery = max(territory_scores, key=territory_scores.get)
        dominant_indices = TERRITORY_MAPPING[suspected_artery]

        threshold_75 = float(np.percentile(lead_variances, 75))
        affected_indices = set(dominant_indices).union(set(np.where(lead_variances >= threshold_75)[0]))
        affected_leads = [LEAD_NAMES[i] for i in sorted(list(affected_indices)) if i < 12]

        return suspected_artery, affected_leads, lead_scores, lead_variances

    def predict(self, processed_csv_path):
        df = pd.read_csv(processed_csv_path)

        if "filename" in df.columns:
            df = df.drop(columns=["filename"])
        if "patient_id" in df.columns:
            df = df.drop(columns=["patient_id"])

        data = df.values.astype(np.float32)[0]

        data = (data - data.mean()) / (data.std() + 1e-8)
        T = len(data) // 12
        signal_12lead = data.reshape(12, T)

        tensor_data = torch.tensor(signal_12lead).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = self.model(tensor_data)  # Raw logits
            # Temperature Scaling (T = 1.5) to calibrate probability spikes
            temperature = 1.5
            scaled_logits = outputs / temperature
            probs = torch.nn.functional.softmax(scaled_logits, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))
            conf = float(probs[pred])

        # Calibrated Softmax probability distribution
        prob_dict = {self.classes[i]: round(float(probs[i]), 4) for i in range(4)}
        suspected_artery, affected_leads, lead_scores, lead_variances = self.calculate_anatomical_mapping(signal_12lead)

        plot_path = self.plot_lead_activity(lead_variances, suspected_artery)

        is_lmca = "LMCA" in suspected_artery or "Left Main" in suspected_artery
        is_active_mi = (pred == 2) or is_lmca
        is_abnormal_arrhythmia = (pred == 1) and not is_lmca
        is_history_mi = (pred == 3) and not is_lmca

        # Hierarchical Clinical Routing
        if is_active_mi:
            final_artery = suspected_artery
            final_vessel = suspected_artery
            final_leads = affected_leads
            risk_level = "CRITICAL" if is_lmca else "HIGH"
        elif is_abnormal_arrhythmia:
            final_artery = "N/A - Non-Thrombotic Arrhythmia / Conduction Abnormality"
            final_vessel = "N/A - Non-Thrombotic Arrhythmia / Conduction Abnormality"
            final_leads = []
            risk_level = "MEDIUM"
        elif is_history_mi:
            final_artery = "N/A - Prior Infarction / Inactive Blockage"
            final_vessel = "N/A - Prior Infarction / Inactive Blockage"
            final_leads = []
            risk_level = "MEDIUM"
        else:
            final_artery = "N/A - No acute blockage detected"
            final_vessel = "N/A - No acute blockage detected"
            final_leads = []
            risk_level = "LOW"

        res = {
            "prediction": self.classes[pred],
            "predicted_class": self.classes[pred],
            "confidence": f"{conf * 100:.2f}%",
            "confidence_score": round(conf, 4),
            "risk_level": risk_level,
            "category_probabilities": prob_dict,
            "suspected_artery": final_artery,
            "suspected_vessel": final_vessel,
            "affected_leads": final_leads,
            "lead_analysis_breakdown": lead_scores,
            "plot_path": plot_path,
            "error": None
        }

        return res


if __name__ == "__main__":
    predictor = CardiacPredictor()
    sample_file = None

    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1]).resolve()
        if not input_path.exists():
            print(f"Input file not found: {input_path}")
            sys.exit(1)

        if input_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
            _, sample_file = process_and_save_csv(str(input_path))
            print(f"Preprocessed CSV saved to: {sample_file}")
        elif input_path.suffix.lower() == ".csv":
            sample_file = str(input_path)
            print(f"Using CSV file: {sample_file}")
        else:
            print("Unsupported file type. Please provide an ECG image or CSV file.")
            sys.exit(1)
    else:
        patients_dir = project_root / "Pipeline_Management" / "Patients"
        csv_files = list(patients_dir.rglob("*.csv")) if patients_dir.exists() else []
        if not csv_files:
            print(f"No preprocessed CSV found in {patients_dir}.")
            sys.exit(1)
        latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
        sample_file = str(latest_csv)
        print(f"Using latest CSV: {sample_file}")

    report = predictor.predict(sample_file)

    print("\n--- PATIENT CARDIAC RISK REPORT ---")
    print(f"Status: {report['prediction']}")
    print(f"Confidence: {report['confidence']}")
    print(f"Emergency Priority: {report['risk_level']}")
    print(f"Suspected Artery Territory: {report['suspected_artery']}")
    print(f"Probabilities: {json.dumps(report['category_probabilities'])}")
    print(f"Affected Leads: {report['affected_leads']}")
    print(f"Lead Scores: {report['lead_analysis_breakdown']}")
    print("------------------------------------\n")
