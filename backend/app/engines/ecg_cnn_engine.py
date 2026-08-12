import os
import sys
import subprocess
import logging
from typing import Dict, Any

from backend.app.core.config import settings
from backend.app.engines.base_engine import BaseMLEngine

INFERENCE_SCRIPT_PATH = (
    settings.PROJECT_ROOT
    / "process_02_cardiac_analysis"
    / "ecg_risk_prediction"
    / "inference"
    / "predict_ecg_risk.py"
)

class ECGCNNEngine(BaseMLEngine):
    def load_models(self) -> None:
        pass  # Script loads model lazily on invocation

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        csv_path = data.get("csv_path")
        if not csv_path or not os.path.isfile(csv_path):
            msg = f"Pre-processed CSV not found at: {csv_path}"
            logging.error(msg)
            return {
                "error": msg, "prediction": "Error", "confidence": "0%",
                "risk_level": "UNKNOWN", "suspected_vessel": "N/A"
            }

        if not INFERENCE_SCRIPT_PATH.exists():
            msg = f"Inference script not found at: {INFERENCE_SCRIPT_PATH}"
            logging.error(msg)
            return {
                "error": msg, "prediction": "Error", "confidence": "0%",
                "risk_level": "UNKNOWN", "suspected_vessel": "N/A"
            }

        try:
            result = subprocess.run(
                [sys.executable, str(INFERENCE_SCRIPT_PATH), csv_path],
                capture_output=True, text=True, timeout=120, cwd=str(settings.PROJECT_ROOT),
            )

            if result.stdout:
                logging.info(f"ECG Inference stdout:\n{result.stdout}")
            if result.stderr:
                logging.error(f"ECG Inference stderr:\n{result.stderr}")

            if result.returncode != 0:
                stderr_clean = result.stderr.strip() or "No stderr captured."
                return {
                    "error": stderr_clean, "prediction": "Error", "confidence": "0%",
                    "risk_level": "UNKNOWN", "suspected_vessel": "N/A"
                }

            prediction_result = {
                "prediction": "Unknown", "confidence": "0%",
                "risk_level": "UNKNOWN", "suspected_vessel": "N/A", "plot_path": None,
            }
            for line in result.stdout.splitlines():
                line = line.strip()
                if line.startswith("Status:"):
                    prediction_result["prediction"] = line.split(":", 1)[1].strip()
                elif line.startswith("Confidence:"):
                    prediction_result["confidence"] = line.split(":", 1)[1].strip()
                elif line.startswith("Emergency Priority:"):
                    prediction_result["risk_level"] = line.split(":", 1)[1].strip()
                elif line.startswith("Artery Localization:"):
                    prediction_result["suspected_vessel"] = line.split(":", 1)[1].strip()

            return prediction_result

        except subprocess.TimeoutExpired:
            return {"error": "Inference script timed out (>120s).", "prediction": "Timeout",
                    "confidence": "0%", "risk_level": "UNKNOWN", "suspected_vessel": "N/A"}
        except Exception as e:
            return {"error": f"Failed to launch inference: {str(e)}", "prediction": "Error",
                    "confidence": "0%", "risk_level": "UNKNOWN", "suspected_vessel": "N/A"}

ecg_cnn_engine = ECGCNNEngine()
