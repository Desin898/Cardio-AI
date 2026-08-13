import os
import pickle
import logging
import warnings
from typing import Dict, Any
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*unpickle.*")
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")

from backend.app.core.config import settings
from backend.app.engines.base_engine import BaseMLEngine

TEN_YEAR_FEATURES = [
    "age", "male", "currentSmoker", "cigsPerDay",
    "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes",
    "totChol", "sysBP", "diaBP", "pulsePressure", "bp_ratio",
    "BMI", "heartRate", "glucose",
]

def _fix_unpickled_estimator(est):
    if isinstance(est, dict):
        for v in est.values():
            _fix_unpickled_estimator(v)
        return
    if isinstance(est, (list, tuple)):
        for item in est:
            _fix_unpickled_estimator(item)
        return
    if hasattr(est, 'steps'):
        for _, step in est.steps:
            _fix_unpickled_estimator(step)
    if hasattr(est, 'named_steps'):
        for _, step in est.named_steps.items():
            _fix_unpickled_estimator(step)
    if hasattr(est, '__dict__'):
        if not hasattr(est, 'multi_class'):
            setattr(est, 'multi_class', 'auto')
        if not hasattr(est, 'keep_empty_features'):
            setattr(est, 'keep_empty_features', False)

class TenYearRiskEngine(BaseMLEngine):
    def __init__(self):
        self.ten_year_model = None
        self.ten_year_threshold = 0.5
        self.is_loaded = False

    def load_models(self) -> None:
        if self.is_loaded:
            return
        model_file = settings.TEN_YEAR_MODEL_DIR / "ten_year_model.pkl"
        thresh_file = settings.TEN_YEAR_MODEL_DIR / "ten_year_threshold.pkl"

        try:
            if model_file.exists():
                with open(model_file, "rb") as f:
                    self.ten_year_model = pickle.load(f)
                _fix_unpickled_estimator(self.ten_year_model)
                if thresh_file.exists():
                    with open(thresh_file, "rb") as f:
                        self.ten_year_threshold = float(pickle.load(f))
                self.is_loaded = True
                logging.info(f"Ten-year CHD model loaded (threshold={self.ten_year_threshold:.4f})")
            else:
                logging.warning(f"Ten-year CHD model NOT loaded: File not found at {model_file}")
        except Exception as e:
            logging.error(f"Failed to load ten-year CHD model: {e}")
            self.ten_year_model, self.ten_year_threshold = None, 0.5

    def build_features(self, data: dict) -> pd.DataFrame:
        sys_bp = float(data.get("systolic_bp", 120))
        dia_bp = float(data.get("diastolic_bp", 80))
        smoker_val = data.get("current_smoker")
        if smoker_val is not None:
            is_current_smoker = 1 if bool(smoker_val) else 0
        else:
            is_current_smoker = 1 if str(data.get("smoker", "")).strip().lower() in ("yes", "true", "1") else 0

        cigs_per_day = 0.0
        if is_current_smoker or str(data.get("smoker", "")).strip() == "Former":
            cigs_per_day = float(data.get("cigs_per_day", data.get("cigsPerDay", 0)) or 0)

        hyp_flag = data.get("prevalent_hyp", "")
        if hyp_flag == "Yes":
            prevalent_hyp = 1
        elif hyp_flag == "No":
            prevalent_hyp = 0
        else:
            prevalent_hyp = 1 if (sys_bp >= 140 or dia_bp >= 90) else 0

        features = {
            "age": float(data.get("age", 45)),
            "male": 1 if str(data.get("gender", "")).strip().lower() in ("male", "m", "1") else 0,
            "currentSmoker": is_current_smoker,
            "cigsPerDay": cigs_per_day,
            "BPMeds": 1 if data.get("bp_treatment") == "Yes" else 0,
            "prevalentStroke": 1 if data.get("previous_stroke") == "Yes" else 0,
            "prevalentHyp": prevalent_hyp,
            "diabetes": 1 if data.get("diabetes") == "Yes" else 0,
            "totChol": float(data.get("cholesterol_total", data.get("cholesterol", data.get("totChol", 200)))),
            "sysBP": sys_bp,
            "diaBP": dia_bp,
            "pulsePressure": sys_bp - dia_bp,
            "bp_ratio": sys_bp / (dia_bp + 1e-6),
            "BMI": float(data.get("bmi", data.get("BMI", 25))),
            "heartRate": float(data.get("heart_rate", data.get("heartRate", 72)) or 72),
            "glucose": float(data.get("glucose", 100)),
        }
        return pd.DataFrame([features])[TEN_YEAR_FEATURES]

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_loaded:
            self.load_models()

        if self.ten_year_model is None:
            return {"success": False, "message": "Ten-year CHD model is not loaded."}

        try:
            X = self.build_features(data)
            prob = float(self.ten_year_model.predict_proba(X)[0][1])
            pct = min(round(prob * 100, 1), 100.0)

            if prob < 0.40:
                category, colour = "LOW", "low"
            elif prob < 0.50:
                category, colour = "MEDIUM", "medium"
            else:
                category, colour = "HIGH", "high"

            advice = (
                "Your projected 10-year risk is low." if category == "LOW" else
                "Your projected 10-year risk is moderate. View personalised recommendations." if category == "MEDIUM" else
                "Your projected 10-year risk is high. Please view personalised recommendations."
            )
            return {
                "success": True,
                "percent": pct,
                "category": category,
                "colour": colour,
                "probability": round(prob, 4),
                "advice": advice,
            }
        except Exception as e:
            logging.exception("TenYearRiskEngine predict failed")
            return {"success": False, "message": str(e)}

ten_year_risk_engine = TenYearRiskEngine()
