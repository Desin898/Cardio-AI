import os
import pickle
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
import shap
import shap.explainers._tree as setree

from backend.app.core.config import settings
from backend.app.engines.base_engine import BaseMLEngine

# Patch SHAP UBJSON decoder for XGBoost 3.x compatibility
orig_decode = setree.decode_ubjson_buffer
def _patched_decode(fd):
    obj = orig_decode(fd)
    try:
        if "learner" in obj and "learner_model_param" in obj["learner"]:
            param = obj["learner"]["learner_model_param"]
            if "base_score" in param:
                bs = param["base_score"]
                if isinstance(bs, str) and bs.startswith("[") and bs.endswith("]"):
                    param["base_score"] = bs.strip("[]")
                elif isinstance(bs, list) and len(bs) > 0:
                    param["base_score"] = str(bs[0])
    except Exception:
        pass
    return obj
setree.decode_ubjson_buffer = _patched_decode

CKM_FEATURES = [
    "age", "gender", "systolic_bp", "diastolic_bp", "bmi", "current_smoker",
    "hba1c", "hs_troponin", "egfr", "cholesterol_total", "cholesterol_hdl", "family_history_cad"
]

FEATURE_DISPLAY_NAMES = {
    "age": "Age",
    "gender": "Gender",
    "systolic_bp": "Systolic BP",
    "diastolic_bp": "Diastolic BP",
    "bmi": "BMI",
    "current_smoker": "Current Smoker",
    "hba1c": "HbA1c",
    "hs_troponin": "hs-Troponin",
    "egfr": "eGFR",
    "cholesterol_total": "Total Cholesterol",
    "cholesterol_hdl": "HDL Cholesterol",
    "family_history_cad": "Family History of CAD"
}

RECOMMENDATION_ADVICE = {
    "Low_Risk": {
        "title": "Low CKM Cardiovascular Risk",
        "icon": "fas fa-check-circle",
        "colour": "success",
        "priority": "Low",
        "advice": (
            "Your CKM risk profile is currently low (<0.30). "
            "Continue maintaining your healthy lifestyle and schedule routine annual check-ups."
        ),
        "actions": [
            "Schedule an annual CKM risk screening",
            "Maintain a balanced Mediterranean or DASH diet",
            "Keep up regular physical activity (≥150 min/week)",
            "Avoid tobacco and maintain normal blood pressure",
        ],
    },
    "Moderate_Risk": {
        "title": "Moderate CKM Cardiovascular Risk",
        "icon": "fas fa-exclamation-circle",
        "colour": "warning",
        "priority": "Medium",
        "advice": (
            "Your profile indicates moderate CKM risk (0.30–0.65). "
            "Targeted lifestyle modification and secondary diagnostic testing (Echo / Stress test) are recommended."
        ),
        "actions": [
            "Schedule a cardiac stress test and echocardiogram",
            "Consult a cardiologist for preventive management",
            "Optimize glycemic control and lipid profile",
            "Monitor blood pressure daily at home",
        ],
    },
    "High_Risk": {
        "title": "High CKM Cardiovascular Risk",
        "icon": "fas fa-exclamation-triangle",
        "colour": "danger",
        "priority": "High",
        "advice": (
            "Your profile indicates high CKM risk (>0.65). "
            "Urgent cardiology consultation, ECG, and aggressive risk factor reduction are strongly advised."
        ),
        "actions": [
            "Urgent cardiology referral for diagnostic workup",
            "Perform 12-lead ECG and comprehensive biomarker panel",
            "Strictly adhere to prescribed cardiovascular/metabolic medications",
            "Implement immediate structured lifestyle modification",
        ],
    },
}

_PRIORITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}

class PrescreeningEngine(BaseMLEngine):
    def __init__(self):
        self.calibrated_model: Optional[CalibratedClassifierCV] = None
        self.base_xgb_model: Optional[xgb.XGBClassifier] = None
        self.risk_explainer: Optional[shap.TreeExplainer] = None
        self.is_loaded: bool = False

    def train_and_save_ckm_model(self) -> Tuple[CalibratedClassifierCV, xgb.XGBClassifier]:
        logging.info("Training upgraded enterprise CKM XGBoost model with Platt Scaling...")
        np.random.seed(42)
        n_samples = 2000

        age = np.random.randint(25, 85, n_samples)
        gender = np.random.choice([0, 1], n_samples)
        systolic_bp = np.random.normal(130, 22, n_samples)
        diastolic_bp = np.random.normal(82, 12, n_samples)
        bmi = np.random.normal(28, 6, n_samples)
        current_smoker = np.random.choice([0, 1], n_samples, p=[0.72, 0.28])

        hba1c = np.random.normal(6.2, 1.4, n_samples)
        mask_hba1c = np.random.rand(n_samples) < 0.25
        hba1c[mask_hba1c] = np.nan

        hs_troponin = np.random.normal(14, 12, n_samples)
        mask_trop = np.random.rand(n_samples) < 0.35
        hs_troponin[mask_trop] = np.nan

        egfr = np.random.normal(82, 22, n_samples)
        mask_egfr = np.random.rand(n_samples) < 0.20
        egfr[mask_egfr] = np.nan

        cholesterol_total = np.random.normal(215, 42, n_samples)
        cholesterol_hdl = np.random.normal(48, 14, n_samples)
        family_history_cad = np.random.choice([0, 1], n_samples, p=[0.75, 0.25])

        risk_score = (
            0.045 * (age - 50) +
            0.35 * gender +
            0.035 * (systolic_bp - 120) +
            0.03 * (bmi - 25) +
            0.6 * current_smoker +
            0.45 * (np.nan_to_num(hba1c, nan=6.0) - 5.7) +
            0.06 * (np.nan_to_num(hs_troponin, nan=10.0) - 10) -
            0.025 * (np.nan_to_num(egfr, nan=90.0) - 90) +
            0.012 * (cholesterol_total - 200) -
            0.025 * (cholesterol_hdl - 50) +
            0.65 * family_history_cad - 1.8
        )
        prob_true = 1 / (1 + np.exp(-risk_score))
        y = (prob_true > 0.5).astype(int)

        X = pd.DataFrame({
            "age": age,
            "gender": gender,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "bmi": bmi,
            "current_smoker": current_smoker,
            "hba1c": hba1c,
            "hs_troponin": hs_troponin,
            "egfr": egfr,
            "cholesterol_total": cholesterol_total,
            "cholesterol_hdl": cholesterol_hdl,
            "family_history_cad": family_history_cad
        })[CKM_FEATURES]

        base_xgb = xgb.XGBClassifier(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.04,
            missing=np.nan,
            eval_metric="logloss",
            random_state=42
        )
        base_xgb.fit(X, y)

        calibrated = CalibratedClassifierCV(estimator=base_xgb, method="sigmoid", cv=3)
        calibrated.fit(X, y)

        os.makedirs(settings.NEW_MODEL_DIR, exist_ok=True)
        calibrated_path = settings.NEW_MODEL_DIR / "ckm_xgb_calibrated.pkl"
        base_path = settings.NEW_MODEL_DIR / "ckm_xgb_base.pkl"

        with open(calibrated_path, "wb") as f:
            pickle.dump(calibrated, f)
        with open(base_path, "wb") as f:
            pickle.dump(base_xgb, f)

        logging.info("CKM XGBoost model trained and saved successfully.")
        return calibrated, base_xgb

    def load_models(self) -> None:
        if self.is_loaded:
            return

        calibrated_path = settings.NEW_MODEL_DIR / "ckm_xgb_calibrated.pkl"
        base_path = settings.NEW_MODEL_DIR / "ckm_xgb_base.pkl"

        try:
            if calibrated_path.exists() and base_path.exists():
                with open(calibrated_path, "rb") as f:
                    self.calibrated_model = pickle.load(f)
                with open(base_path, "rb") as f:
                    self.base_xgb_model = pickle.load(f)
            else:
                self.calibrated_model, self.base_xgb_model = self.train_and_save_ckm_model()

            if self.base_xgb_model is not None:
                try:
                    self.risk_explainer = shap.TreeExplainer(self.base_xgb_model)
                    logging.info("SHAP TreeExplainer initialized for CKM XGBoost model.")
                except Exception as ex:
                    logging.warning(f"SHAP TreeExplainer init warning: {ex}")
                    self.risk_explainer = None

            self.is_loaded = True
            logging.info("PrescreeningEngine CKM models loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load PrescreeningEngine CKM models: {e}")
            # Fallback to retraining in memory
            self.calibrated_model, self.base_xgb_model = self.train_and_save_ckm_model()
            if self.base_xgb_model is not None:
                self.risk_explainer = shap.TreeExplainer(self.base_xgb_model)
            self.is_loaded = True

    def generate_shap_explanation(self, df_input: pd.DataFrame) -> List[Dict[str, Any]]:
        if self.risk_explainer is None:
            if self.base_xgb_model is not None:
                self.risk_explainer = shap.TreeExplainer(self.base_xgb_model)
            else:
                return []

        try:
            shap_vals = self.risk_explainer.shap_values(df_input)
            if isinstance(shap_vals, list):
                row_vals = shap_vals[1][0]
            elif shap_vals.ndim == 2:
                row_vals = shap_vals[0]
            elif shap_vals.ndim == 3:
                row_vals = shap_vals[0, :, 1]
            else:
                row_vals = shap_vals[0]

            drivers = []
            feature_names = list(df_input.columns)
            for idx, feat in enumerate(feature_names):
                val = df_input.iloc[0][feat]
                imp = float(row_vals[idx])
                pct_val = imp * 100
                impact_str = f"+{pct_val:.1f}%" if pct_val > 0 else f"{pct_val:.1f}%"
                direction = "risk_increasing" if imp > 0 else "risk_decreasing"

                display_name = FEATURE_DISPLAY_NAMES.get(feat, feat)
                display_val = None if pd.isna(val) else float(val)
                if feat == "gender":
                    display_val = "Male" if val == 1 else "Female"
                elif feat in ("current_smoker", "family_history_cad"):
                    display_val = True if val == 1 else False

                drivers.append({
                    "feature": display_name,
                    "value": display_val,
                    "impact": impact_str,
                    "direction": direction,
                    "_abs_imp": abs(imp)
                })

            drivers.sort(key=lambda x: x["_abs_imp"], reverse=True)
            for d in drivers:
                del d["_abs_imp"]

            return drivers
        except Exception as e:
            logging.warning(f"generate_shap_explanation failed: {e}")
            return []

    def prepare_input_dataframe(self, data: Dict[str, Any]) -> pd.DataFrame:
        age = float(data.get("age", 45))
        g_raw = str(data.get("gender", "Male")).lower()
        gender = 1.0 if (g_raw.startswith("m") or g_raw == "1") else 0.0

        systolic = float(data.get("systolic_bp", 120))
        diastolic = float(data.get("diastolic_bp", 80))
        bmi = float(data.get("bmi", 25))

        smoker_val = data.get("current_smoker")
        if smoker_val is None:
            smoker_val = str(data.get("smoker", "")).lower() in ("yes", "true", "1")
        current_smoker = 1.0 if bool(smoker_val) else 0.0

        hba1c = float(data["hba1c"]) if data.get("hba1c") is not None else np.nan
        hs_troponin = float(data["hs_troponin"]) if data.get("hs_troponin") is not None else np.nan
        egfr = float(data["egfr"]) if data.get("egfr") is not None else np.nan

        chol_total = float(data.get("cholesterol_total", data.get("cholesterol", 200)))
        chol_hdl = float(data.get("cholesterol_hdl", 50))
        fam_history = 1.0 if bool(data.get("family_history_cad")) else 0.0

        dict_row = {
            "age": age,
            "gender": gender,
            "systolic_bp": systolic,
            "diastolic_bp": diastolic,
            "bmi": bmi,
            "current_smoker": current_smoker,
            "hba1c": hba1c,
            "hs_troponin": hs_troponin,
            "egfr": egfr,
            "cholesterol_total": chol_total,
            "cholesterol_hdl": chol_hdl,
            "family_history_cad": fam_history
        }

        return pd.DataFrame([dict_row])[CKM_FEATURES]

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_loaded:
            self.load_models()

        df_input = self.prepare_input_dataframe(data)

        if self.calibrated_model is not None:
            prob = float(self.calibrated_model.predict_proba(df_input)[0][1])
        elif self.base_xgb_model is not None:
            prob = float(self.base_xgb_model.predict_proba(df_input)[0][1])
        else:
            prob = 0.25

        prob_percentage = round(prob * 100, 1)

        if prob < 0.30:
            risk_category = "LOW"
            recommended_next_path = "ROUTINE_MONITORING_LIFESTYLE"
            decision = "OPTIONAL_ECG"
            message = "Low CKM risk (<0.30). Routine monitoring and healthy lifestyle recommended."
            requires_ecg = False
            color = "green"
        elif prob <= 0.65:
            risk_category = "MODERATE"
            recommended_next_path = "CARDIAC_STRESS_TEST_ECHO"
            decision = "RECOMMENDED_ECG"
            message = "Moderate CKM risk (0.30–0.65). Cardiac stress testing & echocardiogram recommended."
            requires_ecg = True
            color = "yellow"
        else:
            risk_category = "HIGH"
            recommended_next_path = "URGENT_CARDIOLOGY_REFERRAL_ECG"
            decision = "MANDATORY_ECG"
            message = "High CKM risk (>0.65). Urgent cardiology referral & 12-lead ECG required."
            requires_ecg = True
            color = "red"

        shap_breakdown = self.generate_shap_explanation(df_input)

        return {
            "success": True,
            "risk_category": risk_category,
            "risk_probability": round(prob, 4),
            "probability_percentage": prob_percentage,
            "shap_breakdown": shap_breakdown,
            "recommended_next_path": recommended_next_path,

            # Backward compatibility fields
            "risk_status": f"{risk_category}_RISK",
            "decision": decision,
            "message": message,
            "explanation": shap_breakdown,
            "patient_data": data,
            "requires_ecg": requires_ecg,
            "color": color,
        }

    def get_recommendations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        pred_res = self.predict(data)
        category = pred_res["risk_category"]

        rec_key = "Low_Risk" if category == "LOW" else ("Moderate_Risk" if category == "MODERATE" else "High_Risk")
        advice_info = RECOMMENDATION_ADVICE[rec_key]

        rec_item = {
            "category": category,
            "title": advice_info["title"],
            "icon": advice_info["icon"],
            "colour": advice_info["colour"],
            "priority": advice_info["priority"],
            "advice": advice_info["advice"],
            "actions": advice_info["actions"],
            "rationale": f"Based on enterprise CKM risk model calculation (Probability: {pred_res['probability_percentage']}%).",
            "probability": pred_res["risk_probability"],
        }

        return {
            "success": True,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "risk_assessment": {
                "risk_probability": pred_res["risk_probability"],
                "risk_class": pred_res["risk_category"],
                "risk_percentage": pred_res["probability_percentage"],
            },
            "recommendations": [rec_item],
            "total": 1,
            "shap_top_factors": pred_res["shap_breakdown"][:5],
            "graph_features": {},
        }

prescreening_engine = PrescreeningEngine()
