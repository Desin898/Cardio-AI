import os
import pickle
import logging
from datetime import datetime
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import shap

from backend.app.core.config import settings
from backend.app.engines.base_engine import BaseMLEngine

SMOKER_SAFE_MAP = {"Yes": "Yes", "No": "No", "Former": "No"}

RECOMMENDATION_ADVICE = {
    "Low_Risk": {
        "title": "Low Cardiovascular Risk",
        "icon": "fas fa-check-circle",
        "colour": "success",
        "priority": "Low",
        "advice": (
            "Your cardiovascular risk profile is currently low. "
            "Keep up your healthy lifestyle — small consistent habits compound into "
            "long-term heart health."
        ),
        "actions": [
            "Schedule an annual cardiovascular screening",
            "Maintain a balanced diet rich in fruits, vegetables and whole grains",
            "Keep up regular physical activity (≥150 min/week)",
            "Avoid tobacco and limit alcohol intake",
        ],
    },
    "High_Risk": {
        "title": "High Cardiovascular Risk",
        "icon": "fas fa-exclamation-triangle",
        "colour": "danger",
        "priority": "High",
        "advice": (
            "Your profile indicates elevated cardiovascular risk. "
            "Immediate lifestyle modification and prompt medical consultation are "
            "strongly advised to reduce your risk of a cardiac event."
        ),
        "actions": [
            "Consult your cardiologist promptly",
            "Monitor blood pressure and cholesterol regularly",
            "Strictly adhere to any prescribed medications",
            "Adopt a heart-healthy diet and increase physical activity",
        ],
    },
    "Smoking_Cessation": {
        "title": "Smoking Cessation",
        "icon": "fas fa-smoking-ban",
        "colour": "warning",
        "priority": "High",
        "advice": (
            "Smoking is a major, modifiable cardiovascular risk factor. "
            "Quitting can reduce your heart disease risk by up to 50% within just one year."
        ),
        "actions": [
            "Speak to your doctor about a personalised cessation plan",
            "Consider nicotine replacement therapy (patch, gum, or inhaler)",
            "Explore prescription medications such as varenicline or bupropion",
            "Avoid secondhand smoke exposure and identify your personal triggers",
        ],
    },
    "Diet_Cholesterol": {
        "title": "Dietary & Cholesterol Management",
        "icon": "fas fa-apple-alt",
        "colour": "warning",
        "priority": "Medium",
        "advice": (
            "Your cholesterol or dietary indicators suggest that targeted nutritional "
            "changes are needed to protect your cardiovascular health long-term."
        ),
        "actions": [
            "Reduce saturated fats (fatty meats, full-fat dairy) and eliminate trans fats",
            "Increase fibre through oats, legumes, fruits and vegetables",
            "Limit processed foods, fried foods and added sugars",
            "Consider a referral to a registered dietitian for a tailored meal plan",
        ],
    },
    "Exercise": {
        "title": "Increase Physical Activity",
        "icon": "fas fa-running",
        "colour": "info",
        "priority": "Medium",
        "advice": (
            "Regular physical activity is one of the most powerful tools to reduce "
            "cardiovascular risk. Even modest increases in activity levels yield "
            "meaningful benefits."
        ),
        "actions": [
            "Aim for at least 150 minutes of moderate-intensity aerobic activity per week",
            "Start with 30-minute brisk walks 5 days a week if currently sedentary",
            "Add strength/resistance training at least twice weekly",
            "Break up prolonged sitting with movement every 30–60 minutes",
        ],
    },
    "BP_Control": {
        "title": "Blood Pressure Control",
        "icon": "fas fa-tachometer-alt",
        "colour": "danger",
        "priority": "High",
        "advice": (
            "Your blood pressure readings indicate hypertension management is needed. "
            "Uncontrolled high blood pressure is a leading cause of heart disease, "
            "stroke and kidney damage."
        ),
        "actions": [
            "Monitor your blood pressure at home daily and log the readings",
            "Reduce sodium intake to under 2,300 mg/day (ideally 1,500 mg)",
            "Follow the DASH diet (rich in potassium, calcium and magnesium)",
            "Take prescribed antihypertensive medications consistently",
        ],
    },
    "Maintenance": {
        "title": "Health Maintenance",
        "icon": "fas fa-shield-alt",
        "colour": "primary",
        "priority": "Low",
        "advice": (
            "Your current health metrics are within healthy ranges. "
            "The goal now is to sustain and protect these gains over the long term."
        ),
        "actions": [
            "Continue regular medical check-ups (at least annually)",
            "Maintain a healthy weight and stable BMI",
            "Stay consistent with your current exercise routine",
            "Manage stress through mindfulness, adequate sleep and social connection",
        ],
    },
}

_PRIORITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}

class PrescreeningEngine(BaseMLEngine):
    def __init__(self):
        self.risk_model = None
        self.scaler_base = None
        self.scaler_graph = None
        self.knn_graph = None
        self.new_encoders = None
        self.BASE_FEATURES = None
        self.FINAL_FEATURES = None
        self.degree_train = None
        self.clustering_train = None
        self.community_train = None
        self.mlb = None
        self.rec_multilabel_models = None
        self.xgb_rec_label_model = None
        self.risk_explainer = None
        self.is_loaded = False

    def _load_pkl(self, filename: str):
        path = settings.NEW_MODEL_DIR / filename
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self._fix_estimator(obj)
        return obj

    def _fix_estimator(self, est):
        if isinstance(est, dict):
            for v in est.values():
                self._fix_estimator(v)
            return
        if isinstance(est, (list, tuple)):
            for item in est:
                self._fix_estimator(item)
            return
        if hasattr(est, 'steps'):
            for _, step in est.steps:
                self._fix_estimator(step)
        if hasattr(est, 'named_steps'):
            for _, step in est.named_steps.items():
                self._fix_estimator(step)
        if hasattr(est, '__dict__'):
            if not hasattr(est, 'multi_class'):
                setattr(est, 'multi_class', 'auto')
            if not hasattr(est, 'keep_empty_features'):
                setattr(est, 'keep_empty_features', False)

    def load_models(self) -> None:
        if self.is_loaded:
            return
        try:
            self.risk_model = self._load_pkl("xgb_risk_model.pkl")
            self.scaler_base = self._load_pkl("scaler_base.pkl")
            self.scaler_graph = self._load_pkl("scaler_graph.pkl")
            self.knn_graph = self._load_pkl("knn_graph.pkl")
            self.new_encoders = self._load_pkl("encoders.pkl")
            self.BASE_FEATURES = self._load_pkl("feature_columns.pkl")
            self.FINAL_FEATURES = self._load_pkl("final_features.pkl")
            self.degree_train = self._load_pkl("degree_train.pkl")
            self.clustering_train = self._load_pkl("clustering_train.pkl")
            self.community_train = self._load_pkl("community_train.pkl")
            self.mlb = self._load_pkl("mlb.pkl")
            self.rec_multilabel_models = self._load_pkl("rec_multilabel_models.pkl")
            self.xgb_rec_label_model = self._load_pkl("xgb_rec_label_model.pkl")

            try:
                self.risk_explainer = shap.TreeExplainer(self.risk_model)
                logging.info("SHAP TreeExplainer initialised in PrescreeningEngine.")
            except Exception as e:
                self.risk_explainer = None
                logging.warning(f"SHAP explainer could not be initialised: {e}")

            self.is_loaded = True
            logging.info("PrescreeningEngine models loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load PrescreeningEngine models: {e}")
            raise e

    def _encode_safe(self, encoder, value, safe_map=None):
        if safe_map:
            value = safe_map.get(str(value), str(value))
        try:
            return int(encoder.transform([value])[0])
        except Exception:
            logging.warning(f"LabelEncoder could not encode '{value}'; defaulting to 0.")
            return 0

    def prepare_new_features(self, data: dict) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        if not self.is_loaded:
            self.load_models()

        age = float(data.get("age", 45))
        bmi = float(data.get("bmi", 25))
        systolic = float(data.get("systolic_bp", 120))
        diastolic = float(data.get("diastolic_bp", 80))
        cholesterol = float(data.get("cholesterol", 200))
        glucose = float(data.get("glucose", 100))
        exercise_hrs = float(data.get("exercise_hours", 0))

        high_chol = 1 if cholesterol > 200 else 0
        high_glucose = 1 if glucose > 100 else 0
        hypertension = 1 if (systolic >= 140 or diastolic >= 90) else 0
        obesity = 1 if bmi >= 30 else 0

        gender_enc = self._encode_safe(self.new_encoders["Gender"], data.get("gender", "Male"))
        smoker_enc = self._encode_safe(self.new_encoders["Smoker"], data.get("smoker", "No"), SMOKER_SAFE_MAP)

        base_dict = {
            "Age": age,
            "Gender_Encoded": gender_enc,
            "BMI": bmi,
            "Systolic_BP": systolic,
            "Diastolic_BP": diastolic,
            "Cholesterol mg/dL": cholesterol,
            "Glucose mg/dL": glucose,
            "Smoker_Encoded": smoker_enc,
            "High_Cholesterol": high_chol,
            "High_Glucose": high_glucose,
            "Hypertension": hypertension,
            "Obesity": obesity,
            "Exercise hours/week": exercise_hrs,
        }

        X_base = pd.DataFrame([base_dict])[self.BASE_FEATURES]
        X_base_scaled = self.scaler_base.transform(X_base)

        distances, indices = self.knn_graph.kneighbors(X_base_scaled)
        nbr = indices[0]
        degree_val = float(np.mean(self.degree_train[nbr]))
        clustering_val = float(np.mean(self.clustering_train[nbr]))
        community_val = float(np.mean(self.community_train[nbr]))

        full_dict = base_dict.copy()
        full_dict["degree"] = degree_val
        full_dict["clustering"] = clustering_val
        full_dict["community"] = community_val

        X_full = pd.DataFrame([full_dict])[self.FINAL_FEATURES]
        X_full_scaled = self.scaler_graph.transform(X_full)

        return X_base, X_full, X_full_scaled

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_loaded:
            self.load_models()

        X_base, X_full, X_full_scaled = self.prepare_new_features(data)
        prob = float(self.risk_model.predict_proba(X_full_scaled)[0][1])
        pred_class = int(self.risk_model.predict(X_full_scaled)[0])

        top_factors = []
        if self.risk_explainer is not None:
            try:
                shap_values = self.risk_explainer.shap_values(X_full_scaled)
                if isinstance(shap_values, list):
                    class_1_impacts = shap_values[1][0]
                elif shap_values.ndim == 3:
                    class_1_impacts = shap_values[0, :, 1]
                else:
                    class_1_impacts = shap_values[0]
                contributions = sorted(
                    [(self.FINAL_FEATURES[i], class_1_impacts[i], X_full.iloc[0][self.FINAL_FEATURES[i]])
                     for i in range(len(self.FINAL_FEATURES))],
                    key=lambda x: abs(x[1]), reverse=True,
                )
                for feature, impact, value in contributions[:3]:
                    effect = "higher risk" if impact > 0 else "lower risk"
                    display_value = value
                    if feature == "Gender_Encoded" and "Gender" in self.new_encoders:
                        try:
                            display_value = self.new_encoders["Gender"].inverse_transform([int(value)])[0]
                        except Exception:
                            pass
                    elif feature == "Smoker_Encoded" and "Smoker" in self.new_encoders:
                        try:
                            display_value = self.new_encoders["Smoker"].inverse_transform([int(value)])[0]
                        except Exception:
                            pass
                    top_factors.append({
                        "factor": feature.replace("_Encoded", "").replace("_", " "),
                        "impact": round(float(impact), 3),
                        "value": str(display_value),
                        "effect": effect,
                    })
            except Exception as shap_err:
                logging.warning(f"SHAP computation failed: {shap_err}")

        base = {
            "success": True,
            "risk_probability": prob,
            "explanation": top_factors,
            "patient_data": data,
        }

        if pred_class == 1:
            base.update({
                "risk_status": "HIGH_RISK",
                "decision": "MANDATORY_ECG",
                "message": "You show signs of elevated CAD risk. Please upload your ECG.",
                "next_step": "UPLOAD_ECG",
                "requires_ecg": True,
                "color": "red",
            })
        else:
            base.update({
                "risk_status": "LOW_RISK",
                "decision": "OPTIONAL_ECG",
                "message": "You do not currently show significant CAD risk.",
                "next_step": "OPTIONAL_UPLOAD",
                "requires_ecg": False,
                "color": "green",
            })

        return base

    def get_recommendations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_loaded:
            self.load_models()

        try:
            X_base, X_full, X_full_scaled = self.prepare_new_features(data)

            risk_prob = float(self.risk_model.predict_proba(X_full_scaled)[0][1])
            risk_class = "HIGH_RISK" if risk_prob >= 0.5 else "LOW_RISK"
            risk_pct = round(risk_prob * 100, 1)

            shap_top_factors = []
            if self.risk_explainer is not None:
                try:
                    shap_vals = self.risk_explainer.shap_values(X_full_scaled)
                    if isinstance(shap_vals, list):
                        class1_impacts = shap_vals[1][0]
                    elif shap_vals.ndim == 3:
                        class1_impacts = shap_vals[0, :, 1]
                    else:
                        class1_impacts = shap_vals[0]

                    contributions = sorted(
                        [(self.FINAL_FEATURES[i], class1_impacts[i], X_full.iloc[0][self.FINAL_FEATURES[i]])
                         for i in range(len(self.FINAL_FEATURES))],
                        key=lambda x: abs(x[1]), reverse=True,
                    )
                    for feature, impact, value in contributions[:5]:
                        display_value = value
                        if feature == "Gender_Encoded" and "Gender" in self.new_encoders:
                            try:
                                display_value = self.new_encoders["Gender"].inverse_transform([int(value)])[0]
                            except Exception:
                                pass
                        elif feature == "Smoker_Encoded" and "Smoker" in self.new_encoders:
                            try:
                                display_value = self.new_encoders["Smoker"].inverse_transform([int(value)])[0]
                            except Exception:
                                pass
                        shap_top_factors.append({
                            "feature": feature.replace("_Encoded", "").replace("_", " "),
                            "impact": round(float(impact), 3),
                            "direction": "increases" if impact > 0 else "decreases",
                            "value": display_value,
                        })
                except Exception as shap_err:
                    logging.warning(f"SHAP computation in recommendations failed: {shap_err}")

            distances, indices = self.knn_graph.kneighbors(self.scaler_base.transform(X_base))
            nbr = indices[0]
            graph_info = {
                "degree": round(float(np.mean(self.degree_train[nbr])), 4),
                "clustering": round(float(np.mean(self.clustering_train[nbr])), 4),
                "community": round(float(np.mean(self.community_train[nbr])), 4),
            }

            active_categories = []
            for category, clf in self.rec_multilabel_models.items():
                try:
                    pred = int(clf.predict(X_full_scaled)[0])
                    if pred == 1:
                        prob_val = None
                        if hasattr(clf, "predict_proba"):
                            try:
                                prob_val = round(float(clf.predict_proba(X_full_scaled)[0][1]), 4)
                            except Exception:
                                pass
                        active_categories.append((category, prob_val))
                except Exception as clf_err:
                    logging.warning(f"Classifier for '{category}' failed: {clf_err}")

            if not active_categories:
                try:
                    label_pred = self.xgb_rec_label_model.predict(X_full_scaled)[0]
                    decoded = self.mlb.inverse_transform(np.array([[label_pred]]))[0]
                    if decoded:
                        proba_dict = {}
                        if hasattr(self.xgb_rec_label_model, "predict_proba"):
                            try:
                                proba_row = self.xgb_rec_label_model.predict_proba(X_full_scaled)[0]
                                for idx, label in enumerate(self.mlb.classes_):
                                    proba_dict[label] = round(float(proba_row[idx]), 4)
                            except Exception:
                                pass
                        active_categories = [(cat, proba_dict.get(cat)) for cat in decoded]
                    else:
                        active_categories = [("Maintenance", None)]
                except Exception as fb_err:
                    logging.warning(f"XGBoost fallback failed: {fb_err}")
                    active_categories = [("Maintenance", None)]

            if not active_categories:
                active_categories = [("Maintenance", None)]

            recommendations = []
            for cat, prob_val in active_categories:
                if cat not in RECOMMENDATION_ADVICE:
                    continue
                advice_entry = RECOMMENDATION_ADVICE[cat]
                rationale = self._build_rationale(cat, data)
                rec = {
                    "category": cat,
                    "title": advice_entry["title"],
                    "icon": advice_entry["icon"],
                    "colour": advice_entry["colour"],
                    "priority": advice_entry["priority"],
                    "advice": advice_entry["advice"],
                    "actions": advice_entry["actions"],
                    "rationale": rationale,
                    "probability": prob_val,
                }
                recommendations.append(rec)

            if not recommendations:
                advice_entry = RECOMMENDATION_ADVICE["Maintenance"]
                recommendations.append({
                    "category": "Maintenance",
                    "title": advice_entry["title"],
                    "icon": advice_entry["icon"],
                    "colour": advice_entry["colour"],
                    "priority": advice_entry["priority"],
                    "advice": advice_entry["advice"],
                    "actions": advice_entry["actions"],
                    "rationale": "Your overall health indicators are within acceptable ranges.",
                    "probability": None,
                })

            recommendations.sort(key=lambda r: _PRIORITY_ORDER.get(r.get("priority", "Low"), 2))

            return {
                "success": True,
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "risk_assessment": {
                    "risk_probability": round(risk_prob, 4),
                    "risk_class": risk_class,
                    "risk_percentage": risk_pct,
                },
                "recommendations": recommendations,
                "total": len(recommendations),
                "shap_top_factors": shap_top_factors,
                "graph_features": graph_info,
            }
        except Exception as e:
            logging.exception("get_recommendations failed")
            return {"success": False, "message": str(e)}

    def _build_rationale(self, category: str, data: dict) -> str:
        cholesterol = float(data.get("cholesterol", 200))
        systolic = float(data.get("systolic_bp", 120))
        diastolic = float(data.get("diastolic_bp", 80))
        exercise_hrs = float(data.get("exercise_hours", 0))
        smoker = str(data.get("smoker", "No"))

        rationale_map = {
            "High_Risk": "Your overall risk profile indicates potential cardiovascular concerns requiring prompt medical attention.",
            "Low_Risk": "Your overall health indicators are within acceptable ranges.",
            "Smoking_Cessation": f"You are {'a current smoker' if smoker == 'Yes' else 'a former smoker'}, which significantly elevates cardiovascular risk.",
            "Diet_Cholesterol": f"Your cholesterol level ({cholesterol:.0f} mg/dL) " + ("is elevated above the 200 mg/dL threshold." if cholesterol > 200 else "warrants dietary vigilance."),
            "Exercise": f"Your current exercise level ({exercise_hrs:.1f} hours/week) is below the recommended 2.5 hours of moderate-intensity activity." if exercise_hrs < 2.5 else "Maintaining and building on your physical activity is recommended.",
            "BP_Control": f"Your blood pressure ({systolic:.0f}/{diastolic:.0f} mmHg) " + ("is in the hypertensive range." if systolic >= 140 or diastolic >= 90 else "is approaching hypertensive levels and should be monitored."),
            "Maintenance": "Your current health metrics are within healthy ranges. Sustaining your lifestyle habits is key.",
        }
        return rationale_map.get(category, "This recommendation is based on your overall health screening profile.")

prescreening_engine = PrescreeningEngine()
