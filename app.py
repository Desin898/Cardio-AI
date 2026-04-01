import os
import sys
import json
import logging
import tempfile
import pickle
import subprocess
import threading
import time
import numpy as np
import pandas as pd
import shap
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_file
import uuid
from flask import send_from_directory
from werkzeug.utils import secure_filename
from Preprocessing.Angiogram_Preprocessing.Angiogram_DICOM_KeyFrame_Extraction import process_angiogram


#  ECG PREPROCESSING FUNCTIONS

TARGET_LEAD_LENGTH = 737

def preprocess_step1_image(img):
    if img is None:
        return None
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    thr = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 35, 7
    )
    return thr

def crop_12_leads_from_gray(gray_img):
    h, w        = gray_img.shape[:2]
    top_margin  = int(0.18 * h)
    lead_height = int((h - top_margin) / 4)
    lead_width  = int(w / 3)
    leads = {}
    idx = 1
    for row in range(4):
        for col in range(3):
            y1 = top_margin + row * lead_height
            y2 = top_margin + (row + 1) * lead_height
            x1 = col * lead_width
            x2 = (col + 1) * lead_width
            leads[f"Lead_{idx}"] = gray_img[y1:y2, x1:x2].copy()
            idx += 1
    return leads

def clean_lead_for_signal(lead_img):
    gray   = lead_img if len(lead_img.shape) == 2 else cv2.cvtColor(lead_img, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 41, 5)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

def extract_signal(clean_img):
    clean  = clean_img.astype(np.uint8)
    h, w   = clean.shape
    signal = []
    for x in range(w):
        pts = np.where(clean[:, x] > 0)[0]
        signal.append(int(np.mean(pts)) if len(pts) > 0 else (signal[-1] if signal else h // 2))
    sig = np.array(signal, dtype=float)
    if sig.max() - sig.min() < 1e-8:
        return np.zeros_like(sig)
    return (sig - sig.min()) / (sig.max() - sig.min())

def resample_signal(sig, target_len):
    if len(sig) == target_len:
        return sig
    if len(sig) == 0:
        return np.zeros(target_len, dtype=float)
    return np.interp(
        np.linspace(0, 1, target_len),
        np.linspace(0, 1, len(sig)),
        sig,
    )

def process_single_ecg_image(img_path, target_len=TARGET_LEAD_LENGTH,
                              save_images=False, save_folder=None):
    p   = Path(img_path)
    bgr = cv2.imread(str(p))
    if bgr is None:
        print("WARN: cannot read", img_path)
        return None
    thr = preprocess_step1_image(bgr)
    if save_images and save_folder:
        out_dir = Path(save_folder) / p.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "step1_threshold.png"), thr)
    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    leads = crop_12_leads_from_gray(gray)
    all_lead_signals = []
    for i in range(1, 13):
        lead   = leads[f"Lead_{i}"]
        clean  = clean_lead_for_signal(lead)
        sig    = extract_signal(clean)
        sig_rs = resample_signal(sig, target_len)
        all_lead_signals.append(sig_rs)
        if save_images and save_folder:
            cv2.imwrite(str(out_dir / f"clean_lead_{i}.png"), clean)
    return np.concatenate(all_lead_signals, axis=0)


#  RETRIEVAL MANAGER

from Pipeline_Management.metadata_system import EncryptionManager, MetadataManager

class RetrievalManager:
    @staticmethod
    def get_decrypted_ecg_png(metadata_path: Path) -> str:
        metadata  = MetadataManager.read_metadata(metadata_path)
        enc_path  = metadata.get("ecg", {}).get("raw_image_path")
        if enc_path is None:
            raise ValueError("No ECG image found in metadata.")
        enc             = EncryptionManager()
        decrypted_bytes = enc.decrypt_file(Path(enc_path))
        img_array       = np.frombuffer(decrypted_bytes, dtype=np.uint8)
        img             = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        temp_path       = Path(enc_path).with_suffix(".png")
        cv2.imwrite(str(temp_path), img)
        return str(temp_path)


#  ECG INFERENCE

INFERENCE_SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__),
    "process_02_cardiac_analysis", "ecg_risk_prediction",
    "inference", "predict_ecg_risk.py",
)

def run_ecg_inference(csv_path: str) -> dict:
    project_root = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isfile(INFERENCE_SCRIPT_PATH):
        msg = f"Inference script not found at: {INFERENCE_SCRIPT_PATH}"
        logging.error(msg)
        return {"error": msg, "prediction": "Error", "confidence": "0%",
                "risk_level": "UNKNOWN", "suspected_vessel": "N/A"}
    if not os.path.isfile(csv_path):
        msg = f"Pre-processed CSV not found at: {csv_path}"
        logging.error(msg)
        return {"error": msg, "prediction": "Error", "confidence": "0%",
                "risk_level": "UNKNOWN", "suspected_vessel": "N/A"}
    try:
        result = subprocess.run(
            [sys.executable, INFERENCE_SCRIPT_PATH, csv_path],
            capture_output=True, text=True, timeout=120, cwd=project_root,
        )
        if result.stdout:
            logging.info(f"Inference stdout:\n{result.stdout}")
        if result.stderr:
            logging.error(f"Inference stderr:\n{result.stderr}")
        if result.returncode != 0:
            stderr_clean = result.stderr.strip() or "No stderr captured."
            return {"error": stderr_clean, "prediction": "Error", "confidence": "0%",
                    "risk_level": "UNKNOWN", "suspected_vessel": "N/A"}
        prediction_result = {
            "prediction": "Unknown", "confidence": "0%",
            "risk_level": "UNKNOWN", "suspected_vessel": "N/A", "plot_path": None,
        }
        for line in result.stdout.splitlines():
            line = line.strip()
            if   line.startswith("Status:"):              prediction_result["prediction"]       = line.split(":", 1)[1].strip()
            elif line.startswith("Confidence:"):          prediction_result["confidence"]       = line.split(":", 1)[1].strip()
            elif line.startswith("Emergency Priority:"):  prediction_result["risk_level"]       = line.split(":", 1)[1].strip()
            elif line.startswith("Artery Localization:"): prediction_result["suspected_vessel"] = line.split(":", 1)[1].strip()
        return prediction_result
    except subprocess.TimeoutExpired:
        return {"error": "Inference script timed out (>120s).", "prediction": "Timeout",
                "confidence": "0%", "risk_level": "UNKNOWN", "suspected_vessel": "N/A"}
    except Exception as e:
        return {"error": f"Failed to launch inference: {str(e)}", "prediction": "Error",
                "confidence": "0%", "risk_level": "UNKNOWN", "suspected_vessel": "N/A"}


# ============================================================
#  FLASK APPLICATION
# ============================================================

app = Flask(__name__)

APP_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = "Pipeline_Management/Patients"

#  NEW GRAPH-AUGMENTED RISK PIPELINE  (Model/models/)

NEW_MODEL_DIR = os.path.join(APP_DIR, "Model", "models")

def _load_pkl(filename):
    path = os.path.join(NEW_MODEL_DIR, filename)
    with open(path, "rb") as f:
        return pickle.load(f)

risk_model         = _load_pkl("xgb_risk_model.pkl")
scaler_base        = _load_pkl("scaler_base.pkl")
scaler_graph       = _load_pkl("scaler_graph.pkl")
knn_graph          = _load_pkl("knn_graph.pkl")
new_encoders       = _load_pkl("encoders.pkl")        # {'Gender': LE, 'Smoker': LE}
BASE_FEATURES      = _load_pkl("feature_columns.pkl") # list of 13 feature names
FINAL_FEATURES     = _load_pkl("final_features.pkl")  # list of 16 feature names
degree_train       = _load_pkl("degree_train.pkl")    # ndarray (543,)
clustering_train   = _load_pkl("clustering_train.pkl")
community_train    = _load_pkl("community_train.pkl")
mlb                = _load_pkl("mlb.pkl")
rec_multilabel_models = _load_pkl("rec_multilabel_models.pkl")
xgb_rec_label_model   = _load_pkl("xgb_rec_label_model.pkl")

try:
    risk_explainer = shap.TreeExplainer(risk_model)
    logging.info("✅ SHAP TreeExplainer initialised for risk model.")
except Exception as e:
    risk_explainer = None
    logging.warning(f"⚠  SHAP explainer could not be initialised: {e}")

logging.info(
    f"✅ Graph-augmented pipeline loaded — "
    f"base features: {len(BASE_FEATURES)}, final features: {len(FINAL_FEATURES)}"
)

# Smoker encoder was trained on Yes/No; map Former → No for compatibility
SMOKER_SAFE_MAP = {"Yes": "Yes", "No": "No", "Former": "No"}


#  RECOMMENDATION ADVICE CATALOGUE

RECOMMENDATION_ADVICE = {
    "Low_Risk": {
        "title":   "Low Cardiovascular Risk",
        "icon":    "fas fa-check-circle",
        "colour":  "success",
        "advice":  (
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
        "title":   "High Cardiovascular Risk",
        "icon":    "fas fa-exclamation-triangle",
        "colour":  "danger",
        "advice":  (
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
        "title":   "Smoking Cessation",
        "icon":    "fas fa-smoking-ban",
        "colour":  "warning",
        "advice":  (
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
        "title":   "Dietary & Cholesterol Management",
        "icon":    "fas fa-apple-alt",
        "colour":  "warning",
        "advice":  (
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
        "title":   "Increase Physical Activity",
        "icon":    "fas fa-running",
        "colour":  "info",
        "advice":  (
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
        "title":   "Blood Pressure Control",
        "icon":    "fas fa-tachometer-alt",
        "colour":  "danger",
        "advice":  (
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
        "title":   "Health Maintenance",
        "icon":    "fas fa-shield-alt",
        "colour":  "primary",
        "advice":  (
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


#  GRAPH-AUGMENTED FEATURE PIPELINE

def _encode_safe(encoder, value, safe_map=None):
    """Encode a categorical value, with optional remapping for unseen labels."""
    if safe_map:
        value = safe_map.get(str(value), str(value))
    try:
        return int(encoder.transform([value])[0])
    except Exception:
        # Fall back to 0 if completely unknown
        logging.warning(f"LabelEncoder could not encode value '{value}'; defaulting to 0.")
        return 0


def prepare_new_features(data: dict):
    """
    Build base (13) and full (16) feature vectors for the graph-augmented pipeline.

    Returns:
        X_base        — pd.DataFrame (1 × 13), unscaled base features
        X_full        — pd.DataFrame (1 × 16), unscaled full features (with graph cols)
        X_full_scaled — np.ndarray   (1 × 16), scaler_graph-scaled full features
    """
    # --- raw clinical values ---
    age          = float(data.get("age", 45))
    bmi          = float(data.get("bmi", 25))
    systolic     = float(data.get("systolic_bp", 120))
    diastolic    = float(data.get("diastolic_bp", 80))
    cholesterol  = float(data.get("cholesterol", 200))
    glucose      = float(data.get("glucose", 100))
    exercise_hrs = float(data.get("exercise_hours", 0))

    # --- derived binary flags ---
    high_chol    = 1 if cholesterol > 200 else 0
    high_glucose = 1 if glucose > 100 else 0
    hypertension = 1 if (systolic >= 140 or diastolic >= 90) else 0
    obesity      = 1 if bmi >= 30 else 0

    # --- categorical encodings ---
    gender_enc = _encode_safe(new_encoders["Gender"], data.get("gender", "Male"))
    smoker_enc = _encode_safe(new_encoders["Smoker"],  data.get("smoker", "No"), SMOKER_SAFE_MAP)

    base_dict = {
        "Age":                age,
        "Gender_Encoded":     gender_enc,
        "BMI":                bmi,
        "Systolic_BP":        systolic,
        "Diastolic_BP":       diastolic,
        "Cholesterol mg/dL":  cholesterol,
        "Glucose mg/dL":      glucose,
        "Smoker_Encoded":     smoker_enc,
        "High_Cholesterol":   high_chol,
        "High_Glucose":       high_glucose,
        "Hypertension":       hypertension,
        "Obesity":            obesity,
        "Exercise hours/week": exercise_hrs,
    }

    # --- scale base features ---
    X_base       = pd.DataFrame([base_dict])[BASE_FEATURES]
    X_base_scaled = scaler_base.transform(X_base)

    # --- compute graph features by averaging k-nearest training neighbours ---
    distances, indices = knn_graph.kneighbors(X_base_scaled)
    nbr = indices[0]
    degree_val     = float(np.mean(degree_train[nbr]))
    clustering_val = float(np.mean(clustering_train[nbr]))
    community_val  = float(np.mean(community_train[nbr]))

    # --- build full 16-feature dict ---
    full_dict = base_dict.copy()
    full_dict["degree"]     = degree_val
    full_dict["clustering"] = clustering_val
    full_dict["community"]  = community_val

    X_full        = pd.DataFrame([full_dict])[FINAL_FEATURES]
    X_full_scaled = scaler_graph.transform(X_full)

    return X_base, X_full, X_full_scaled


#  TEN-YEAR CHD MODEL

TEN_YEAR_MODEL_DIR = os.path.join(APP_DIR, "ten_year_models/ten_year_models")

TEN_YEAR_FEATURES = [
    "age", "male", "currentSmoker", "cigsPerDay",
    "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes",
    "totChol", "sysBP", "diaBP", "pulsePressure", "bp_ratio",
    "BMI", "heartRate", "glucose",
]

_ten_year_model_path     = os.path.join(TEN_YEAR_MODEL_DIR, "ten_year_model.pkl")
_ten_year_threshold_path = os.path.join(TEN_YEAR_MODEL_DIR, "ten_year_threshold.pkl")

try:
    with open(_ten_year_model_path, "rb") as f:
        ten_year_model = pickle.load(f)
    with open(_ten_year_threshold_path, "rb") as f:
        ten_year_threshold = float(pickle.load(f))
    logging.info(f"✅ Ten-year CHD model loaded (threshold={ten_year_threshold:.4f})")
except FileNotFoundError as e:
    ten_year_model     = None
    ten_year_threshold = 0.5
    logging.warning(f"⚠  Ten-year CHD model NOT loaded: {e}")
except Exception as e:
    ten_year_model     = None
    ten_year_threshold = 0.5
    logging.error(f"❌ Failed to load ten-year CHD model: {e}")


#  DOCTOR CREDENTIALS

VALID_DOCTORS = {
    "doctor1": "pass123",
    "admin":   "admin",
}

# Standalone angiogram demo dirs
UPLOAD_FOLDER = Path("uploads")
OUTPUT_ROOT   = Path("Preprocessed_Angiogram_Output")
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_ROOT.mkdir(exist_ok=True)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["OUTPUT_ROOT"]   = str(OUTPUT_ROOT)

# DeepSA (Gradio) settings
DEEPSA_PORT   = int(os.environ.get("DEEPSA_PORT", 7860))
DEEPSA_URL    = os.environ.get("DEEPSA_URL", f"http://127.0.0.1:{DEEPSA_PORT}")
DEEPSA_SCRIPT = os.path.join(APP_DIR, "demo.py")


#  DeepSA PROCESS MANAGEMENT

_deepsa_proc: subprocess.Popen | None = None
_deepsa_lock = threading.Lock()


def _deepsa_running() -> bool:
    import socket
    try:
        with socket.create_connection(("127.0.0.1", DEEPSA_PORT), timeout=1):
            return True
    except OSError:
        return False


def ensure_deepsa_running():
    global _deepsa_proc
    with _deepsa_lock:
        if _deepsa_running():
            return
        logging.info(f"Starting DeepSA from: {DEEPSA_SCRIPT}")
        _deepsa_proc = subprocess.Popen(
            [sys.executable, DEEPSA_SCRIPT],
            cwd=os.path.dirname(DEEPSA_SCRIPT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    deadline = time.time() + 30
    while time.time() < deadline:
        if _deepsa_running():
            return
        time.sleep(0.5)
    raise RuntimeError(
        "DeepSA (demo.py) failed to start within 30 seconds. "
        "Check that the model checkpoint exists and all dependencies are installed."
    )


#  TEN-YEAR CHD HELPERS

def _build_ten_year_features(data: dict) -> pd.DataFrame:
    sys_bp     = float(data.get("systolic_bp", 120))
    dia_bp     = float(data.get("diastolic_bp", 80))
    smoker_val = str(data.get("smoker", "No"))

    is_current_smoker = 1 if smoker_val == "Yes" else 0

    cigs_per_day = 0.0
    if smoker_val in ("Yes", "Former"):
        cigs_per_day = float(data.get("cigs_per_day", data.get("cigsPerDay", 0)) or 0)

    prevalent_hyp_val = data.get("prevalent_hyp", "")
    if prevalent_hyp_val == "Yes":
        prevalent_hyp = 1
    elif prevalent_hyp_val == "No":
        prevalent_hyp = 0
    else:
        prevalent_hyp = 1 if (sys_bp >= 140 or dia_bp >= 90) else 0

    features = {
        "age":             float(data.get("age", 45)),
        "male":            1 if str(data.get("gender", "")).strip().lower() == "male" else 0,
        "currentSmoker":   is_current_smoker,
        "cigsPerDay":      cigs_per_day,
        "BPMeds":          1 if data.get("bp_treatment") == "Yes" else 0,
        "prevalentStroke": 1 if data.get("previous_stroke") == "Yes" else 0,
        "prevalentHyp":    prevalent_hyp,
        "diabetes":        1 if data.get("diabetes") == "Yes" else 0,
        "totChol":         float(data.get("cholesterol", data.get("totChol", 200))),
        "sysBP":           sys_bp,
        "diaBP":           dia_bp,
        "pulsePressure":   sys_bp - dia_bp,
        "bp_ratio":        sys_bp / (dia_bp + 1e-6),
        "BMI":             float(data.get("bmi", data.get("BMI", 25))),
        "heartRate":       float(data.get("heart_rate", data.get("heartRate", 72)) or 72),
        "glucose":         float(data.get("glucose", 100)),
    }
    return pd.DataFrame([features])[TEN_YEAR_FEATURES]


def _compute_ten_year(data: dict) -> dict:
    if ten_year_model is None:
        return {
            "success": False,
            "message": (
                "Ten-year CHD model is not loaded. "
                "Run xx.ipynb, place the pkl files in ten_year_models/, "
                "then restart Flask."
            ),
        }
    try:
        X    = _build_ten_year_features(data)
        prob = float(ten_year_model.predict_proba(X)[0][1])
        pct  = min(round(prob * 100, 1), 100.0)

        if prob < 0.40:
            category, colour = "LOW", "low"
        elif prob < 0.50:
            category, colour = "MEDIUM", "medium"
        else:
            category, colour = "HIGH", "high"

        advice = (
            "Your projected 10-year risk is low."
            if category == "LOW" else
            "Your projected 10-year risk is moderate. View personalised recommendations."
            if category == "MEDIUM" else
            "Your projected 10-year risk is high. Please view personalised recommendations."
        )
        return {
            "success":     True,
            "percent":     pct,
            "category":    category,
            "colour":      colour,
            "probability": round(prob, 4),
            "advice":      advice,
        }
    except Exception as e:
        logging.exception("_compute_ten_year failed")
        return {"success": False, "message": str(e)}



#  ROUTES — General / Patient


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/patient_login.html")
def patient_login():
    return render_template("patient_login.html")

@app.route("/patient_register.html")
def patient_register():
    return render_template("patient_register.html")

@app.route("/patient_dashboard.html")
def patient_dashboard():
    return render_template("patient_dashboard.html")

@app.route("/upload_ecg.html", methods=["GET"])
def upload_page():
    return render_template("upload_ecg.html")

@app.route("/pre_screening.html")
def screening():
    return render_template("pre_screening.html")

@app.route("/pre_screening_results.html")
def result():
    return render_template("pre_screening_results.html")



#  ROUTES — Doctor portal stubs


@app.route("/doctor_upload.html")
def doctor_upload():
    return render_template("doctor_upload.html")

@app.route("/doctor_analysis_results.html")
def doctor_analysis_results():
    return render_template("doctor_analysis_results.html")



#  ROUTES — Doctor login


@app.route("/doctor_login.html")
@app.route("/doctor_login", methods=["GET"])
def doctor_login():
    session_id = request.args.get("session_id", "")
    patient_id = request.args.get("patient_id", "")
    return render_template("doctor_login.html", session_id=session_id,
                           patient_id=patient_id, error=None)

@app.route("/doctor_login", methods=["POST"])
def doctor_login_post():
    username   = request.form.get("username", "").strip()
    password   = request.form.get("password", "").strip()
    session_id = request.form.get("session_id", "")
    patient_id = request.form.get("patient_id", "")
    if VALID_DOCTORS.get(username) == password:
        return redirect(url_for("angiogram_upload_page",
                                session_id=session_id, patient_id=patient_id))
    return render_template("doctor_login.html", session_id=session_id,
                           patient_id=patient_id,
                           error="Invalid credentials. Please try again.")



#  ROUTES — Angiogram upload  (doctor portal)


@app.route("/upload_angiogram", methods=["GET"])
def angiogram_upload_page():
    session_id = request.args.get("session_id", "")
    patient_id = request.args.get("patient_id", "")
    return render_template("angiogram_processing.html",
                           session_id=session_id, patient_id=patient_id)


@app.route("/upload_angiogram", methods=["POST"])
def upload_angiogram():
    try:
        session_id   = request.form["session_id"]
        patient_id   = request.form["patient_id"]
        angio_type   = request.form.get("angio_type", "unknown")
        doctor_notes = request.form.get("doctor_notes", "")
        file         = request.files.get("angio_file")

        if not file or file.filename == "":
            return "No angiogram file uploaded.", 400

        session_path = None
        for patient_dir in Path(BASE_DIR).iterdir():
            if not patient_dir.is_dir():
                continue
            candidate = patient_dir / "sessions" / session_id
            if candidate.exists():
                session_path = candidate
                break

        if not session_path:
            return "Session not found.", 404

        angio_folder = session_path / "angiogram"
        angio_folder.mkdir(parents=True, exist_ok=True)

        raw_filename = secure_filename(file.filename) or "angiogram"
        raw_path     = angio_folder / raw_filename
        file.save(str(raw_path))

        preprocessed_root = angio_folder / "preprocessed"
        pipeline_result   = process_angiogram(str(raw_path),
                                              output_root=str(preprocessed_root))
        variants         = pipeline_result["variants"]
        output_directory = pipeline_result["output_directory"]

        metadata_file = session_path / "metadata.json"
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        metadata["angiogram"] = {
            "raw_file_path":       str(raw_path),
            "angio_type":          angio_type,
            "doctor_notes":        doctor_notes,
            "uploaded_by":         "doctor",
            "preprocessed_folder": output_directory,
            "variants":            variants,
            "selected_variant":    None,
            "selected_image_path": None,
            "localization_result": None,
        }

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        return redirect(url_for("angiogram_select", session_id=session_id))

    except Exception as e:
        logging.exception("Angiogram upload / preprocessing failed")
        return f"Upload failed: {str(e)}", 500



#  ROUTE — Serve one preprocessed variant image


@app.route("/angiogram_image/<session_id>/<filename>")
def angiogram_image(session_id, filename):
    for patient_dir in Path(BASE_DIR).iterdir():
        if not patient_dir.is_dir():
            continue
        candidate = patient_dir / "sessions" / session_id
        if not candidate.exists():
            continue
        metadata_file = candidate / "metadata.json"
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            preprocessed_folder = metadata.get("angiogram", {}).get("preprocessed_folder")
            if preprocessed_folder:
                img_path = Path(preprocessed_folder) / filename
                if img_path.exists():
                    return send_file(str(img_path), mimetype="image/png")
        except Exception:
            pass
        for img_path in (candidate / "angiogram").rglob(filename):
            return send_file(str(img_path), mimetype="image/png")

    return "Image not found", 404



#  ROUTE — Variant selection page


@app.route("/angiogram_select/<session_id>")
def angiogram_select(session_id):
    try:
        session_path = None
        for patient_dir in Path(BASE_DIR).iterdir():
            if not patient_dir.is_dir():
                continue
            candidate = patient_dir / "sessions" / session_id
            if candidate.exists():
                session_path = candidate
                break

        if not session_path:
            return "Session not found", 404

        with open(session_path / "metadata.json", "r") as f:
            metadata = json.load(f)

        patient_id = metadata.get("patient_id", "Unknown")
        variants   = metadata.get("angiogram", {}).get("variants", [])

        if not variants:
            return "No preprocessed variants found. Please re-upload the angiogram.", 400

        for v in variants:
            v["url"] = url_for("angiogram_image",
                               session_id=session_id, filename=v["filename"])

        return render_template("angiogram_select.html",
                               session_id=session_id,
                               patient_id=patient_id,
                               variants=variants)
    except Exception as e:
        logging.exception("Error loading angiogram selection page")
        return f"Error: {str(e)}", 500


#  ROUTE — Confirm selected variant → launch DeepSA & redirect

@app.route("/angiogram_confirm", methods=["POST"])
def angiogram_confirm():
    import urllib.parse
    try:
        session_id        = request.form.get("session_id", "").strip()
        patient_id_form   = request.form.get("patient_id", "").strip()
        selected_filename = request.form.get("selected_variant", "").strip()

        if not selected_filename:
            return "Missing selected_variant — no frame was selected.", 400

        use_standalone = bool(patient_id_form) and not session_id

        if session_id:
            session_path = None
            for patient_dir in Path(BASE_DIR).iterdir():
                if not patient_dir.is_dir():
                    continue
                candidate = patient_dir / "sessions" / session_id
                if candidate.exists():
                    session_path = candidate
                    break
            if session_path:
                try:
                    with open(session_path / "metadata.json", "r") as f:
                        metadata = json.load(f)
                    metadata.setdefault("angiogram", {})["selected_variant"] = selected_filename
                    with open(session_path / "metadata.json", "w") as f:
                        json.dump(metadata, f, indent=2)
                except Exception:
                    logging.warning("Could not update session metadata.", exc_info=True)

        if use_standalone:
            meta_path = OUTPUT_ROOT / patient_id_form / "metadata.json"
            if meta_path.exists():
                try:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    meta["selected_variant"] = selected_filename
                    with open(meta_path, "w") as f:
                        json.dump(meta, f, indent=2)
                except Exception:
                    logging.warning("Could not update standalone metadata.", exc_info=True)

        ensure_deepsa_running()

        if use_standalone:
            flask_image_url = url_for(
                "serve_preprocessed_frame",
                patient_id=patient_id_form,
                filename=selected_filename,
                _external=True,
            )
        else:
            if not session_id:
                return "Could not determine flow: neither session_id nor patient_id provided.", 400
            flask_image_url = url_for(
                "angiogram_image",
                session_id=session_id,
                filename=selected_filename,
                _external=True,
            )

        encoded_url     = urllib.parse.quote(flask_image_url, safe="")
        deepsa_redirect = f"http://127.0.0.1:{DEEPSA_PORT}/?image={encoded_url}"
        return redirect(deepsa_redirect)

    except RuntimeError as e:
        return (
            f"<h2>DeepSA could not be started</h2><p>{str(e)}</p>"
            f"<p>Ensure demo.py dependencies are installed and the model checkpoint exists.</p>"
        ), 500
    except Exception as e:
        logging.exception("Angiogram confirmation failed")
        return f"Confirmation failed: {str(e)}", 500


#  ROUTE — ECG Upload Handler

@app.route("/upload_ecg", methods=["POST"])
def upload_ecg():
    try:
        patient_id = request.form["patient_id"]
        age        = int(request.form["age"])
        gender     = request.form["gender"]
        notes      = request.form.get("notes", "")
        ecg_type   = request.form.get("ecg_type", "unknown")
        file       = request.files["ecg_file"]

        if not file:
            return "No file uploaded", 400

        filename   = file.filename
        file_bytes = file.read()

        patient_data = {
            "patient_id": patient_id, "age": age, "gender": gender,
            "notes": notes, "ecg_type": ecg_type,
        }

        from Pipeline_Management.metadata_system import PatientSessionManager, StorageManager, MetadataManager

        metadata_path  = PatientSessionManager.initialize_patient_session(
            patient_data, base_dir=BASE_DIR)
        session_folder = metadata_path.parent

        ecg_folder = session_folder / "ecg"
        ecg_folder.mkdir(parents=True, exist_ok=True)

        encrypted_path = StorageManager.save_encrypted_ecg(metadata_path, file_bytes, filename)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        vector = process_single_ecg_image(tmp_path, target_len=TARGET_LEAD_LENGTH)
        if vector is None:
            raise ValueError("ECG preprocessing failed")

        csv_filename = f"{Path(filename).stem}_preprocessed.csv"
        csv_path     = ecg_folder / csv_filename
        columns      = [f"Lead{lead}_{i}" for lead in range(1, 13) for i in range(TARGET_LEAD_LENGTH)]
        pd.DataFrame([vector], columns=columns).to_csv(csv_path, index=False)
        os.unlink(tmp_path)

        prediction_result = run_ecg_inference(str(csv_path))

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        metadata.setdefault("ecg", {})
        metadata["ecg"]["raw_image_path"]        = encrypted_path
        metadata["ecg"]["processed_csv_path"]    = str(csv_path)
        metadata["ecg"]["classification_result"] = json.dumps(prediction_result)

        # Persist screening form data for the 10-year risk and recommendation panels
        screening_data_raw = request.form.get("screening_form_data", "")
        if screening_data_raw:
            try:
                sfd = json.loads(screening_data_raw)
                sfd.setdefault("heart_rate",      72)
                sfd.setdefault("cigs_per_day",    0)
                sfd.setdefault("bp_treatment",    "No")
                sfd.setdefault("previous_stroke", "No")
                sfd.setdefault("prevalent_hyp",
                    "Yes" if (float(sfd.get("systolic_bp", 0)) >= 140 or
                              float(sfd.get("diastolic_bp", 0)) >= 90) else "No")
                metadata["screening_form_data"] = sfd
            except Exception:
                logging.warning("Could not parse screening_form_data", exc_info=True)

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return redirect(url_for("ecg_result", session_id=session_folder.name))

    except Exception as e:
        logging.exception("Upload + prediction failed")
        return f"Upload failed: {str(e)}", 500


#  ROUTE — ECG Result Display

ECG_RISK_LEVELS_SHOW_TEN_YEAR = {"LOW", "MEDIUM"}

@app.route("/ecg_result/<session_id>")
def ecg_result(session_id):
    try:
        session_path = None
        for patient_dir in Path(BASE_DIR).iterdir():
            if not patient_dir.is_dir():
                continue
            direct = patient_dir / "sessions" / session_id
            if direct.exists():
                session_path = direct
                break

        if not session_path:
            return "Session not found", 404

        metadata_file = session_path / "metadata.json"
        if not metadata_file.exists():
            return "Metadata not found", 404

        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        patient_id         = metadata.get("patient_id", "Unknown")
        classification_raw = metadata.get("ecg", {}).get("classification_result")

        if not classification_raw:
            prediction = {"error": "No prediction data found"}
        else:
            try:
                prediction = (
                    json.loads(classification_raw)
                    if isinstance(classification_raw, str)
                    else classification_raw
                )
            except Exception as e:
                prediction = {"error": f"JSON parse error: {str(e)}"}

        angio_info         = metadata.get("angiogram", {})
        angiogram_uploaded = bool(angio_info)
        angio_selected     = angio_info.get("selected_variant") if angio_info else None
        loc_result         = angio_info.get("localization_result") if angio_info else None

        plot_url  = None
        plot_path = Path(APP_DIR) / "outputs" / "lead_activity_report.png"
        if plot_path.exists():
            plot_url = url_for("ecg_plot", session_id=session_id,
                               filename="lead_activity_report.png")

        ecg_risk_level = prediction.get("risk_level", "UNKNOWN").upper().strip()
        patient_data   = metadata.get("screening_form_data")

        ten_year_data = None
        if ecg_risk_level in ECG_RISK_LEVELS_SHOW_TEN_YEAR and patient_data:
            ten_year_data = _compute_ten_year(patient_data)

        # Determine whether recommendations are available for this session
        has_screening_data = bool(patient_data)

        return render_template(
            "ecg_result.html",
            session_id=session_id,
            patient_id=patient_id,
            prediction=prediction,
            plot_url=plot_url,
            angiogram_uploaded=angiogram_uploaded,
            angio_selected=angio_selected,
            loc_result=loc_result,
            patient_data=patient_data,
            show_ten_year_risk=(ecg_risk_level in ECG_RISK_LEVELS_SHOW_TEN_YEAR),
            ten_year_model_available=(ten_year_model is not None),
            ten_year_data=ten_year_data,
            has_screening_data=has_screening_data,
        )

    except Exception as e:
        logging.exception("Error displaying results")
        return f"Error displaying results: {str(e)}", 500


#  ROUTE — Serve lead activity plot

@app.route("/ecg_plot/<session_id>/<filename>")
def ecg_plot(session_id, filename):
    plot_path = Path(APP_DIR) / "outputs" / filename
    if plot_path.exists():
        return send_file(str(plot_path), mimetype="image/png")

    session_path = None
    for patient_dir in Path(BASE_DIR).iterdir():
        if not patient_dir.is_dir():
            continue
        candidate = patient_dir / "sessions" / session_id
        if candidate.exists():
            session_path = candidate
            break

    if not session_path:
        return "Session not found", 404

    fallback_path = session_path / "ecg" / filename
    if not fallback_path.exists():
        return "Plot not found", 404
    return send_file(str(fallback_path), mimetype="image/png")


#  ROUTE — Pre-screening prediction  (/predict)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "No JSON body received"}), 400

        # Minimum fields required by the new pipeline
        required = [
            "age", "gender", "bmi", "systolic_bp", "diastolic_bp",
            "cholesterol", "glucose", "smoker", "exercise_hours",
            # ten-year fields
            "heart_rate", "bp_treatment", "previous_stroke",
            "prevalent_hyp", "cigs_per_day",
        ]
        missing_fields = [f for f in required if f not in data]
        if missing_fields:
            return jsonify({
                "success": False,
                "message": f"Missing fields: {', '.join(missing_fields)}",
            }), 400

        # ── Graph-augmented risk prediction ───────────────────────────
        X_base, X_full, X_full_scaled = prepare_new_features(data)

        prob       = float(risk_model.predict_proba(X_full_scaled)[0][1])
        pred_class = int(risk_model.predict(X_full_scaled)[0])

        # ── SHAP feature importance ────────────────────────────────────
        top_factors = []
        if risk_explainer is not None:
            try:
                shap_values     = risk_explainer.shap_values(X_full_scaled)
                class_1_impacts = (
                    shap_values[1][0]
                    if isinstance(shap_values, list)
                    else shap_values[0, :, 1]
                )
                contributions = sorted(
                    [
                        (FINAL_FEATURES[i], class_1_impacts[i], X_full.iloc[0][FINAL_FEATURES[i]])
                        for i in range(len(FINAL_FEATURES))
                    ],
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )
                for feature, impact, value in contributions[:3]:
                    effect        = "higher risk" if impact > 0 else "lower risk"
                    display_value = value
                    if feature == "Gender_Encoded" and "Gender" in new_encoders:
                        try:
                            display_value = new_encoders["Gender"].inverse_transform([int(value)])[0]
                        except Exception:
                            pass
                    elif feature == "Smoker_Encoded" and "Smoker" in new_encoders:
                        try:
                            display_value = new_encoders["Smoker"].inverse_transform([int(value)])[0]
                        except Exception:
                            pass
                    top_factors.append({
                        "factor": feature.replace("_Encoded", "").replace("_", " "),
                        "impact": round(float(impact), 3),
                        "value":  str(display_value),
                        "effect": effect,
                    })
            except Exception as shap_err:
                logging.warning(f"SHAP computation failed (non-fatal): {shap_err}")

        # ── Ten-year model ─────────────────────────────────────────────
        ten_year_result = _compute_ten_year(data)

        base = {
            "success":                  True,
            "risk_probability":         prob,
            "explanation":              top_factors,
            "patient_data":             data,
            "ten_year_result":          ten_year_result,
            "ten_year_model_available": ten_year_model is not None,
        }

        if pred_class == 1:
            base.update({
                "risk_status":  "HIGH_RISK",
                "decision":     "MANDATORY_ECG",
                "message":      "You show signs of elevated CAD risk. Please upload your ECG.",
                "next_step":    "UPLOAD_ECG",
                "requires_ecg": True,
                "color":        "red",
            })
        else:
            base.update({
                "risk_status":  "LOW_RISK",
                "decision":     "OPTIONAL_ECG",
                "message":      "You do not currently show significant CAD risk.",
                "next_step":    "OPTIONAL_UPLOAD",
                "requires_ecg": False,
                "color":        "green",
            })

        return jsonify(base)

    except Exception as e:
        logging.exception("/predict failed")
        return jsonify({"success": False, "message": str(e)}), 500



#  ROUTE — Standalone ten-year


@app.route("/predict_ten_year", methods=["POST"])
def predict_ten_year():
    if ten_year_model is None:
        return jsonify({
            "success": False,
            "message": "Ten-year model is not loaded.",
        }), 503

    data   = request.get_json() or {}
    result = _compute_ten_year(data)
    return jsonify(result), (200 if result.get("success") else 500)



#  ROUTE — Recommendation

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "No patient data received"}), 400

        # Build the same 16-feature vector used by the risk model
        _, X_full, X_full_scaled = prepare_new_features(data)

        # Run each per-category binary classifier
        active_categories = []
        for category, clf in rec_multilabel_models.items():
            try:
                pred = int(clf.predict(X_full_scaled)[0])
                if pred == 1:
                    active_categories.append(category)
            except Exception as clf_err:
                logging.warning(f"Classifier for '{category}' failed: {clf_err}")

        # If multilabel models returned nothing, fall back to xgb_rec_label_model
        if not active_categories:
            try:
                label_pred = xgb_rec_label_model.predict(X_full_scaled)[0]
                # Try MultiLabelBinarizer inverse_transform
                decoded = mlb.inverse_transform(
                    np.array([[label_pred]])
                )[0]
                active_categories = list(decoded) if decoded else ["Maintenance"]
            except Exception:
                active_categories = ["Maintenance"]

        # Build response
        recommendations = []
        for cat in active_categories:
            if cat in RECOMMENDATION_ADVICE:
                rec = {"category": cat}
                rec.update(RECOMMENDATION_ADVICE[cat])
                recommendations.append(rec)

        # If still nothing matched the advice catalogue, default to Maintenance
        if not recommendations and "Maintenance" in RECOMMENDATION_ADVICE:
            rec = {"category": "Maintenance"}
            rec.update(RECOMMENDATION_ADVICE["Maintenance"])
            recommendations.append(rec)

        return jsonify({
            "success":        True,
            "recommendations": recommendations,
            "total":          len(recommendations),
        })

    except Exception as e:
        logging.exception("/recommend failed")
        return jsonify({"success": False, "message": str(e)}), 500



#  ROUTE — Recommendations page  (/recommendations)


@app.route("/recommendations")
def recommendations_page():
    session_id = request.args.get("session_id", "")
    source     = request.args.get("source", "screening")  # "screening" | "ecg"
    return render_template("recommendations.html",
                           session_id=session_id,
                           source=source)



#  ROUTE — Get saved screening data for a session  (/get_screening_data)
#  Used by the recommendations page when source=ecg


@app.route("/get_screening_data/<session_id>")
def get_screening_data(session_id):
    try:
        for patient_dir in Path(BASE_DIR).iterdir():
            if not patient_dir.is_dir():
                continue
            candidate = patient_dir / "sessions" / session_id
            if candidate.exists():
                with open(candidate / "metadata.json", "r") as f:
                    metadata = json.load(f)
                sfd = metadata.get("screening_form_data")
                if sfd:
                    return jsonify({"success": True, "patient_data": sfd})
                return jsonify({
                    "success": False,
                    "message": "No screening data found for this session. "
                               "Please complete the pre-screening form first.",
                }), 404
        return jsonify({"success": False, "message": "Session not found."}), 404
    except Exception as e:
        logging.exception("get_screening_data failed")
        return jsonify({"success": False, "message": str(e)}), 500



#  ROUTES — Standalone angiogram processing demo


@app.route("/angiogram_processing")
def angiogram_upload_form():
    return render_template("angiogram_processing.html")


@app.route("/angiogram_process", methods=["POST"])
def process_angiogram_file():
    if "file" not in request.files:
        return "No file part", 400
    file = request.files["file"]
    if file.filename == "":
        return "No file selected", 400

    filename  = secure_filename(file.filename) or "angiogram"
    temp_path = UPLOAD_FOLDER / filename
    file.save(str(temp_path))

    try:
        pipeline_result = process_angiogram(str(temp_path), output_root=str(OUTPUT_ROOT))
        patient_id      = pipeline_result["patient_id"]
        return redirect(url_for("angiogram_results", patient_id=patient_id))
    except Exception as e:
        logging.exception("Standalone angiogram processing failed")
        return f"<h2>Error processing file</h2><p>{str(e)}</p>", 500
    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.route("/angiogram_results")
def angiogram_results():
    patient_id = request.args.get("patient_id", "")
    if not patient_id:
        return "Missing patient_id", 400

    patient_dir   = OUTPUT_ROOT / patient_id
    metadata_file = patient_dir / "metadata.json"
    if not metadata_file.exists():
        return f"Result folder not found for patient_id: {patient_id}", 404

    with open(metadata_file, "r") as f:
        meta = json.load(f)

    variants         = meta.get("variants", [])
    selected_indices = meta.get("selected_frame_indices", [])
    for v in variants:
        v["url"] = url_for("serve_preprocessed_frame",
                           patient_id=patient_id, filename=v["filename"])

    return render_template(
        "angiogram_results1.html",
        patient_id=patient_id,
        variants=variants,
        selected_indices=selected_indices,
    )


@app.route("/angiogram_frame/<patient_id>/<filename>")
def serve_preprocessed_frame(patient_id, filename):
    img_path = OUTPUT_ROOT / patient_id / filename
    if not img_path.exists():
        return "Frame not found", 404
    return send_file(str(img_path), mimetype="image/png")



#  Entry point


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    app.run(debug=True)
