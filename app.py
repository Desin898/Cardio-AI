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


# ============================================================
#  ECG PREPROCESSING FUNCTIONS
# ============================================================

TARGET_LEAD_LENGTH = 737

def preprocess_step1_image(img):
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    thr = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 35, 7
    )
    return thr

def crop_12_leads_from_gray(gray_img):
    h, w = gray_img.shape[:2]
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
    return np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(sig)), sig)

def process_single_ecg_image(img_path, target_len=TARGET_LEAD_LENGTH, save_images=False, save_folder=None):
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


# ============================================================
#  RETRIEVAL MANAGER
# ============================================================

from Pipeline_Management.metadata_system import EncryptionManager, MetadataManager

class RetrievalManager:
    @staticmethod
    def get_decrypted_ecg_png(metadata_path: Path) -> str:
        metadata        = MetadataManager.read_metadata(metadata_path)
        enc_path        = metadata.get("ecg", {}).get("raw_image_path")
        if enc_path is None:
            raise ValueError("No ECG image found in metadata.")
        enc             = EncryptionManager()
        decrypted_bytes = enc.decrypt_file(Path(enc_path))
        img_array       = np.frombuffer(decrypted_bytes, dtype=np.uint8)
        img             = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        temp_path       = Path(enc_path).with_suffix(".png")
        cv2.imwrite(str(temp_path), img)
        return str(temp_path)


# ============================================================
#  ECG INFERENCE
# ============================================================

INFERENCE_SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__),
    "process_02_cardiac_analysis", "ecg_risk_prediction",
    "inference", "predict_ecg_risk.py"
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
            capture_output=True, text=True, timeout=120, cwd=project_root
        )
        if result.stdout:
            logging.info(f"Inference stdout:\n{result.stdout}")
        if result.stderr:
            logging.error(f"Inference stderr:\n{result.stderr}")
        if result.returncode != 0:
            stderr_clean = result.stderr.strip() or "No stderr captured."
            return {"error": stderr_clean, "prediction": "Error", "confidence": "0%",
                    "risk_level": "UNKNOWN", "suspected_vessel": "N/A"}
        prediction_result = {"prediction": "Unknown", "confidence": "0%",
                             "risk_level": "UNKNOWN", "suspected_vessel": "N/A", "plot_path": None}
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

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
BASE_DIR      = "Pipeline_Management/Patients"
MODEL_PATH    = 'gateway_model.pkl'
SCALER_PATH   = 'gateway_scaler.pkl'
ENCODERS_PATH = 'gateway_encoders.pkl'
FEATURES_PATH = 'gateway_features.pkl'

with open(MODEL_PATH,    'rb') as f: model            = pickle.load(f)
with open(SCALER_PATH,   'rb') as f: scaler           = pickle.load(f)
with open(ENCODERS_PATH, 'rb') as f: encoders         = pickle.load(f)
with open(FEATURES_PATH, 'rb') as f: GATEWAY_FEATURES = pickle.load(f)

explainer = shap.TreeExplainer(model)

# ── Ten-year CHD model (trained by xx.ipynb) ──────────────────
TEN_YEAR_MODEL_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ten_year_models')
TEN_YEAR_FEATURES = [
    "age", "male", "currentSmoker", "cigsPerDay",
    "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes",
    "totChol", "sysBP", "diaBP", "pulsePressure", "bp_ratio",
    "BMI", "heartRate", "glucose"
]
try:
    with open(os.path.join(TEN_YEAR_MODEL_DIR, 'ten_year_model.pkl'),     'rb') as f: ten_year_model     = pickle.load(f)
    with open(os.path.join(TEN_YEAR_MODEL_DIR, 'ten_year_threshold.pkl'), 'rb') as f: ten_year_threshold = pickle.load(f)
    logging.info("✅ Ten-year CHD model loaded successfully.")
except FileNotFoundError:
    ten_year_model     = None
    ten_year_threshold = 0.5
    logging.warning("⚠  ten_year_models/ not found — run xx.ipynb first to train and save the model.")

VALID_DOCTORS = {
    'doctor1': 'pass123',
    'admin':   'admin',
}

# Standalone angiogram demo dirs
UPLOAD_FOLDER = Path("uploads")
OUTPUT_ROOT   = Path("Preprocessed_Angiogram_Output")
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_ROOT.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['OUTPUT_ROOT']   = str(OUTPUT_ROOT)

# DeepSA (Gradio) settings — demo.py is launched automatically on first use
DEEPSA_PORT   = int(os.environ.get("DEEPSA_PORT", 7860))
DEEPSA_URL    = os.environ.get("DEEPSA_URL", f"http://127.0.0.1:{DEEPSA_PORT}")
DEEPSA_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo.py")


# ============================================================
#  DeepSA PROCESS MANAGEMENT
# ============================================================

_deepsa_proc: subprocess.Popen | None = None
_deepsa_lock = threading.Lock()


def _deepsa_running() -> bool:
    """Return True if the DeepSA Gradio server is already accepting connections."""
    import socket
    try:
        with socket.create_connection(("127.0.0.1", DEEPSA_PORT), timeout=1):
            return True
    except OSError:
        return False


def ensure_deepsa_running():
    """
    Launch demo.py as a background subprocess if it is not already listening
    on DEEPSA_PORT. Waits up to 30 seconds for the server to become ready.
    Raises RuntimeError if it never starts.
    """
    global _deepsa_proc

    with _deepsa_lock:
        if _deepsa_running():
            logging.info("DeepSA is already running — skipping launch.")
            return

        logging.info(f"Starting DeepSA from: {DEEPSA_SCRIPT}")
        _deepsa_proc = subprocess.Popen(
            [sys.executable, DEEPSA_SCRIPT],
            cwd=os.path.dirname(DEEPSA_SCRIPT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    # Poll until the server is ready (up to 30 s)
    deadline = time.time() + 30
    while time.time() < deadline:
        if _deepsa_running():
            logging.info("DeepSA is ready and accepting connections.")
            return
        time.sleep(0.5)

    logging.error("DeepSA did not become ready within 30 seconds.")
    raise RuntimeError(
        "DeepSA (demo.py) failed to start within 30 seconds. "
        "Check that the model checkpoint exists and all dependencies are installed."
    )


# -------------------------------------------------------------------
# Helper: prepare feature vector from JSON
# -------------------------------------------------------------------
def prepare_features(data):
    high_chol    = 1 if data['cholesterol'] > 200 else 0
    high_glucose = 1 if data['glucose'] > 100 else 0
    hypertension = 1 if (data['systolic_bp'] >= 140 or data['diastolic_bp'] >= 90) else 0
    obesity      = 1 if data['bmi'] >= 30 else 0
    risk_score   = sum([
        data['age'] > 50, data['smoker'] == 'Yes', data['diabetes'] == 'Yes',
        high_chol, hypertension, obesity, data['exercise_hours'] < 2
    ])
    gender_enc   = encoders['Gender'].transform([data['gender']])[0]
    smoker_enc   = encoders['Smoker'].transform([data['smoker']])[0]
    alcohol_enc  = encoders['Alcohol'].transform([data['alcohol']])[0]
    activity_enc = encoders['Physical_Activity'].transform([data['physical_activity']])[0]
    stress_enc   = encoders['Stress_Level'].transform([data['stress_level']])[0]
    feature_dict = {
        'Age': data['age'], 'Gender_Encoded': gender_enc, 'BMI': data['bmi'],
        'Systolic_BP': data['systolic_bp'], 'Diastolic_BP': data['diastolic_bp'],
        'Cholesterol mg/dL': data['cholesterol'], 'Glucose mg/dL': data['glucose'],
        'Smoker_Encoded': smoker_enc, 'Diabetes': 1 if data['diabetes'] == 'Yes' else 0,
        'Alcohol_Encoded': alcohol_enc, 'Physical_Activity_Encoded': activity_enc,
        'Family_History': 1 if data['family_history'] == 'Yes' else 0,
        'Stress_Level_Encoded': stress_enc, 'Sleep_Hours': data['sleep_hours'],
        'Years_Smoking': data['years_smoking'], 'Exercise hours/week': data['exercise_hours'],
        'High_Cholesterol': high_chol, 'High_Glucose': high_glucose,
        'Hypertension': hypertension, 'Obesity': obesity, 'Risk_Score': risk_score
    }
    return pd.DataFrame([feature_dict])[GATEWAY_FEATURES]


# ====================================================================
#  ROUTES — General / Patient
# ====================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/patient_login.html')
def patient_login():
    return render_template('patient_login.html')

@app.route('/patient_register.html')
def patient_register():
    return render_template('patient_register.html')

@app.route('/patient_dashboard.html')
def patient_dashboard():
    return render_template('patient_dashboard.html')

@app.route('/upload_ecg.html', methods=['GET'])
def upload_page():
    return render_template('upload_ecg.html')

@app.route('/pre_screening.html')
def screening():
    return render_template('pre_screening.html')

@app.route('/pre_screening_results.html')
def result():
    return render_template('pre_screening_results.html')


# ====================================================================
#  ROUTES — Doctor portal stubs
# ====================================================================

@app.route('/doctor_upload.html')
def doctor_upload():
    return render_template('doctor_upload.html')

@app.route('/doctor_analysis_results.html')
def doctor_analysis_results():
    return render_template('doctor_analysis_results.html')


# ====================================================================
#  ROUTES — Doctor login
# ====================================================================

@app.route('/doctor_login.html')
@app.route('/doctor_login', methods=['GET'])
def doctor_login():
    session_id = request.args.get('session_id', '')
    patient_id = request.args.get('patient_id', '')
    return render_template('doctor_login.html', session_id=session_id,
                           patient_id=patient_id, error=None)

@app.route('/doctor_login', methods=['POST'])
def doctor_login_post():
    username   = request.form.get('username', '').strip()
    password   = request.form.get('password', '').strip()
    session_id = request.form.get('session_id', '')
    patient_id = request.form.get('patient_id', '')
    if VALID_DOCTORS.get(username) == password:
        return redirect(url_for('angiogram_upload_page',
                                session_id=session_id, patient_id=patient_id))
    return render_template('doctor_login.html', session_id=session_id,
                           patient_id=patient_id,
                           error='Invalid credentials. Please try again.')


# ====================================================================
#  ROUTES — Angiogram upload  (doctor portal)
# ====================================================================

@app.route('/upload_angiogram', methods=['GET'])
def angiogram_upload_page():
    session_id = request.args.get('session_id', '')
    patient_id = request.args.get('patient_id', '')
    return render_template('angiogram_processing.html',
                           session_id=session_id, patient_id=patient_id)


@app.route('/upload_angiogram', methods=['POST'])
def upload_angiogram():
    try:
        session_id   = request.form['session_id']
        patient_id   = request.form['patient_id']
        angio_type   = request.form.get('angio_type', 'unknown')
        doctor_notes = request.form.get('doctor_notes', '')
        file         = request.files.get('angio_file')

        if not file or file.filename == '':
            return "No angiogram file uploaded.", 400

        # ── Locate session folder ─────────────────────────────────────
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

        # ── Save raw angiogram ────────────────────────────────────────
        angio_folder = session_path / "angiogram"
        angio_folder.mkdir(parents=True, exist_ok=True)

        raw_filename = secure_filename(file.filename) or "angiogram"
        raw_path     = angio_folder / raw_filename
        file.save(str(raw_path))
        logging.info(f"Raw angiogram saved: {raw_path}")

        # ── Run preprocessing ─────────────────────────────────────────
        preprocessed_root = angio_folder / "preprocessed"
        pipeline_result   = process_angiogram(str(raw_path),
                                              output_root=str(preprocessed_root))

        variants         = pipeline_result["variants"]
        output_directory = pipeline_result["output_directory"]

        logging.info(f"Preprocessing complete. Variants: {[v['filename'] for v in variants]}")

        # ── Update session metadata ───────────────────────────────────
        metadata_file = session_path / "metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        metadata['angiogram'] = {
            'raw_file_path':        str(raw_path),
            'angio_type':           angio_type,
            'doctor_notes':         doctor_notes,
            'uploaded_by':          'doctor',
            'preprocessed_folder':  output_directory,
            'variants':             variants,
            'selected_variant':     None,
            'selected_image_path':  None,
            'localization_result':  None,
        }

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        return redirect(url_for('angiogram_select', session_id=session_id))

    except Exception as e:
        logging.exception("Angiogram upload / preprocessing failed")
        return f"Upload failed: {str(e)}", 500


# ====================================================================
#  ROUTE — Serve one preprocessed variant image  (doctor portal)
# ====================================================================

@app.route('/angiogram_image/<session_id>/<filename>')
def angiogram_image(session_id, filename):
    for patient_dir in Path(BASE_DIR).iterdir():
        if not patient_dir.is_dir():
            continue
        candidate = patient_dir / "sessions" / session_id
        if not candidate.exists():
            continue

        metadata_file = candidate / "metadata.json"
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            preprocessed_folder = metadata.get('angiogram', {}).get('preprocessed_folder')
            if preprocessed_folder:
                img_path = Path(preprocessed_folder) / filename
                if img_path.exists():
                    return send_file(str(img_path), mimetype='image/png')
        except Exception:
            pass

        for img_path in (candidate / "angiogram").rglob(filename):
            return send_file(str(img_path), mimetype='image/png')

    return "Image not found", 404


# ====================================================================
#  ROUTE — Variant selection page  (doctor portal, GET)
# ====================================================================

@app.route('/angiogram_select/<session_id>')
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

        with open(session_path / "metadata.json", 'r') as f:
            metadata = json.load(f)

        patient_id = metadata.get("patient_id", "Unknown")
        variants   = metadata.get("angiogram", {}).get("variants", [])

        if not variants:
            return "No preprocessed variants found. Please re-upload the angiogram.", 400

        for v in variants:
            v['url'] = url_for('angiogram_image',
                               session_id=session_id, filename=v['filename'])

        return render_template('angiogram_select.html',
                               session_id=session_id,
                               patient_id=patient_id,
                               variants=variants)

    except Exception as e:
        logging.exception("Error loading angiogram selection page")
        return f"Error: {str(e)}", 500


# ====================================================================
#  ROUTE — Doctor confirms selected variant → launch DeepSA & redirect
# ====================================================================

@app.route('/angiogram_confirm', methods=['POST'])
def angiogram_confirm():
    """
    Handles TWO flows:

    A) Doctor portal flow  — form sends: session_id + selected_variant
       Image is served by:  /angiogram_image/<session_id>/<filename>

    B) Standalone demo flow — form sends: patient_id + selected_variant
       Image is served by:  /angiogram_frame/<patient_id>/<filename>

    In both cases:
      1. Persist the selection to metadata (best-effort).
      2. Ensure demo.py (DeepSA/Gradio) is running — launch it if not.
      3. Build the absolute Flask URL for the selected image.
      4. Redirect to DeepSA with ?image=<url> so the injected JS
         auto-loads the image into the Gradio input widget.
    """
    import urllib.parse

    try:
        session_id        = request.form.get('session_id', '').strip()
        patient_id_form   = request.form.get('patient_id', '').strip()
        selected_filename = request.form.get('selected_variant', '').strip()

        logging.info(
            f"angiogram_confirm — session_id='{session_id}' "
            f"patient_id='{patient_id_form}' "
            f"selected_variant='{selected_filename}'"
        )

        if not selected_filename:
            return "Missing selected_variant — no frame was selected.", 400

        # ── Determine which flow we're in ─────────────────────────────
        use_standalone = bool(patient_id_form) and not session_id

        # ── 1A. Doctor portal: persist to session metadata ────────────
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
                metadata_file = session_path / "metadata.json"
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    metadata.setdefault('angiogram', {})['selected_variant'] = selected_filename
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    logging.info(f"Saved selected_variant '{selected_filename}' to session metadata.")
                except Exception:
                    logging.warning("Could not update session metadata.", exc_info=True)
            else:
                logging.warning(f"Session '{session_id}' not found — skipping metadata update.")

        # ── 1B. Standalone: persist to OUTPUT_ROOT metadata ──────────
        if use_standalone:
            meta_path = OUTPUT_ROOT / patient_id_form / "metadata.json"
            if meta_path.exists():
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    meta['selected_variant'] = selected_filename
                    with open(meta_path, 'w') as f:
                        json.dump(meta, f, indent=2)
                    logging.info(f"Saved selected_variant '{selected_filename}' to standalone metadata.")
                except Exception:
                    logging.warning("Could not update standalone metadata.", exc_info=True)

        # ── 2. Start DeepSA if not already running ────────────────────
        ensure_deepsa_running()

        # ── 3. Build the Flask URL for the selected image ─────────────
        if use_standalone:
            # Standalone flow — image served by serve_preprocessed_frame
            flask_image_url = url_for(
                'serve_preprocessed_frame',
                patient_id=patient_id_form,
                filename=selected_filename,
                _external=True,
            )
        else:
            # Doctor portal flow — image served by angiogram_image
            if not session_id:
                return (
                    "Could not determine flow: neither session_id nor patient_id was provided. "
                    "Please go back and try again."
                ), 400
            flask_image_url = url_for(
                'angiogram_image',
                session_id=session_id,
                filename=selected_filename,
                _external=True,
            )

        logging.info(f"Redirecting to DeepSA with image URL: {flask_image_url}")

        # ── 4. Redirect to DeepSA with the image URL ──────────────────
        encoded_url     = urllib.parse.quote(flask_image_url, safe='')
        deepsa_redirect = f"http://127.0.0.1:{DEEPSA_PORT}/?image={encoded_url}"

        return redirect(deepsa_redirect)

    except RuntimeError as e:
        logging.exception("DeepSA startup error")
        return (
            f"<h2>DeepSA could not be started</h2>"
            f"<p>{str(e)}</p>"
            f"<p>Ensure <code>demo.py</code> dependencies are installed "
            f"and the model checkpoint exists, then try again.</p>"
        ), 500

    except Exception as e:
        logging.exception("Angiogram confirmation failed")
        return f"Confirmation failed: {str(e)}", 500


# ====================================================================
#  ROUTE — ECG Upload Handler (POST)
# ====================================================================

@app.route('/upload_ecg', methods=['POST'])
def upload_ecg():
    try:
        patient_id = request.form['patient_id']
        age        = int(request.form['age'])
        gender     = request.form['gender']
        notes      = request.form.get('notes', '')
        ecg_type   = request.form.get('ecg_type', 'unknown')
        file       = request.files['ecg_file']

        if not file:
            return "No file uploaded", 400

        filename   = file.filename
        file_bytes = file.read()

        patient_data = {"patient_id": patient_id, "age": age, "gender": gender,
                        "notes": notes, "ecg_type": ecg_type}

        from Pipeline_Management.metadata_system import PatientSessionManager, StorageManager, MetadataManager

        metadata_path  = PatientSessionManager.initialize_patient_session(
            patient_data, base_dir=BASE_DIR)
        session_folder = metadata_path.parent

        ecg_folder = session_folder / "ecg"
        ecg_folder.mkdir(parents=True, exist_ok=True)

        encrypted_path = StorageManager.save_encrypted_ecg(metadata_path, file_bytes, filename)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        vector = process_single_ecg_image(tmp_path, target_len=TARGET_LEAD_LENGTH)
        if vector is None:
            raise ValueError("ECG preprocessing failed")

        csv_filename = f"{Path(filename).stem}_preprocessed.csv"
        csv_path     = ecg_folder / csv_filename
        columns      = [f"Lead{lead}_{i}" for lead in range(1, 13) for i in range(TARGET_LEAD_LENGTH)]
        pd.DataFrame([vector], columns=columns).to_csv(csv_path, index=False)
        logging.info(f"CSV created at: {csv_path}")

        os.unlink(tmp_path)

        prediction_result = run_ecg_inference(str(csv_path))
        logging.info(f"Prediction result: {prediction_result}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        metadata.setdefault('ecg', {})
        metadata['ecg']['raw_image_path']        = encrypted_path
        metadata['ecg']['processed_csv_path']    = str(csv_path)
        metadata['ecg']['classification_result'] = json.dumps(prediction_result)

        # Persist screening form data so the ECG result page can run 10-year risk.
        # Inject safe defaults for optional ten-year fields before saving.
        screening_data_raw = request.form.get('screening_form_data', '')
        if screening_data_raw:
            try:
                sfd = json.loads(screening_data_raw)
                sfd.setdefault('heart_rate',      72)
                sfd.setdefault('cigs_per_day',    0)
                sfd.setdefault('bp_treatment',    'No')
                sfd.setdefault('previous_stroke', 'No')
                sfd.setdefault('prevalent_hyp',
                    'Yes' if (sfd.get('systolic_bp', 0) >= 140 or
                              sfd.get('diastolic_bp', 0) >= 90) else 'No')
                metadata['screening_form_data'] = sfd
            except Exception:
                pass

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return redirect(url_for('ecg_result', session_id=session_folder.name))

    except Exception as e:
        logging.exception("Upload + prediction failed")
        return f"Upload failed: {str(e)}", 500


# ====================================================================
#  ROUTE — ECG Result Display
# ====================================================================

@app.route('/ecg_result/<session_id>')
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

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        patient_id         = metadata.get("patient_id", "Unknown")
        classification_raw = metadata.get("ecg", {}).get("classification_result")

        if not classification_raw:
            prediction = {"error": "No prediction data found"}
        else:
            try:
                prediction = json.loads(classification_raw) if isinstance(classification_raw, str) else classification_raw
            except Exception as e:
                prediction = {"error": f"JSON parse error: {str(e)}"}

        angio_info         = metadata.get("angiogram", {})
        angiogram_uploaded = bool(angio_info)
        angio_selected     = angio_info.get("selected_variant") if angio_info else None
        loc_result         = angio_info.get("localization_result") if angio_info else None

        plot_url  = None
        plot_path = Path(os.path.dirname(__file__)) / "outputs" / "lead_activity_report.png"
        if plot_path.exists():
            plot_url = url_for('ecg_plot', session_id=session_id,
                               filename="lead_activity_report.png")

        return render_template('ecg_result.html',
                               session_id=session_id,
                               patient_id=patient_id,
                               prediction=prediction,
                               plot_url=plot_url,
                               angiogram_uploaded=angiogram_uploaded,
                               angio_selected=angio_selected,
                               loc_result=loc_result,
                               patient_data=metadata.get('screening_form_data'))

    except Exception as e:
        logging.exception("Error displaying results")
        return f"Error displaying results: {str(e)}", 500


# ====================================================================
#  ROUTE — Serve lead activity plot
# ====================================================================

@app.route('/ecg_plot/<session_id>/<filename>')
def ecg_plot(session_id, filename):
    plot_path = Path(os.path.dirname(__file__)) / "outputs" / filename
    if plot_path.exists():
        return send_file(str(plot_path), mimetype='image/png')

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
    return send_file(str(fallback_path), mimetype='image/png')


# ====================================================================
#  ROUTE — Pre-screening prediction endpoint
# ====================================================================

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_json = request.get_json()
        required = [
            'age', 'gender', 'bmi', 'systolic_bp', 'diastolic_bp',
            'cholesterol', 'glucose', 'smoker', 'years_smoking',
            'exercise_hours', 'diabetes', 'alcohol', 'physical_activity',
            'family_history', 'stress_level', 'sleep_hours'
        ]
        for field in required:
            if field not in input_json:
                return jsonify({'success': False, 'message': f'Missing field: {field}'}), 400

        X        = prepare_features(input_json)
        X_scaled = scaler.transform(X)
        prob       = model.predict_proba(X_scaled)[0][1]
        pred_class = model.predict(X_scaled)[0]

        shap_values = explainer.shap_values(X_scaled)
        class_1_impacts = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0, :, 1]

        contributions = sorted(
            [(GATEWAY_FEATURES[i], class_1_impacts[i], X.iloc[0][GATEWAY_FEATURES[i]])
             for i in range(len(GATEWAY_FEATURES))],
            key=lambda x: abs(x[1]), reverse=True
        )

        top_factors = []
        for feature, impact, value in contributions[:3]:
            effect        = "higher risk" if impact > 0 else "lower risk"
            display_value = value
            if '_Encoded' in feature:
                orig = feature.replace('_Encoded', '')
                if orig in encoders:
                    try:
                        display_value = encoders[orig].inverse_transform([int(value)])[0]
                    except Exception:
                        pass
            top_factors.append({
                'factor': feature.replace('_Encoded', '').replace('_', ' '),
                'impact': round(float(impact), 3),
                'value':  str(display_value),
                'effect': effect
            })

        # Echo back patient_data so results pages can use it for 10-year risk.
        # Set safe defaults for optional ten-year fields so /predict_ten_year
        # never raises a KeyError even if the screening form omitted them.
        patient_data_echo = {k: input_json[k] for k in input_json}
        patient_data_echo.setdefault('heart_rate',      72)
        patient_data_echo.setdefault('cigs_per_day',    0)
        patient_data_echo.setdefault('bp_treatment',    'No')
        patient_data_echo.setdefault('previous_stroke', 'No')
        patient_data_echo.setdefault('prevalent_hyp',
            'Yes' if (input_json.get('systolic_bp', 0) >= 140 or
                      input_json.get('diastolic_bp', 0) >= 90) else 'No')

        if pred_class == 1:
            result = {
                'success': True, 'risk_status': 'HIGH_RISK',
                'risk_probability': float(prob), 'decision': 'MANDATORY_ECG',
                'message': "You show signs of elevated CAD risk. Please upload your ECG.",
                'explanation': top_factors, 'next_step': 'UPLOAD_ECG',
                'requires_ecg': True, 'color': 'red',
                'patient_data': patient_data_echo
            }
        else:
            result = {
                'success': True, 'risk_status': 'LOW_RISK',
                'risk_probability': float(prob), 'decision': 'OPTIONAL_ECG',
                'message': "You do not currently show significant CAD risk.",
                'explanation': top_factors, 'next_step': 'OPTIONAL_UPLOAD',
                'requires_ecg': False, 'color': 'green',
                'patient_data': patient_data_echo
            }
        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


def _framingham_ten_year_risk(features: dict) -> float:
    """
    Simplified Framingham 10-year CHD risk score (point-based approximation).
    Returns a probability in [0.0, 1.0].
    Used as a rule-based fallback when the trained ML model is unavailable.

    Reference: Wilson et al. (1998), Circulation 97:1837-1847.
    """
    age         = features['age']
    male        = features['male']
    total_chol  = features['totChol']
    hdl_chol    = 50.0          # assumed average when not collected
    sys_bp      = features['sysBP']
    bp_meds     = features['BPMeds']
    smoker      = features['currentSmoker']
    diabetes    = features['diabetes']

    # --- Age points ---
    if male:
        age_pts = (0 if age < 35 else 2 if age < 40 else 5 if age < 45 else
                   6 if age < 50 else 8 if age < 55 else 10 if age < 60 else
                   11 if age < 65 else 12 if age < 70 else 14 if age < 75 else 15)
    else:
        age_pts = (0 if age < 35 else 2 if age < 40 else 4 if age < 45 else
                   5 if age < 50 else 7 if age < 55 else 8 if age < 60 else
                   9 if age < 65 else 10 if age < 70 else 11 if age < 75 else 12)

    # --- Total cholesterol points ---
    if male:
        chol_pts = (0 if total_chol < 160 else 1 if total_chol < 200 else
                    2 if total_chol < 240 else 3 if total_chol < 280 else 4)
    else:
        chol_pts = (0 if total_chol < 160 else 1 if total_chol < 200 else
                    3 if total_chol < 240 else 4 if total_chol < 280 else 5)

    # --- HDL-C points (using assumed average) ---
    hdl_pts = (2 if hdl_chol < 35 else 1 if hdl_chol < 45 else
               0 if hdl_chol < 50 else -1 if hdl_chol < 60 else -2)

    # --- Systolic BP points ---
    if male:
        sbp_pts = (0 if sys_bp < 120 else
                   (1 if bp_meds else 0) if sys_bp < 130 else
                   (2 if bp_meds else 1) if sys_bp < 140 else
                   (2 if bp_meds else 1) if sys_bp < 160 else
                   (3 if bp_meds else 2))
    else:
        sbp_pts = (0 if sys_bp < 120 else
                   (3 if bp_meds else 1) if sys_bp < 130 else
                   (4 if bp_meds else 2) if sys_bp < 140 else
                   (5 if bp_meds else 3) if sys_bp < 160 else
                   (6 if bp_meds else 4))

    # --- Smoker / Diabetes points ---
    smoker_pts   = 4 if (male and smoker) else (3 if smoker else 0)
    diabetes_pts = 3 if (male and diabetes) else (4 if diabetes else 0)

    total_pts = age_pts + chol_pts + hdl_pts + sbp_pts + smoker_pts + diabetes_pts

    # --- Point → 10-year risk lookup (male / female) ---
    male_risk   = {-3:0.01,-2:0.02,-1:0.02, 0:0.03, 1:0.04, 2:0.04, 3:0.06,
                    4:0.07, 5:0.09, 6:0.11, 7:0.14, 8:0.18, 9:0.22,10:0.27,
                   11:0.33,12:0.40,13:0.47,14:0.56,15:0.67,16:0.79,17:0.90}
    female_risk = {-2:0.01,-1:0.01, 0:0.01, 1:0.01, 2:0.01, 3:0.02, 4:0.02,
                    5:0.03, 6:0.03, 7:0.04, 8:0.05, 9:0.06,10:0.08,11:0.10,
                   12:0.12,13:0.14,14:0.17,15:0.20,16:0.24,17:0.27,18:0.32,
                   19:0.37,20:0.43,21:0.50,22:0.56,23:0.64,24:0.71,25:0.78}

    lookup = male_risk if male else female_risk
    clamped_pts = max(min(total_pts, max(lookup.keys())), min(lookup.keys()))
    # Nearest key
    nearest_key = min(lookup.keys(), key=lambda k: abs(k - clamped_pts))
    return lookup[nearest_key]


@app.route('/predict_ten_year', methods=['POST'])
def predict_ten_year():
    """
    10-Year Coronary Heart Disease risk prediction.

    Primary:  ML model loaded from ten_year_models/ten_year_model.pkl.
    Fallback: Framingham point-score formula (returns meaningful results
              even when the model pkl contains None).

    Accepts the same JSON payload as /predict (pre-screening form data).
    Returns:  { success, percent, category, colour, probability, advice,
                method }   ← 'ml' or 'framingham'
    """
    try:
        data   = request.get_json() or {}
        sys_bp = float(data.get('systolic_bp', 120))
        dia_bp = float(data.get('diastolic_bp', 80))

        features = {
            'age':            float(data.get('age', 45)),
            'male':           1 if data.get('gender') == 'Male' else 0,
            'currentSmoker':  1 if data.get('smoker') == 'Yes' else 0,
            'cigsPerDay':     float(data.get('cigs_per_day', 0) or 0),
            'BPMeds':         1 if data.get('bp_treatment') == 'Yes' else 0,
            'prevalentStroke':1 if data.get('previous_stroke') == 'Yes' else 0,
            'prevalentHyp':   1 if data.get('prevalent_hyp') == 'Yes' else 0,
            'diabetes':       1 if data.get('diabetes') == 'Yes' else 0,
            'totChol':        float(data.get('cholesterol', 200)),
            'sysBP':          sys_bp,
            'diaBP':          dia_bp,
            'pulsePressure':  sys_bp - dia_bp,
            'bp_ratio':       sys_bp / (dia_bp + 1e-6),
            'BMI':            float(data.get('bmi', 25)),
            'heartRate':      float(data.get('heart_rate', 72)),   # safe default
            'glucose':        float(data.get('glucose', 100)),
        }

        method = 'ml'
        if ten_year_model is not None:
            X    = pd.DataFrame([features])[TEN_YEAR_FEATURES]
            prob = ten_year_model.predict_proba(X)[0][1]
        else:
            logging.warning("ten_year_model is None — using Framingham fallback.")
            prob   = _framingham_ten_year_risk(features)
            method = 'framingham'

        # Apply custom threshold if available and it was ML-derived
        threshold = ten_year_threshold if ten_year_threshold else 0.5
        pct       = min(round(float(prob) * 100, 1), 100.0)

        # Category uses probability thresholds (10 % / 20 %) for clinical alignment
        if prob < 0.10:
            category, colour = 'LOW', 'low'
        elif prob < 0.20:
            category, colour = 'MEDIUM', 'medium'
        else:
            category, colour = 'HIGH', 'high'

        advice = (
            'Your projected 10-year risk is low. Maintain healthy habits and attend regular check-ups.'
            if category == 'LOW' else
            'Your projected 10-year risk is moderate. Lifestyle changes — diet, exercise, quitting smoking — can significantly reduce this.'
            if category == 'MEDIUM' else
            'Your projected 10-year risk is high. Please discuss these results with your doctor as soon as possible.'
        )

        return jsonify({
            'success':     True,
            'percent':     pct,
            'category':    category,
            'colour':      colour,
            'probability': round(float(prob), 4),
            'advice':      advice,
            'method':      method,
        })

    except Exception as e:
        logging.exception("Ten-year prediction failed")
        return jsonify({'success': False, 'message': str(e)}), 500


# ====================================================================
#  ROUTES — Standalone angiogram processing demo
# ====================================================================

@app.route('/angiogram_processing')
def angiogram_upload_form():
    return render_template('angiogram_processing.html')


@app.route('/angiogram_process', methods=['POST'])
def process_angiogram_file():
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']
    if file.filename == '':
        return 'No file selected', 400

    filename  = secure_filename(file.filename) or "angiogram"
    temp_path = UPLOAD_FOLDER / filename
    file.save(str(temp_path))

    try:
        pipeline_result = process_angiogram(str(temp_path), output_root=str(OUTPUT_ROOT))
        patient_id      = pipeline_result["patient_id"]
        return redirect(url_for('angiogram_results', patient_id=patient_id))

    except Exception as e:
        logging.exception("Standalone angiogram processing failed")
        return f'<h2>Error processing file</h2><p>{str(e)}</p>', 500

    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.route('/angiogram_results')
def angiogram_results():
    patient_id = request.args.get('patient_id', '')
    if not patient_id:
        return 'Missing patient_id', 400

    patient_dir   = OUTPUT_ROOT / patient_id
    metadata_file = patient_dir / "metadata.json"

    if not metadata_file.exists():
        return f'Result folder not found for patient_id: {patient_id}', 404

    with open(metadata_file, 'r') as f:
        meta = json.load(f)

    variants         = meta.get("variants", [])
    selected_indices = meta.get("selected_frame_indices", [])

    for v in variants:
        v['url'] = url_for('serve_preprocessed_frame',
                           patient_id=patient_id,
                           filename=v['filename'])

    return render_template(
        'angiogram_results1.html',
        patient_id=patient_id,
        variants=variants,
        selected_indices=selected_indices,
    )


@app.route('/angiogram_frame/<patient_id>/<filename>')
def serve_preprocessed_frame(patient_id, filename):
    img_path = OUTPUT_ROOT / patient_id / filename
    if not img_path.exists():
        return "Frame not found", 404
    return send_file(str(img_path), mimetype='image/png')


# ====================================================================
#  Entry point
# ====================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)