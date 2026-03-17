from Preprocessing.ECG_Preprocessing.ECG_Preprocessing_Functions import (
    process_single_ecg_image,
    TARGET_LEAD_LENGTH
)
from flask import Flask, request, jsonify, render_template, redirect, url_for
import pandas as pd
import numpy as np
import pickle
import shap
import os
from pathlib import Path

# Import metadata system
from Pipeline_Management.metadata_system import (
    PatientSessionManager,
    StorageManager,
    MetadataManager,
    EncryptionManager,
    RetrievalManager
)

app = Flask(__name__)

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
BASE_DIR = "Pipeline_Management/Patients"          # Root folder for all patient data
MODEL_PATH = 'gateway_model.pkl'
SCALER_PATH = 'gateway_scaler.pkl'
ENCODERS_PATH = 'gateway_encoders.pkl'
FEATURES_PATH = 'gateway_features.pkl'

# Load model artifacts
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)
with open(ENCODERS_PATH, 'rb') as f:
    encoders = pickle.load(f)
with open(FEATURES_PATH, 'rb') as f:
    GATEWAY_FEATURES = pickle.load(f)

explainer = shap.TreeExplainer(model)

# -------------------------------------------------------------------
# Helper: prepare feature vector from JSON
# -------------------------------------------------------------------
def prepare_features(data):
    high_chol = 1 if data['cholesterol'] > 200 else 0
    high_glucose = 1 if data['glucose'] > 100 else 0
    hypertension = 1 if (data['systolic_bp'] >= 140 or data['diastolic_bp'] >= 90) else 0
    obesity = 1 if data['bmi'] >= 30 else 0

    risk_score = sum([
        data['age'] > 50,
        data['smoker'] == 'Yes',
        data['diabetes'] == 'Yes',
        high_chol,
        hypertension,
        obesity,
        data['exercise_hours'] < 2
    ])

    # Encode categoricals
    gender_enc = encoders['Gender'].transform([data['gender']])[0]
    smoker_enc = encoders['Smoker'].transform([data['smoker']])[0]
    alcohol_enc = encoders['Alcohol'].transform([data['alcohol']])[0]
    activity_enc = encoders['Physical_Activity'].transform([data['physical_activity']])[0]
    stress_enc = encoders['Stress_Level'].transform([data['stress_level']])[0]

    feature_dict = {
        'Age': data['age'],
        'Gender_Encoded': gender_enc,
        'BMI': data['bmi'],
        'Systolic_BP': data['systolic_bp'],
        'Diastolic_BP': data['diastolic_bp'],
        'Cholesterol mg/dL': data['cholesterol'],
        'Glucose mg/dL': data['glucose'],
        'Smoker_Encoded': smoker_enc,
        'Diabetes': 1 if data['diabetes'] == 'Yes' else 0,
        'Alcohol_Encoded': alcohol_enc,
        'Physical_Activity_Encoded': activity_enc,
        'Family_History': 1 if data['family_history'] == 'Yes' else 0,
        'Stress_Level_Encoded': stress_enc,
        'Sleep_Hours': data['sleep_hours'],
        'Years_Smoking': data['years_smoking'],
        'Exercise hours/week': data['exercise_hours'],
        'High_Cholesterol': high_chol,
        'High_Glucose': high_glucose,
        'Hypertension': hypertension,
        'Obesity': obesity,
        'Risk_Score': risk_score
    }

    df = pd.DataFrame([feature_dict])[GATEWAY_FEATURES]
    return df

# -------------------------------------------------------------------
# Routes for all pages
# -------------------------------------------------------------------
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

@app.route('/doctor_login.html')
def doctor_login():
    return render_template('doctor_login.html')

@app.route('/doctor_upload.html')
def doctor_upload():
    return render_template('doctor_upload.html')

@app.route('/doctor_analysis_results.html')
def doctor_analysis_results():
    return render_template('doctor_analysis_results.html')

@app.route('/upload_ecg.html', methods=['GET'])
def upload_page():
    return render_template('upload_ecg.html')

@app.route('/pre_screening.html')
def screening():
    return render_template('pre_screening.html')

@app.route('/pre_screening_results.html')
def result():
    return render_template('pre_screening_results.html')

# -------------------------------------------------------------------
# ECG Upload Handler (POST)
# -------------------------------------------------------------------


@app.route('/upload_success/<session_id>')
def upload_success(session_id):
    return f"ECG uploaded successfully. Session ID: {session_id}"

# -------------------------------------------------------------------
# Prediction endpoint (used by pre_screening.html)
# -------------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_json = request.get_json()
        required = ['age','gender','bmi','systolic_bp','diastolic_bp',
                    'cholesterol','glucose','smoker','years_smoking',
                    'exercise_hours','diabetes','alcohol','physical_activity',
                    'family_history','stress_level','sleep_hours']
        for field in required:
            if field not in input_json:
                return jsonify({'success': False, 'message': f'Missing field: {field}'}), 400

        X = prepare_features(input_json)
        X_scaled = scaler.transform(X)

        prob = model.predict_proba(X_scaled)[0][1]
        pred_class = model.predict(X_scaled)[0]

        # SHAP explanations
        shap_values = explainer.shap_values(X_scaled)
        if isinstance(shap_values, list):
            class_1_impacts = shap_values[1][0]
        else:
            class_1_impacts = shap_values[0, :, 1]

        contributions = [(GATEWAY_FEATURES[i], class_1_impacts[i], X.iloc[0][GATEWAY_FEATURES[i]])
                         for i in range(len(GATEWAY_FEATURES))]
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        top_factors = []
        for feature, impact, value in contributions[:3]:
            effect = "higher risk" if impact > 0 else "lower risk"
            display_value = value
            if '_Encoded' in feature:
                orig = feature.replace('_Encoded', '')
                if orig in encoders:
                    try:
                        display_value = encoders[orig].inverse_transform([int(value)])[0]
                    except:
                        pass
            top_factors.append({
                'factor': feature.replace('_Encoded', '').replace('_', ' '),
                'impact': round(float(impact), 3),
                'value': str(display_value),
                'effect': effect
            })

        # Optionally store risk in metadata if patient_id is provided
        # (You could extend this by expecting patient_id in the JSON)
        # For now, just return result

        if pred_class == 1:
            result = {
                'success': True,
                'risk_status': 'HIGH_RISK',
                'risk_probability': float(prob),
                'decision': 'MANDATORY_ECG',
                'message': "You show signs of elevated CAD risk based on your profile. Please upload your ECG image for detailed analysis.",
                'explanation': top_factors,
                'next_step': 'UPLOAD_ECG',
                'requires_ecg': True,
                'color': 'red'
            }
        else:
            result = {
                'success': True,
                'risk_status': 'LOW_RISK',
                'risk_probability': float(prob),
                'decision': 'OPTIONAL_ECG',
                'message': "You do not currently show significant CAD risk. You may upload an ECG for additional verification if you wish.",
                'explanation': top_factors,
                'next_step': 'OPTIONAL_UPLOAD',
                'requires_ecg': False,
                'color': 'green'
            }

        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/upload_ecg', methods=['POST'])
def upload_ecg():
    """
    Handles ECG file upload, creates patient session, stores encrypted file,
    preprocesses the image, saves the CSV, and updates metadata.
    """
    try:
        # Extract form data
        patient_id = request.form['patient_id']
        age = int(request.form['age'])
        gender = request.form['gender']
        notes = request.form.get('notes', '')
        ecg_type = request.form.get('ecg_type', 'unknown')
        file = request.files['ecg_file']

        if not file:
            return "No file uploaded", 400

        # Prepare patient data dictionary
        patient_data = {
            "patient_id": patient_id,
            "age": age,
            "gender": gender,
            "notes": notes,
            "ecg_type": ecg_type
        }

        # Initialize patient session (creates folders and metadata.json)
        metadata_path = PatientSessionManager.initialize_patient_session(
            patient_data, base_dir=BASE_DIR
        )

        # Read file bytes
        file_bytes = file.read()
        filename = file.filename

        # Save encrypted ECG
        encrypted_path = StorageManager.save_encrypted_ecg(
            metadata_path, file_bytes, filename
        )

        # ---- NEW: Preprocessing ----
        # 1. Write bytes to a temporary PNG file for preprocessing
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        # 2. Run the preprocessing pipeline
        vector = process_single_ecg_image(tmp_path, target_len=TARGET_LEAD_LENGTH)
        if vector is None:
            raise ValueError("Preprocessing failed for the uploaded image")

        # 3. Create DataFrame with proper column names
        columns = [f"Lead{lead}_{i}" for lead in range(1, 13) for i in range(TARGET_LEAD_LENGTH)]
        df = pd.DataFrame([vector], columns=columns)

        # 4. Save CSV inside the patient's session folder
        session_folder = metadata_path.parent
        csv_filename = f"{Path(filename).stem}_preprocessed.csv"
        csv_path = session_folder / csv_filename
        df.to_csv(csv_path, index=False)

        # 5. Clean up temporary file
        os.unlink(tmp_path)
        # ---- End of preprocessing ----

        # Update metadata with ECG information (including the CSV path)
        # TODO: Replace classification_result with actual model call
        classification_result = "pending"
        MetadataManager.update_ecg_metadata(
            metadata_path,
            raw_image_path=encrypted_path,
            processed_csv_path=str(csv_path),
            classification_result=classification_result
        )

        # Redirect to a success page
        return redirect(url_for('upload_success', session_id=session_folder.name))

    except Exception as e:
        return f"Upload failed: {str(e)}", 500

# -------------------------------------------------------------------
# Run the app
# -------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)


