"""
Metadata Extraction and Tagging System

This module acts as the centralized metadata controller for the system.

Purpose:
- Connects the Flask frontend with ML models and file storage.
- Maintains one structured metadata.json file per patient session.
- Stores structured information only.
- Stores file paths for binary data (images, DICOM, CSV), not the files themselves.

patients/
   └── P001/
        ├── ecg/
        ├── angiogram/
        └── metadata.json
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


def _read_metadata(metadata_path: Path) -> Dict:
    """
    Reads the metadata.json file and returns its content as a dictionary.

    This function ensures that metadata is consistently loaded from disk
    before any update or retrieval operation.

    Parameters:
        metadata_path (Path): Path to the metadata.json file.

    Returns:
        Dict: Parsed JSON metadata.

    Raises:
        FileNotFoundError: If metadata.json does not exist.
    """
    if not metadata_path.exists():
        raise FileNotFoundError("Metadata file does not exist.")

    with open(metadata_path, "r") as f:
        return json.load(f)


def _write_metadata(metadata_path: Path, data: Dict) -> None:
    """
    Writes updated metadata safely using atomic replacement.
    Prevents corruption during unexpected shutdowns.
    """

    data["last_updated"] = datetime.utcnow().isoformat()

    temp_path = metadata_path.with_suffix(".tmp")

    with open(temp_path, "w") as f:
        json.dump(data, f, indent=4)

    # Atomic replace
    temp_path.replace(metadata_path)

# Add validation function
def _validate_patient_data(patient_data: Dict) -> None:
    """
    Validates required patient fields before session creation.
    Ensures data consistency and prevents incomplete records.
    """

    required_fields = ["patient_id", "age", "gender"]

    for field in required_fields:
        if field not in patient_data:
            raise ValueError(f"Missing required field: {field}")



#Patient Session Initialization
# Multi-Session support
def initialize_patient_session(patient_data: Dict, base_dir: str = "patients") -> Path:
    """
    Creates a unique session folder per patient visit.
    Supports scalability for multiple consultations.
    """

    _validate_patient_data(patient_data)

    patient_id = patient_data["patient_id"]

    session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(base_dir) / patient_id / "sessions" / session_id

    (session_dir / "ecg").mkdir(parents=True, exist_ok=True)
    (session_dir / "angiogram").mkdir(parents=True, exist_ok=True)

    metadata = {
        "patient_id": patient_id,
        "session_id": session_id,
        "schema_version": "1.1",
        "created_at": datetime.utcnow().isoformat(),
        "last_updated": datetime.utcnow().isoformat(),
        "patient_profile": patient_data,
        "risk_prediction": {},
        "ecg": {"uploaded": False},
        "angiogram": {"uploaded": False}
    }

    metadata_path = session_dir / "metadata.json"
    _write_metadata(metadata_path, metadata)

    return metadata_path

#Update risk details of the patient
def update_risk_prediction(
    metadata_path: Path,
    risk_level: str,
    risk_percentage: float,
    model_name: str = "XGBoost"
) -> None:
    """
    Updates the latest risk prediction result.
    Preserves structure and allows future extensions.
    """

    metadata = _read_metadata(metadata_path)

    metadata.setdefault("risk_prediction", {})

    metadata["risk_prediction"]["latest"] = {
        "risk_level": risk_level,
        "risk_percentage": risk_percentage,
        "model": model_name,
        "timestamp": datetime.utcnow().isoformat()
    }

    _write_metadata(metadata_path, metadata)


#Update ecg details of the patient
def update_ecg_metadata(
    metadata_path: Path,
    raw_image_path: str,
    processed_csv_path: str,
    preprocessing_steps: List[str],
    classification_result: str,
    model_name: str = "ECG_CNN"
) -> None:
    """
    Updates ECG-related metadata without overwriting existing fields.
    Ensures scalability and future extensibility.
    """

    metadata = _read_metadata(metadata_path)

    # Ensure ECG block exists
    metadata.setdefault("ecg", {})

    metadata["ecg"]["uploaded"] = True
    metadata["ecg"]["raw_image_path"] = raw_image_path
    metadata["ecg"]["processed_csv_path"] = processed_csv_path

    metadata["ecg"]["preprocessing"] = {
        "steps": preprocessing_steps,
        "timestamp": datetime.utcnow().isoformat()
    }

    metadata["ecg"]["classification"] = {
        "patient_type": classification_result,
        "model": model_name,
        "timestamp": datetime.utcnow().isoformat()
    }

    _write_metadata(metadata_path, metadata)
    metadata = _read_metadata(metadata_path)


#Update Angiogram details of the patient
def update_angiogram_metadata(
    metadata_path: Path,
    dicom_path: str,
    selected_frame_paths: List[str],
    preprocessing_steps: List[str],
    segmentation_mask_path: str,
    segmentation_model: str = "DeepSA-based"
) -> None:
    """
    Updates angiogram-related metadata safely.
    Avoids overwriting unrelated data fields.
    """

    metadata = _read_metadata(metadata_path)

    metadata.setdefault("angiogram", {})

    metadata["angiogram"]["uploaded"] = True
    metadata["angiogram"]["dicom_path"] = dicom_path
    metadata["angiogram"]["selected_frames"] = selected_frame_paths

    metadata["angiogram"]["preprocessing"] = {
        "steps": preprocessing_steps,
        "timestamp": datetime.utcnow().isoformat()
    }

    metadata["angiogram"]["segmentation"] = {
        "mask_path": segmentation_mask_path,
        "model": segmentation_model,
        "timestamp": datetime.utcnow().isoformat()
    }

    _write_metadata(metadata_path, metadata)

# Retrieval Functions
def get_patient_profile(metadata_path: Path) -> Dict:
    metadata = _read_metadata(metadata_path)
    return metadata.get("patient_profile", {})

def get_ecg_image_path(metadata_path: Path) -> str:
    metadata = _read_metadata(metadata_path)
    return metadata.get("ecg", {}).get("raw_image_path")

def get_risk_percentage(metadata_path: Path) -> float:
    metadata = _read_metadata(metadata_path)
    return metadata.get("risk_prediction", {}).get("risk_percentage")

def get_selected_frames(metadata_path: Path) -> List[str]:
    metadata = _read_metadata(metadata_path)
    return metadata.get("angiogram", {}).get("selected_frames", [])