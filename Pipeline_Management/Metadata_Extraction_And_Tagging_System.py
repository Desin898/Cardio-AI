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
    Writes updated metadata back to metadata.json.

    Before saving, it updates the 'last_updated' field to maintain
    an audit trail of changes.

    Parameters:
        metadata_path (Path): Path to metadata.json.
        data (Dict): Updated metadata content.
    """
    data["last_updated"] = datetime.utcnow().isoformat()

    with open(metadata_path, "w") as f:
        json.dump(data, f, indent=4)


def initialize_patient_session(patient_data: Dict, base_dir: str = "patients") -> Path:
    """
    Creates a patient session folder and initializes metadata.json.
    """

    patient_id = patient_data["patient_id"]
    patient_dir = Path(base_dir) / patient_id

    # Create structured folders
    (patient_dir / "ecg").mkdir(parents=True, exist_ok=True)
    (patient_dir / "angiogram").mkdir(parents=True, exist_ok=True)

    metadata = {
        "patient_id": patient_id,
        "schema_version": "1.0",
        "created_at": datetime.utcnow().isoformat(),
        "last_updated": datetime.utcnow().isoformat(),

        "patient_profile": patient_data,

        "risk_prediction": {},

        "ecg": {
            "uploaded": False
        },

        "angiogram": {
            "uploaded": False
        }
    }

    metadata_path = patient_dir / "metadata.json"
    _write_metadata(metadata_path, metadata)

    return metadata_path

def update_risk_prediction(
    metadata_path: Path,
    risk_level: str,
    risk_percentage: float,
    model_name: str = "XGBoost"
) -> None:

    metadata = _read_metadata(metadata_path)

    metadata["risk_prediction"] = {
        "risk_level": risk_level,
        "risk_percentage": risk_percentage,
        "model": model_name,
        "timestamp": datetime.utcnow().isoformat()
    }

    _write_metadata(metadata_path, metadata)

def update_ecg_metadata(
    metadata_path: Path,
    raw_image_path: str,
    processed_csv_path: str,
    preprocessing_steps: List[str],
    classification_result: str,
    model_name: str = "ECG_CNN"
) -> None:

    metadata = _read_metadata(metadata_path)

    metadata["ecg"] = {
        "uploaded": True,
        "raw_image_path": raw_image_path,
        "processed_csv_path": processed_csv_path,
        "preprocessing": {
            "steps": preprocessing_steps
        },
        "classification": {
            "patient_type": classification_result,
            "model": model_name,
            "timestamp": datetime.utcnow().isoformat()
        }
    }

    _write_metadata(metadata_path, metadata)

def update_angiogram_metadata(
    metadata_path: Path,
    dicom_path: str,
    selected_frame_paths: List[str],
    preprocessing_steps: List[str],
    segmentation_mask_path: str,
    segmentation_model: str = "DeepSA-based"
) -> None:

    metadata = _read_metadata(metadata_path)

    metadata["angiogram"] = {
        "uploaded": True,
        "dicom_path": dicom_path,
        "selected_frames": selected_frame_paths,
        "preprocessing": {
            "steps": preprocessing_steps
        },
        "segmentation": {
            "mask_path": segmentation_mask_path,
            "model": segmentation_model,
            "timestamp": datetime.utcnow().isoformat()
        }
    }

    _write_metadata(metadata_path, metadata)

def get_patient_profile(metadata_path: Path) -> Dict:
    metadata = _read_metadata(metadata_path)
    return metadata["patient_profile"]

def get_ecg_image_path(metadata_path: Path) -> str:
    metadata = _read_metadata(metadata_path)
    return metadata["ecg"].get("raw_image_path")

def get_risk_percentage(metadata_path: Path) -> float:
    metadata = _read_metadata(metadata_path)
    return metadata["risk_prediction"].get("risk_percentage")

def get_selected_frames(metadata_path: Path) -> List[str]:
    metadata = _read_metadata(metadata_path)
    return metadata["angiogram"].get("selected_frames", [])