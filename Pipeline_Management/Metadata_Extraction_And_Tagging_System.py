"""Metadata Extraction and Tagging System.ipynb

# Metadata Extraction and Tagging System (Process 01)

This notebook implements the metadata extraction and tagging functionality
described in **Process 01 – ECG and Angiogram Preprocessing System**.

The system stores all patient-entered data, preprocessing outputs, and
downstream model results in a **single structured JSON file per patient session**,
ensuring traceability and seamless integration with a web-based application.

## System Interaction Stages

The patient interacts with the system in three stages:

1. **Stage 01 – Patient Data Entry**
   - Patient enters demographic, clinical, and lifestyle information via a web UI.

2. **Stage 02 – ECG Upload and Analysis**
   - Patient uploads ECG image.
   - ECG preprocessing and classification results are attached to metadata.

3. **Stage 03 – Angiogram Upload and Analysis**
   - Patient uploads angiogram (DICOM).
   - Key-frame extraction, preprocessing, and segmentation outputs are attached.

All information is incrementally stored in **one JSON file per patient**.

## Step 01: Initialize Patient Metadata (From Web Application)

Patient demographic and clinical information is collected through a web
application (Flask-based UI).

This metadata is considered **authoritative**, as it is directly entered
by the patient or clinician, rather than inferred from images.

A single metadata JSON file is created at this stage.
"""

import json
from pathlib import Path
from datetime import datetime

def initialize_patient_metadata(patient_data: dict, base_dir="patients"):
    """
    Initializes metadata.json for a new patient session.
    """
    patient_id = patient_data["patient_id"]
    patient_dir = Path(base_dir) / f"patient_{patient_id}"
    patient_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "patient_id": patient_id,
        "created_at": datetime.utcnow().isoformat(),
        "patient_data": patient_data,
        "ecg": {},
        "angiogram": {}
    }

    metadata_path = patient_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    return metadata_path


def update_ecg_metadata(
    metadata_path,
    preprocessing_steps,
    classification_result,
    model_name="ECG_CNN"
):
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    metadata["ecg"] = {
        "uploaded": True,
        "preprocessing": {
            "steps": preprocessing_steps,
            "output_format": "CSV"
        },
        "classification": {
            "patient_type": classification_result,
            "model": model_name,
            "timestamp": datetime.utcnow().isoformat()
        }
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

"""## Step 03: Attach Angiogram Processing & Segmentation Metadata

If the ECG-based risk exceeds the clinical threshold, the patient uploads
an angiogram video (DICOM).

Key-frame selection, preprocessing, and segmentation results are recorded
to ensure traceability and integration with the blockage detection module.
"""

def update_angiogram_metadata(
    metadata_path,
    selected_frames,
    preprocessing_steps,
    segmentation_model="DeepSA-based"
):
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    metadata["angiogram"] = {
        "uploaded": True,
        "dicom_processing": {
            "frame_selection_method": "Top-K diagnostic scoring",
            "selected_frames": selected_frames
        },
        "preprocessing": {
            "steps": preprocessing_steps
        },
        "segmentation": {
            "performed": True,
            "model": segmentation_model,
            "timestamp": datetime.utcnow().isoformat()
        }
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

"""## Final Notes

- A **single metadata.json file** acts as the backbone of Process 01.
- Each pipeline stage updates the same file.
- This design avoids databases and server-side complexity.
- The system is easily integrated with Flask by calling these functions
  inside appropriate API endpoints.

This implementation satisfies the **Metadata Extraction and Tagging**
requirement of Process 01.
"""