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
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from cryptography.fernet import Fernet


"""
Handles secure encryption and decryption of medical files.
Manages encryption key generation and protects sensitive ECG and angiogram data before storage.
"""

class EncryptionManager:

    def __init__(self, key_path: str = "secret.key"):
        self.key = self._load_or_create_key(key_path)
        self.cipher = Fernet(self.key)

    def _load_or_create_key(self, key_path: str) -> bytes:
        """
        Loads encryption key from file.
        If not exists, creates one and saves it.
        """
        if os.path.exists(key_path):
            with open(key_path, "rb") as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_path, "wb") as f:
                f.write(key)
            return key

    def encrypt_and_save(self, file_bytes: bytes, save_path: Path) -> Path:
        """
        Encrypts binary content and saves as .enc file.
        """
        encrypted_data = self.cipher.encrypt(file_bytes)

        encrypted_path = save_path.with_suffix(".enc")

        with open(encrypted_path, "wb") as f:
            f.write(encrypted_data)

        return encrypted_path

    def decrypt_file(self, encrypted_path: Path) -> bytes:
        """
        Decrypts encrypted file.
        """
        with open(encrypted_path, "rb") as f:
            encrypted_data = f.read()

        return self.cipher.decrypt(encrypted_data)

"""
Responsible for managing metadata.json operations.
Handles reading, writing, and updating structured patient metadata throughout the pipeline.
"""

class MetadataManager:
    def read_metadata(metadata_path: Path) -> Dict:
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

    def write_metadata(metadata_path: Path, data: Dict) -> None:
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

    #Update risk details of the patient
    def update_risk_prediction(
        metadata_path: Path,
        risk_level: str,
        risk_percentage: float,
    ) -> None:
        """
        Updates the latest risk prediction result.
        Preserves structure and allows future extensions.
        """

        metadata = MetadataManager.read_metadata(metadata_path)

        metadata.setdefault("risk_prediction", {})

        metadata["risk_prediction"]["latest"] = {
            "risk_level": risk_level,
            "risk_percentage": risk_percentage,
            "timestamp": datetime.utcnow().isoformat()
        }

        MetadataManager.write_metadata(metadata_path, metadata)


    #Update ecg details of the patient
    def update_ecg_metadata(
        metadata_path: Path,
        raw_image_path: str,
        processed_csv_path: str,
        classification_result: str,
    ) -> None:
        """
        Updates ECG-related metadata without overwriting existing fields.
        Ensures scalability and future extensibility.
        """

        metadata = MetadataManager.read_metadata(metadata_path)
        metadata.setdefault("ecg", {})

        metadata["ecg"].update({
            "uploaded": True,
            "raw_image_path": raw_image_path,
            "processed_csv_path": processed_csv_path,
            "classification": {
                "patient_type": classification_result,
                "timestamp": datetime.utcnow().isoformat()
            }
        })

        MetadataManager.write_metadata(metadata_path, metadata)
        metadata = MetadataManager.read_metadata(metadata_path)


    # Update Angiogram details of the patient
    def update_angiogram_metadata(
        metadata_path: Path,
        dicom_path: str,
        selected_frame_paths: List[str],
        segmentation_mask_path: str,
    ) -> None:

        """
        Updates angiogram-related metadata safely.
        Avoids overwriting unrelated data fields.
        """

        metadata = MetadataManager.read_metadata(metadata_path)

        metadata.setdefault("angiogram", {})

        metadata["angiogram"].update({
            "uploaded": True,
            "dicom_path": dicom_path,
            "selected_frames": selected_frame_paths,
            "segmentation": {
                "mask_path": segmentation_mask_path,
                "timestamp": datetime.utcnow().isoformat()
            }
        })

        MetadataManager.write_metadata(metadata_path, metadata)

"""
Manages patient session lifecycle and validation.
Creates session directories and ensures demographic consistency across multiple visits.
"""

class PatientSessionManager:

    # Add validation function
    def validate_patient_data(patient_data: Dict) -> None:
        """
        Validates required patient fields before session creation.
        Ensures data consistency and prevents incomplete records.
        """

        required_fields = ["patient_id", "age", "gender"]

        for field in required_fields:
            if field not in patient_data:
                raise ValueError(f"Missing required field: {field}")

    def validate_existing_patient(base_dir: str, patient_data: Dict) -> None:
        """
        Ensures that if a patient already exists, their demographic
        information matches previous records.
        """

        patient_root = Path(base_dir) / patient_data["patient_id"]

        # If patient folder does not exist, this is a new patient → OK
        if not patient_root.exists():
            return

        sessions_dir = patient_root / "sessions"

        # If no sessions exist yet, allow creation
        if not sessions_dir.exists():
            return

        # Get any existing metadata.json (first session is enough)
        metadata_files = list(sessions_dir.glob("*/metadata.json"))

        if not metadata_files:
            return

        existing_metadata = MetadataManager.read_metadata(metadata_files[0])
        existing_profile = existing_metadata.get("patient_profile", {})

        # Compare immutable demographic fields
        if (
                existing_profile.get("age") != patient_data.get("age")
                or existing_profile.get("gender") != patient_data.get("gender")
        ):
            raise ValueError(
                "Patient demographic mismatch for existing patient_id."
            )

    # Patient Session Initialization
    # Multi-Session support
    def initialize_patient_session(patient_data: Dict, base_dir: str = "patients") -> Path:
        """
        Creates a unique session folder per patient visit.
        Supports scalability for multiple consultations.
        """

        PatientSessionManager.validate_patient_data(patient_data)
        PatientSessionManager.validate_existing_patient(base_dir, patient_data)

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
        MetadataManager.write_metadata(metadata_path, metadata)

        return metadata_path

"""
Handles encrypted storage of uploaded medical files.
Stores ECG images and angiogram data securely within the patient session directory.
"""
class StorageManager:
    # ENCRYPTED SAVE HELPERS
    def save_encrypted_ecg(metadata_path: Path, file_bytes: bytes, filename: str) -> str:
        session_dir = metadata_path.parent
        ecg_dir = session_dir / "ecg"

        save_path = ecg_dir / filename
        enc = EncryptionManager()
        encrypted_path = enc.encrypt_and_save(file_bytes, save_path)

        return str(encrypted_path)

    def save_encrypted_angiogram(metadata_path: Path, file_bytes: bytes, filename: str) -> str:
        session_dir = metadata_path.parent
        angio_dir = session_dir / "angiogram"

        save_path = angio_dir / filename
        enc = EncryptionManager()
        encrypted_path = enc.encrypt_and_save(file_bytes, save_path)

        return str(encrypted_path)

"""
Provides helper methods for retrieving stored patient information.
Allows Flask or other modules to access metadata, ECG paths, and angiogram frames easily.
"""
class RetrievalManager:
    # Retrieval Functions
    def get_patient_profile(metadata_path: Path) -> Dict:
        metadata = MetadataManager.read_metadata(metadata_path)
        return metadata.get("patient_profile", {})

    def get_ecg_image_path(metadata_path: Path) -> str:
        metadata = MetadataManager.read_metadata(metadata_path)
        return metadata.get("ecg", {}).get("raw_image_path")

    def get_ecg_image_path_csv(metadata_path: Path) -> str:
        metadata = MetadataManager.read_metadata(metadata_path)
        return metadata.get("ecg", {}).get("processed_csv_path")

    def get_risk_percentage(metadata_path: Path) -> float:
        metadata = MetadataManager.read_metadata(metadata_path)
        return metadata.get("risk_prediction", {}) \
            .get("latest", {}) \
            .get("risk_percentage")

    def get_selected_frames(metadata_path: Path) -> List[str]:
        metadata = MetadataManager.read_metadata(metadata_path)
        return metadata.get("angiogram", {}).get("selected_frames", [])