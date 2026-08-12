import json
from pathlib import Path
from typing import Dict, List, Optional
from Pipeline_Management.Metadata_Extraction_And_Tagging_System import (
    MetadataManager,
    PatientSessionManager,
    StorageManager,
    RetrievalManager,
)
from backend.app.core.config import settings

class PatientService:
    @staticmethod
    def find_session_path(session_id: str) -> Optional[Path]:
        base_path = settings.BASE_DIR
        if not base_path.exists():
            return None
        for patient_dir in base_path.iterdir():
            if not patient_dir.is_dir():
                continue
            candidate = patient_dir / "sessions" / session_id
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def initialize_session(patient_data: Dict) -> Path:
        return PatientSessionManager.initialize_patient_session(
            patient_data, base_dir=str(settings.BASE_DIR)
        )

    @staticmethod
    def read_metadata(metadata_path: Path) -> Dict:
        return MetadataManager.read_metadata(metadata_path)

    @staticmethod
    def write_metadata(metadata_path: Path, data: Dict) -> None:
        MetadataManager.write_metadata(metadata_path, data)

    @staticmethod
    def update_ecg_metadata(metadata_path: Path, raw_image_path: str, processed_csv_path: str, classification_result: str):
        MetadataManager.update_ecg_metadata(
            metadata_path,
            raw_image_path=raw_image_path,
            processed_csv_path=processed_csv_path,
            classification_result=classification_result,
        )

    @staticmethod
    def update_risk_prediction(metadata_path: Path, risk_level: str, risk_percentage: float):
        MetadataManager.update_risk_prediction(metadata_path, risk_level, risk_percentage)

    @staticmethod
    def save_encrypted_ecg(metadata_path: Path, file_bytes: bytes, filename: str) -> str:
        return StorageManager.save_encrypted_ecg(metadata_path, file_bytes, filename)

    @staticmethod
    def save_encrypted_angiogram(metadata_path: Path, file_bytes: bytes, filename: str) -> str:
        return StorageManager.save_encrypted_angiogram(metadata_path, file_bytes, filename)

patient_service = PatientService()
