import os
import json
import shutil
import logging
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

from Preprocessing.Angiogram_Preprocessing.Angiogram_DICOM_KeyFrame_Extraction import process_angiogram
from backend.app.core.security import EncryptionManager

class AngiogramService:
    @staticmethod
    def process_raw_angiogram(file_bytes: bytes, raw_filename: str, preprocessed_root: Path) -> Dict[str, Any]:
        with tempfile.NamedTemporaryFile(
            suffix=Path(raw_filename).suffix, delete=False
        ) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            pipeline_result = process_angiogram(tmp_path, output_root=str(preprocessed_root))
            return pipeline_result
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @staticmethod
    def copy_frames_to_angio_folder(
        variants: List[Dict[str, Any]],
        preprocessed_folder: str,
        angio_folder: Path,
        metadata_file: Path,
    ) -> List[Dict[str, Any]]:
        enc = EncryptionManager()
        enriched = []
        for v in variants:
            src = next(Path(preprocessed_folder).rglob(v["filename"]), None)
            if not src:
                logging.warning(f"Frame not found: {v['filename']}")
                enriched.append({**v, "angio_folder_path": None})
                continue

            try:
                with open(src, "rb") as f:
                    file_bytes = f.read()

                encrypted_path = enc.encrypt_and_save(file_bytes, angio_folder / v["filename"])
                dst = Path(encrypted_path)
            except Exception:
                logging.warning(f"Could not process frame {src}", exc_info=True)
                dst = None

            enriched.append({**v, "angio_folder_path": str(dst) if dst else None})
        return enriched

    @staticmethod
    def write_frame_metadata(angio_folder: Path, payload: dict) -> Path:
        out_path = angio_folder / "angiogram_frame_metadata.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=4)
        logging.info(f"angiogram_frame_metadata.json written → {out_path}")
        return out_path

    @staticmethod
    def read_pipeline_metadata(preprocessed_folder: str) -> dict:
        try:
            path = Path(preprocessed_folder) / "metadata.json"
            if path.exists():
                with open(path, "r") as f:
                    return json.load(f)
        except Exception:
            logging.warning("Could not read pipeline metadata.json", exc_info=True)
        return {}

angiogram_service = AngiogramService()
