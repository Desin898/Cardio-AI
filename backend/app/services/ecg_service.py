import os
import tempfile
import pandas as pd
from pathlib import Path
from Preprocessing.ECG_Preprocessing.ECG_Preprocessing_Functions import (
    process_single_ecg_image,
    TARGET_LEAD_LENGTH,
)

class ECGService:
    @staticmethod
    def preprocess_ecg_file(file_bytes: bytes, filename: str, ecg_folder: Path) -> Path:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            vector = process_single_ecg_image(tmp_path, target_len=TARGET_LEAD_LENGTH)
            if vector is None:
                raise ValueError("ECG signal extraction failed")

            csv_filename = f"{Path(filename).stem}_preprocessed.csv"
            csv_path = ecg_folder / csv_filename
            columns = [
                f"Lead{lead}_{i}"
                for lead in range(1, 13)
                for i in range(TARGET_LEAD_LENGTH)
            ]
            pd.DataFrame([vector], columns=columns).to_csv(csv_path, index=False)
            return csv_path
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

ecg_service = ECGService()
