import os
import warnings
from pathlib import Path
from pydantic import BaseModel

# Suppress scikit-learn model version unpickling warnings cleanly
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
warnings.filterwarnings("ignore", message=".*Trying to unpickle estimator.*")

class Settings(BaseModel):
    APP_TITLE: str = "Coronary AI Detection System"
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent / "Pipeline_Management" / "Patients"
    NEW_MODEL_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent / "Model" / "models"
    TEN_YEAR_MODEL_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent / "ten_year_models" / "ten_year_models"
    UPLOAD_FOLDER: Path = Path(__file__).resolve().parent.parent.parent.parent / "uploads"
    OUTPUT_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent / "Preprocessed_Angiogram_Output"
    SECRET_KEY_PATH: Path = Path(__file__).resolve().parent.parent.parent.parent / "secret.key"
    
    DEEPSA_PORT: int = int(os.environ.get("DEEPSA_PORT", 7860))
    DEEPSA_URL: str = os.environ.get("DEEPSA_URL", f"http://127.0.0.1:{os.environ.get('DEEPSA_PORT', 7860)}")
    DEEPSA_SCRIPT: Path = Path(__file__).resolve().parent.parent.parent.parent / "demo.py"

settings = Settings()

