from typing import Optional
from pydantic import BaseModel

class ECGInferenceResult(BaseModel):
    prediction: str = "Unknown"
    confidence: str = "0%"
    risk_level: str = "UNKNOWN"
    suspected_vessel: str = "N/A"
    plot_path: Optional[str] = None
    error: Optional[str] = None

class ECGUploadResponse(BaseModel):
    success: bool
    session_id: str
    patient_id: str
    prediction: ECGInferenceResult
