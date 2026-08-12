from typing import Optional, List, Dict, Any
from pydantic import BaseModel

class AngiogramConfirmRequest(BaseModel):
    session_id: Optional[str] = ""
    patient_id: Optional[str] = ""
    selected_variant: str

class AngiogramUploadResponse(BaseModel):
    success: bool
    session_id: str
    patient_id: str
    variants_count: int
    redirect_url: str
