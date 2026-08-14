from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field

class ECGInferenceResult(BaseModel):
    predicted_class: str = Field(default="Unknown", description="Predicted cardiac category")
    confidence_score: float = Field(default=0.0, description="Model prediction confidence (0.0 - 1.0)")
    risk_level: str = Field(default="UNKNOWN", description="Emergency priority level: HIGH, MEDIUM, LOW")
    category_probabilities: Dict[str, float] = Field(default_factory=dict, description="Probabilities for each cardiac condition class")
    suspected_artery: str = Field(default="N/A", description="Coronary artery territory suspected of acute occlusion (LAD, LCx, RCA)")
    affected_leads: List[str] = Field(default_factory=list, description="Leads demonstrating significant deviation/elevation")
    lead_analysis_breakdown: Dict[str, float] = Field(default_factory=dict, description="Individual lead variance/deviation scores")
    
    # Backward compatibility fields
    prediction: str = Field(default="Unknown", description="Alias for predicted_class")
    confidence: str = Field(default="0%", description="Formatted string representation of confidence")
    suspected_vessel: str = Field(default="N/A", description="Alias for suspected_artery")
    plot_path: Optional[str] = Field(default=None, description="Path to lead activity visualization plot")
    error: Optional[str] = Field(default=None, description="Error message if inference failed")


class ECGPredictRequest(BaseModel):
    patient_id: Optional[str] = Field(default="ANONYMOUS", description="Patient identification string")
    csv_path: Optional[str] = Field(default=None, description="Path to preprocessed CSV signal vector")
    signal_vector: Optional[List[float]] = Field(default=None, description="Direct list of 12-lead signal values")
    image_path: Optional[str] = Field(default=None, description="Path to raw ECG image file")


class ECGUploadResponse(BaseModel):
    success: bool
    session_id: str
    patient_id: str
    prediction: ECGInferenceResult
