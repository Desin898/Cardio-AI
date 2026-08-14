from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class QCAMetricsResponse(BaseModel):
    stenosis_percentage: float = Field(default=0.0, description="Calculated percentage stenosis (0 - 100%)")
    severity_grade: str = Field(default="MILD", description="Lesion severity grade: MILD (<50%), MODERATE (50-69%), SEVERE (>=70%)")
    d_min: float = Field(default=0.0, description="Minimum lumen diameter at narrowest bottleneck")
    d_ref: float = Field(default=0.0, description="Reference vessel diameter")
    lesion_coordinates: Dict[str, int] = Field(default_factory=dict, description="Pixel coordinates (x, y) of narrowest bottleneck")
    intervention_recommended: bool = Field(default=False, description="True if catheter intervention is recommended (Stenosis >= 70%)")


class QCAAnalysisResult(BaseModel):
    success: bool = Field(default=True, description="Whether QCA processing succeeded")
    session_id: str = Field(default="", description="Session identification string")
    patient_id: str = Field(default="", description="Patient identification string")
    qca_metrics: QCAMetricsResponse = Field(default_factory=QCAMetricsResponse, description="Quantitative Coronary Angiography metrics")
    qca_image_url: Optional[str] = Field(default=None, description="URL or endpoint path to annotated QCA visual diagnostic image")
    keyframe_url: Optional[str] = Field(default=None, description="URL to extracted keyframe image")
    error: Optional[str] = Field(default=None, description="Error message if analysis failed")


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
