from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field, ConfigDict, model_validator

class PatientScreeningInput(BaseModel):
    # Demographic & Vitals
    age: int = Field(..., ge=1, le=120, description="Patient age in years")
    gender: str = Field(..., description="Gender: Male/Female")
    systolic_bp: float = Field(..., ge=60, le=260, description="Systolic Blood Pressure (mmHg)")
    diastolic_bp: float = Field(..., ge=40, le=160, description="Diastolic Blood Pressure (mmHg)")
    bmi: float = Field(..., ge=10, le=70, description="Body Mass Index")
    current_smoker: bool = Field(..., description="Current smoker status (True/False)")

    # Metabolic & Biomarkers
    hba1c: Optional[float] = Field(default=None, description="HbA1c (%)")
    hs_troponin: Optional[float] = Field(default=None, description="High-sensitivity Troponin (ng/L)")
    egfr: Optional[float] = Field(default=None, description="eGFR (mL/min/1.73m^2)")
    cholesterol_total: float = Field(..., description="Total Cholesterol (mg/dL)")
    cholesterol_hdl: float = Field(..., description="HDL Cholesterol (mg/dL)")
    family_history_cad: bool = Field(..., description="Family history of CAD (True/False)")

    @model_validator(mode="before")
    @classmethod
    def handle_legacy_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Gender mapping: 1 / "1" / "Male" -> "Male", 0 / "0" / "Female" -> "Female"
            if "gender" in data:
                g = data["gender"]
                if g == 1 or g == "1" or str(g).lower() in ("male", "m"):
                    data["gender"] = "Male"
                elif g == 0 or g == "0" or str(g).lower() in ("female", "f"):
                    data["gender"] = "Female"

            # Smoker mapping: 1 / "1" / "Yes" / True -> True, 0 / "0" / "No" / False -> False
            if "current_smoker" not in data and "smoker" in data:
                sm = data["smoker"]
                if isinstance(sm, str):
                    data["current_smoker"] = sm.lower() in ("yes", "true", "1")
                else:
                    data["current_smoker"] = bool(sm)
            elif "current_smoker" in data:
                sm = data["current_smoker"]
                if isinstance(sm, str):
                    data["current_smoker"] = sm.lower() in ("yes", "true", "1")
                else:
                    data["current_smoker"] = bool(sm)

            # Family history mapping
            if "family_history_cad" not in data and "family_history" in data:
                fh = data["family_history"]
                if isinstance(fh, str):
                    data["family_history_cad"] = fh.lower() in ("yes", "true", "1")
                else:
                    data["family_history_cad"] = bool(fh)
            elif "family_history_cad" in data:
                fh = data["family_history_cad"]
                if isinstance(fh, str):
                    data["family_history_cad"] = fh.lower() in ("yes", "true", "1")
                else:
                    data["family_history_cad"] = bool(fh)

            # Cholesterol & Troponin mapping
            if "cholesterol_total" not in data and "cholesterol" in data:
                data["cholesterol_total"] = float(data["cholesterol"])
            if "cholesterol_hdl" not in data and "hdl_cholesterol" in data:
                data["cholesterol_hdl"] = float(data["hdl_cholesterol"])
            if "cholesterol_hdl" not in data:
                data["cholesterol_hdl"] = 50.0

            if "hs_troponin" not in data and "troponin" in data:
                data["hs_troponin"] = float(data["troponin"]) if data["troponin"] is not None else None

        return data

    model_config = ConfigDict(extra="ignore")

class SHAPFactor(BaseModel):
    factor: str
    impact: float
    value: str
    effect: str

class PrescreeningResponse(BaseModel):
    success: bool = True
    risk_category: str
    risk_probability: float
    probability_percentage: float
    shap_breakdown: List[Dict[str, Any]] = []
    recommended_next_path: str

    # Backward compatibility fields
    risk_status: Optional[str] = None
    decision: Optional[str] = None
    message: Optional[str] = None
    explanation: List[Dict[str, Any]] = []
    patient_data: Dict[str, Any] = {}
    ten_year_result: Optional[Dict[str, Any]] = None
    ten_year_model_available: bool = True
    next_step: Optional[str] = None
    requires_ecg: Optional[bool] = None
    color: Optional[str] = None

class TenYearRiskInput(BaseModel):
    age: float = 45.0
    male: int = 1
    currentSmoker: int = 0
    cigsPerDay: float = 0.0
    BPMeds: int = 0
    prevalentStroke: int = 0
    prevalentHyp: int = 0
    diabetes: int = 0
    totChol: float = 200.0
    sysBP: float = 120.0
    diaBP: float = 80.0
    pulsePressure: float = 40.0
    bp_ratio: float = 1.5
    BMI: float = 25.0
    heartRate: float = 72.0
    glucose: float = 100.0

class TenYearRiskResponse(BaseModel):
    success: bool
    percent: Optional[float] = None
    category: Optional[str] = None
    colour: Optional[str] = None
    probability: Optional[float] = None
    advice: Optional[str] = None
    message: Optional[str] = None

class RecommendationAction(BaseModel):
    title: str
    icon: str
    colour: str
    priority: str
    advice: str
    actions: List[str]
    rationale: str
    probability: Optional[float] = None

class RecommendationResponse(BaseModel):
    success: bool
    timestamp: str
    risk_assessment: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    total: int
    shap_top_factors: List[Dict[str, Any]]
    graph_features: Dict[str, Any]

class AISummaryRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, description="Optional patient session ID to auto-load engine outputs")
    prescreening_output: Optional[Dict[str, Any]] = Field(default=None, description="JSON output from PrescreeningEngine")
    ecg_output: Optional[Dict[str, Any]] = Field(default=None, description="JSON output from ECGCNNEngine")
    deepsa_output: Optional[Dict[str, Any]] = Field(default=None, description="JSON output from DeepSAEngine")

class AISummaryResponse(BaseModel):
    success: bool
    clinical_summary: str = Field(..., description="Professional 3-sentence clinical summary")
    patient_next_steps: str = Field(..., description="Patient-friendly next steps")
    model_used: str = Field(default="gemini-1.5-pro", description="Model used for generation")
    error: Optional[str] = Field(default=None, description="Error message if generation failed or key is missing")

