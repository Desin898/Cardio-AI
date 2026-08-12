from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field, ConfigDict

class PatientScreeningInput(BaseModel):
    age: float = Field(..., ge=1, le=120, description="Patient age in years")
    gender: str = Field(..., description="Gender: Male/Female")
    bmi: float = Field(..., ge=10, le=70, description="Body Mass Index")
    systolic_bp: float = Field(..., ge=60, le=260, description="Systolic Blood Pressure (mmHg)")
    diastolic_bp: float = Field(..., ge=40, le=160, description="Diastolic Blood Pressure (mmHg)")
    cholesterol: float = Field(..., ge=50, le=600, description="Total Cholesterol (mg/dL)")
    glucose: float = Field(..., ge=40, le=500, description="Fasting Glucose (mg/dL)")
    smoker: str = Field(..., description="Smoker status: Yes/No/Former")
    exercise_hours: float = Field(..., ge=0, le=50, description="Exercise hours per week")
    heart_rate: Optional[float] = Field(default=72.0, description="Heart rate (bpm)")
    bp_treatment: Optional[str] = Field(default="No", description="Blood pressure treatment: Yes/No")
    previous_stroke: Optional[str] = Field(default="No", description="Previous stroke: Yes/No")
    prevalent_hyp: Optional[str] = Field(default="No", description="Prevalent hypertension: Yes/No")
    cigs_per_day: Optional[float] = Field(default=0.0, description="Cigarettes per day")

    model_config = ConfigDict(extra="ignore")

class SHAPFactor(BaseModel):
    factor: str
    impact: float
    value: str
    effect: str

class PrescreeningResponse(BaseModel):
    success: bool
    risk_probability: float
    risk_status: str
    decision: str
    message: str
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
