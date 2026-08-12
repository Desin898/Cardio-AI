from fastapi import APIRouter, HTTPException, status
from backend.app.schemas.patient import (
    PatientScreeningInput,
    PrescreeningResponse,
    TenYearRiskInput,
    TenYearRiskResponse,
    RecommendationResponse,
)
from backend.app.engines.prescreening_engine import prescreening_engine
from backend.app.engines.ten_year_risk_engine import ten_year_risk_engine

router = APIRouter(prefix="/prescreening", tags=["prescreening"])

@router.post("/predict", response_model=PrescreeningResponse)
def predict(input_data: PatientScreeningInput):
    try:
        data_dict = input_data.model_dump()
        base_res = prescreening_engine.predict(data_dict)
        ten_year_res = ten_year_risk_engine.predict(data_dict)
        
        base_res["ten_year_result"] = ten_year_res
        base_res["ten_year_model_available"] = (ten_year_risk_engine.ten_year_model is not None)
        
        return base_res
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"/predict failed: {str(e)}",
        )

@router.post("/predict_ten_year", response_model=TenYearRiskResponse)
def predict_ten_year(input_data: TenYearRiskInput):
    res = ten_year_risk_engine.predict(input_data.model_dump())
    if not res.get("success"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=res.get("message", "10-year risk prediction failed"),
        )
    return res

@router.post("/recommend", response_model=RecommendationResponse)
def recommend(input_data: PatientScreeningInput):
    res = prescreening_engine.get_recommendations(input_data.model_dump())
    if not res.get("success"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=res.get("message", "Recommendation generation failed"),
        )
    return res
