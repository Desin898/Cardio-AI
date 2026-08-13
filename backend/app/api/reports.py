import json
import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, status, Body

from backend.app.services.patient_service import patient_service
from backend.app.services.gemini_service import gemini_service
from backend.app.engines.prescreening_engine import prescreening_engine
from backend.app.schemas.patient import AISummaryRequest, AISummaryResponse

router = APIRouter(prefix="/reports", tags=["reports"])

@router.get("/get_screening_data/{session_id}")
async def get_screening_data(session_id: str):
    session_path = patient_service.find_session_path(session_id)
    if not session_path:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    metadata = patient_service.read_metadata(session_path / "metadata.json")
    sfd = metadata.get("screening_form_data")
    if sfd:
        return {"success": True, "patient_data": sfd}
    
    raise HTTPException(status_code=404, detail="No screening data found for this session.")

@router.get("/session_recommendations/{session_id}")
async def session_recommendations(session_id: str):
    session_path = patient_service.find_session_path(session_id)
    if not session_path:
        raise HTTPException(status_code=404, detail="Session not found.")

    metadata = patient_service.read_metadata(session_path / "metadata.json")
    sfd = metadata.get("screening_form_data")
    if not sfd:
        raise HTTPException(status_code=404, detail="No screening data available for this session.")

    result = prescreening_engine.get_recommendations(sfd)
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("message", "Recommendation pipeline failed."))
        
    return result

@router.post("/ai_summary", response_model=AISummaryResponse)
async def generate_ai_summary_endpoint(payload: AISummaryRequest = Body(...)):
    """
    Generates a professional 3-sentence clinical summary and patient-friendly next steps
    from PrescreeningEngine, ECGCNNEngine, and DeepSAEngine JSON outputs via Gemini 1.5 Pro.
    """
    prescreening_data = payload.prescreening_output or {}
    ecg_data = payload.ecg_output or {}
    deepsa_data = payload.deepsa_output or {}

    if payload.session_id:
        session_path = patient_service.find_session_path(payload.session_id)
        if session_path:
            metadata_file = session_path / "metadata.json"
            if metadata_file.exists():
                metadata = patient_service.read_metadata(metadata_file)
                if not prescreening_data and metadata.get("screening_form_data"):
                    try:
                        prescreening_data = prescreening_engine.predict(metadata["screening_form_data"])
                    except Exception as pe:
                        logging.warning(f"Session prescreening prediction failed: {pe}")
                        prescreening_data = metadata["screening_form_data"]

                if not ecg_data and metadata.get("ecg", {}).get("classification_result"):
                    ecg_raw = metadata["ecg"]["classification_result"]
                    if isinstance(ecg_raw, str):
                        try:
                            ecg_data = json.loads(ecg_raw)
                        except Exception:
                            ecg_data = {"raw": ecg_raw}
                    elif isinstance(ecg_raw, dict):
                        ecg_data = ecg_raw

                if not deepsa_data and metadata.get("angiogram"):
                    deepsa_data = metadata["angiogram"]

    result = gemini_service.generate_ai_summary(
        prescreening_output=prescreening_data,
        ecg_output=ecg_data,
        deepsa_output=deepsa_data,
    )

    return result

