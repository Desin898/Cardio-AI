from fastapi import APIRouter, HTTPException, status
from backend.app.services.patient_service import patient_service
from backend.app.engines.prescreening_engine import prescreening_engine

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
