from fastapi import APIRouter, Form, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from backend.app.schemas.auth import DoctorLoginRequest

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/doctor_login")
async def doctor_login_post(
    username: str = Form(""),
    password: str = Form(""),
    session_id: str = Form(""),
    patient_id: str = Form(""),
):
    if session_id:
        return RedirectResponse(
            url=f"/upload_angiogram?session_id={session_id}&patient_id={patient_id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )
    return RedirectResponse(
        url=f"/doctor_upload.html?doctor={username}",
        status_code=status.HTTP_303_SEE_OTHER,
    )
