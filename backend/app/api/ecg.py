import os
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import RedirectResponse, FileResponse

from backend.app.core.config import settings
from backend.app.services.patient_service import patient_service
from backend.app.services.ecg_service import ecg_service
from backend.app.engines.ecg_cnn_engine import ecg_cnn_engine

router = APIRouter(prefix="/ecg", tags=["ecg"])

@router.post("/upload_ecg")
async def upload_ecg(
    patient_id: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    notes: Optional[str] = Form(""),
    ecg_type: Optional[str] = Form("unknown"),
    screening_form_data: Optional[str] = Form(""),
    ecg_file: UploadFile = File(...),
):
    try:
        if not ecg_file:
            raise HTTPException(status_code=400, detail="No file uploaded")

        filename = ecg_file.filename
        file_bytes = await ecg_file.read()

        patient_data = {
            "patient_id": patient_id,
            "age": age,
            "gender": gender,
            "notes": notes,
            "ecg_type": ecg_type,
        }

        metadata_path = patient_service.initialize_session(patient_data)
        session_folder = metadata_path.parent
        ecg_folder = session_folder / "ecg"
        ecg_folder.mkdir(parents=True, exist_ok=True)

        encrypted_path = patient_service.save_encrypted_ecg(metadata_path, file_bytes, filename)
        csv_path = ecg_service.preprocess_ecg_file(file_bytes, filename, ecg_folder)
        prediction_result = ecg_cnn_engine.predict({"csv_path": str(csv_path)})

        patient_service.update_ecg_metadata(
            metadata_path,
            raw_image_path=str(encrypted_path),
            processed_csv_path=str(csv_path),
            classification_result=json.dumps(prediction_result),
        )

        metadata = patient_service.read_metadata(metadata_path)
        metadata["ecg"]["classification_result"] = json.dumps(prediction_result)

        if screening_form_data and screening_form_data.strip():
            try:
                sfd = json.loads(screening_form_data.strip())
                sfd.setdefault("heart_rate", 72)
                sfd.setdefault("cigs_per_day", 0)
                sfd.setdefault("bp_treatment", "No")
                sfd.setdefault("previous_stroke", "No")
                sfd.setdefault("diabetes", "No")
                sfd.setdefault(
                    "prevalent_hyp",
                    "Yes" if (float(sfd.get("systolic_bp", 0)) >= 140 or float(sfd.get("diastolic_bp", 0)) >= 90) else "No"
                )
                metadata["screening_form_data"] = sfd
            except Exception as je:
                logging.warning(f"screening_form_data parsing warning: {je}")

        patient_service.write_metadata(metadata_path, metadata)

        confidence_str = prediction_result.get("confidence", "0%")
        try:
            risk_pct = float(str(confidence_str).replace("%", "").strip())
        except (ValueError, AttributeError):
            risk_pct = 0.0

        patient_service.update_risk_prediction(
            metadata_path,
            prediction_result.get("risk_level", "UNKNOWN"),
            risk_pct,
        )

        return RedirectResponse(
            url=f"/ecg_result.html?session_id={session_folder.name}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    except Exception as e:
        logging.exception("Upload + prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}",
        )

@router.get("/ecg_plot/{session_id}/{filename}")
async def ecg_plot(session_id: str, filename: str):
    plot_path = settings.PROJECT_ROOT / "outputs" / filename
    if plot_path.exists():
        return FileResponse(str(plot_path), media_type="image/png")

    session_path = patient_service.find_session_path(session_id)
    if not session_path:
        raise HTTPException(status_code=404, detail="Session not found")

    fallback_path = session_path / "ecg" / filename
    if not fallback_path.exists():
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(str(fallback_path), media_type="image/png")
