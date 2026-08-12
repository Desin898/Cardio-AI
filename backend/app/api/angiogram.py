import os
import json
import shutil
import logging
import urllib.parse
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import RedirectResponse, FileResponse
from werkzeug.utils import secure_filename

from backend.app.core.config import settings
from backend.app.services.patient_service import patient_service
from backend.app.services.angiogram_service import angiogram_service
from backend.app.engines.deepsa_engine import deepsa_engine

router = APIRouter(prefix="/angiogram", tags=["angiogram"])

@router.post("/upload_angiogram")
async def upload_angiogram(
    session_id: str = Form(""),
    patient_id: str = Form(""),
    angio_type: str = Form("unknown"),
    doctor_notes: str = Form(""),
    angio_file: UploadFile = File(...),
):
    try:
        session_id = session_id.strip()
        patient_id = patient_id.strip()

        if not angio_file or not angio_file.filename:
            raise HTTPException(status_code=400, detail="No angiogram file uploaded.")

        session_path = patient_service.find_session_path(session_id)
        if not session_path:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

        angio_folder = session_path / "angiogram"
        angio_folder.mkdir(parents=True, exist_ok=True)
        metadata_file = session_path / "metadata.json"

        raw_filename = secure_filename(angio_file.filename) or "angiogram"
        file_bytes = await angio_file.read()

        encrypted_path = patient_service.save_encrypted_angiogram(
            metadata_file, file_bytes, raw_filename
        )

        preprocessed_root = angio_folder / "preprocessed"
        pipeline_result = angiogram_service.process_raw_angiogram(
            file_bytes, raw_filename, preprocessed_root
        )

        output_directory = pipeline_result.get("output_directory", "")
        variants = pipeline_result.get("variants", [])

        enriched_variants = []
        if variants and output_directory:
            enriched_variants = angiogram_service.copy_frames_to_angio_folder(
                variants, output_directory, angio_folder, metadata_file
            )

        metadata = patient_service.read_metadata(metadata_file)
        metadata.setdefault("angiogram", {})
        metadata["angiogram"].update({
            "uploaded": True,
            "dicom_path": str(encrypted_path),
            "angio_type": angio_type,
            "doctor_notes": doctor_notes,
            "uploaded_by": "doctor",
            "preprocessed_folder": output_directory,
            "variants": enriched_variants or variants,
            "selected_variant": None,
            "selected_image_path": None,
            "localization_result": None,
        })
        patient_service.write_metadata(metadata_file, metadata)

        return RedirectResponse(
            url=f"/angiogram_select.html?session_id={session_id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )

    except Exception as e:
        logging.exception("Angiogram upload failed")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/angiogram_image/{session_id}/{filename}")
async def angiogram_image(session_id: str, filename: str):
    session_path = patient_service.find_session_path(session_id)
    if not session_path:
        raise HTTPException(status_code=404, detail="Session not found")

    direct_path = session_path / "angiogram" / filename
    if direct_path.exists():
        return FileResponse(str(direct_path), media_type="image/png")

    try:
        metadata = patient_service.read_metadata(session_path / "metadata.json")
        preprocessed_folder = metadata.get("angiogram", {}).get("preprocessed_folder")
        if preprocessed_folder:
            pp_path = Path(preprocessed_folder) / filename
            if pp_path.exists():
                return FileResponse(str(pp_path), media_type="image/png")
    except Exception:
        pass

    for img_path in (session_path / "angiogram").rglob(filename):
        return FileResponse(str(img_path), media_type="image/png")

    raise HTTPException(status_code=404, detail="Image not found")

@router.post("/angiogram_confirm")
async def angiogram_confirm(
    session_id: str = Form(""),
    patient_id: str = Form(""),
    selected_variant: str = Form(""),
):
    try:
        session_id = session_id.strip()
        patient_id_form = patient_id.strip()
        selected_filename = selected_variant.strip()

        if not selected_filename:
            raise HTTPException(status_code=400, detail="Missing selected_variant — no frame selected.")

        use_standalone = bool(patient_id_form) and not session_id

        if session_id:
            session_path = patient_service.find_session_path(session_id)
            if not session_path:
                raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

            metadata_file = session_path / "metadata.json"
            metadata = patient_service.read_metadata(metadata_file)
            angio = metadata.get("angiogram", {})
            dicom_path = angio.get("dicom_path", "")
            preprocessed_folder = angio.get("preprocessed_folder", "")
            existing_mask = angio.get("segmentation", {}).get("mask_path", "")
            angio_folder = session_path / "angiogram"

            source_frame_path = angio_folder / selected_filename
            if not source_frame_path.exists() and preprocessed_folder:
                pp_candidate = Path(preprocessed_folder) / selected_filename
                if pp_candidate.exists():
                    source_frame_path = pp_candidate
            if not source_frame_path.exists():
                found = next(angio_folder.rglob(selected_filename), None)
                if found:
                    source_frame_path = found

            if not source_frame_path.exists():
                raise HTTPException(
                    status_code=500, detail=f"Selected frame '{selected_filename}' could not be located."
                )

            saved_frame_path = angio_folder / f"selected_{selected_filename}"
            shutil.copy2(source_frame_path, saved_frame_path)

            pipeline_meta = angiogram_service.read_pipeline_metadata(preprocessed_folder)
            variant_info = next(
                (v for v in pipeline_meta.get("variants", []) if v.get("filename") == selected_filename),
                {}
            )
            frame_details = {
                "label": variant_info.get("label", selected_filename),
                "filename": selected_filename,
                "saved_frame_path": str(saved_frame_path),
                "source_path": str(source_frame_path),
                "selected_frame_indices": pipeline_meta.get("selected_frame_indices", []),
                "source": pipeline_meta.get("source", "Unknown"),
                "number_of_original_frames": pipeline_meta.get("number_of_original_frames", 0),
                "pipeline_patient_id": pipeline_meta.get("patient_id", ""),
            }
            all_variants = pipeline_meta.get("variants", angio.get("variants", []))

            angiogram_service.write_frame_metadata(angio_folder, {
                "selected_filename": selected_filename,
                "saved_frame_path": str(saved_frame_path),
                "source_preprocessed_path": str(source_frame_path),
                "preprocessed_folder": preprocessed_folder,
                "frame_details": frame_details,
                "all_pipeline_variants": all_variants,
                "saved_at": datetime.utcnow().isoformat(),
            })

            angio.update({
                "dicom_path": dicom_path,
                "selected_frames": [str(saved_frame_path)],
                "selected_variant": selected_filename,
                "segmentation": {
                    "mask_path": existing_mask,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            })
            metadata["angiogram"] = angio
            patient_service.write_metadata(metadata_file, metadata)

        if use_standalone:
            meta_path = settings.OUTPUT_ROOT / patient_id_form / "metadata.json"
            if meta_path.exists():
                try:
                    meta = patient_service.read_metadata(meta_path)
                    meta["selected_variant"] = selected_filename
                    patient_service.write_metadata(meta_path, meta)
                except Exception:
                    pass

        deepsa_engine.ensure_running()

        if use_standalone:
            flask_image_url = f"http://127.0.0.1:5000/angiogram_frame/{patient_id_form}/{selected_filename}"
        else:
            flask_image_url = f"http://127.0.0.1:5000/angiogram_image/{session_id}/{selected_filename}"

        encoded_url = urllib.parse.quote(flask_image_url, safe="")
        deepsa_redirect = f"{settings.DEEPSA_URL}/?image={encoded_url}"
        return RedirectResponse(url=deepsa_redirect, status_code=status.HTTP_303_SEE_OTHER)

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Angiogram confirmation failed")
        raise HTTPException(status_code=500, detail=f"Confirmation failed: {str(e)}")

@router.get("/angiogram_frame/{patient_id}/{filename}")
async def serve_preprocessed_frame(patient_id: str, filename: str):
    img_path = settings.OUTPUT_ROOT / patient_id / filename
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Frame not found")
    return FileResponse(str(img_path), media_type="image/png")
