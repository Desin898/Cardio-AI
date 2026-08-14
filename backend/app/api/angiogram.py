import os
import json
import shutil
import logging
import urllib.parse
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Body
from fastapi.responses import RedirectResponse, FileResponse, JSONResponse
from werkzeug.utils import secure_filename

from backend.app.core.config import settings
from backend.app.services.patient_service import patient_service
from backend.app.services.angiogram_service import angiogram_service
from backend.app.engines.deepsa_engine import deepsa_engine
from backend.app.schemas.angiogram import (
    QCAMetricsResponse,
    QCAAnalysisResult,
    AngiogramConfirmRequest,
    AngiogramUploadResponse,
)

router = APIRouter(prefix="/angiogram", tags=["angiogram"])


@router.post("/process_video", response_model=QCAAnalysisResult)
async def process_video(
    session_id: Optional[str] = Form(""),
    patient_id: Optional[str] = Form(""),
    angio_file: UploadFile = File(...),
):
    """
    Accept DICOM / MP4 / Image angiogram uploads, perform keyframe extraction,
    execute DeepSA segmentation + Quantitative Coronary Angiography (QCA) profiling,
    and return structured QCA findings and visualization URLs.
    """
    try:
        if not angio_file or not angio_file.filename:
            raise HTTPException(status_code=400, detail="No angiogram file uploaded.")

        raw_filename = secure_filename(angio_file.filename) or "angiogram_file"
        file_bytes = await angio_file.read()

        temp_dir = settings.PROJECT_ROOT / "outputs" / "temp_angio"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_filepath = temp_dir / raw_filename
        with open(temp_filepath, "wb") as f:
            f.write(file_bytes)

        # 1. Run keyframe extraction & preprocessing
        pipeline_res = angiogram_service.process_raw_angiogram(
            file_bytes, raw_filename, temp_dir
        )

        variants = pipeline_res.get("variants", [])
        output_dir = pipeline_res.get("output_directory", str(temp_dir))

        # Select target keyframe (first variant or raw file)
        target_frame_path = None
        if variants:
            first_var_fn = variants[0].get("filename")
            target_frame_path = Path(output_dir) / first_var_fn
        if not target_frame_path or not target_frame_path.exists():
            target_frame_path = temp_filepath

        # 2. Execute DeepSA + QCA Engine profiling
        qca_res = deepsa_engine.analyze_qca(
            image_input=target_frame_path,
            save_dir=settings.PROJECT_ROOT / "outputs" / "qca_reports"
        )

        qca_vis_path = Path(qca_res["qca_image_path"])
        qca_img_filename = qca_vis_path.name

        qca_metrics_data = qca_res["qca_metrics"]
        qca_metrics = QCAMetricsResponse(
            stenosis_percentage=qca_metrics_data["stenosis_percentage"],
            severity_grade=qca_metrics_data["severity_grade"],
            d_min=qca_metrics_data["d_min"],
            d_ref=qca_metrics_data["d_ref"],
            lesion_coordinates=qca_metrics_data["lesion_coordinates"],
            intervention_recommended=qca_metrics_data["intervention_recommended"],
        )

        return QCAAnalysisResult(
            success=True,
            session_id=session_id or "",
            patient_id=patient_id or "",
            qca_metrics=qca_metrics,
            qca_image_url=f"/api/v1/angiogram/qca_image/{qca_img_filename}",
            keyframe_url=f"/api/v1/angiogram/qca_image/{qca_img_filename}",
            error=None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Angiogram QCA video processing failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"QCA video processing failed: {str(e)}"
        )


@router.post("/analyze_frame", response_model=QCAAnalysisResult)
async def analyze_frame(
    session_id: Optional[str] = Form(""),
    patient_id: Optional[str] = Form(""),
    frame_file: Optional[UploadFile] = File(None),
    frame_path: Optional[str] = Form(None),
):
    """
    Performs Quantitative Coronary Angiography (QCA) profiling on a single frame or image file.
    """
    try:
        target_img_path = None
        temp_dir = settings.PROJECT_ROOT / "outputs" / "temp_angio"
        temp_dir.mkdir(parents=True, exist_ok=True)

        if frame_file is not None and frame_file.filename:
            file_bytes = await frame_file.read()
            raw_fn = secure_filename(frame_file.filename) or "frame.png"
            target_img_path = temp_dir / raw_fn
            with open(target_img_path, "wb") as f:
                f.write(file_bytes)
        elif frame_path and os.path.exists(frame_path):
            target_img_path = Path(frame_path)

        if not target_img_path or not target_img_path.exists():
            raise HTTPException(status_code=400, detail="No valid frame file or frame_path provided.")

        qca_res = deepsa_engine.analyze_qca(
            image_input=target_img_path,
            save_dir=settings.PROJECT_ROOT / "outputs" / "qca_reports"
        )

        qca_vis_path = Path(qca_res["qca_image_path"])
        qca_img_filename = qca_vis_path.name

        qca_metrics_data = qca_res["qca_metrics"]
        qca_metrics = QCAMetricsResponse(
            stenosis_percentage=qca_metrics_data["stenosis_percentage"],
            severity_grade=qca_metrics_data["severity_grade"],
            d_min=qca_metrics_data["d_min"],
            d_ref=qca_metrics_data["d_ref"],
            lesion_coordinates=qca_metrics_data["lesion_coordinates"],
            intervention_recommended=qca_metrics_data["intervention_recommended"],
        )

        return QCAAnalysisResult(
            success=True,
            session_id=session_id or "",
            patient_id=patient_id or "",
            qca_metrics=qca_metrics,
            qca_image_url=f"/api/v1/angiogram/qca_image/{qca_img_filename}",
            keyframe_url=f"/api/v1/angiogram/qca_image/{qca_img_filename}",
            error=None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Angiogram QCA frame analysis failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"QCA frame analysis failed: {str(e)}"
        )


@router.get("/qca_image/{filename}")
async def get_qca_image(filename: str):
    """Serves annotated QCA visual diagnostic report images."""
    qca_path = settings.PROJECT_ROOT / "outputs" / "qca_reports" / filename
    if qca_path.exists():
        return FileResponse(str(qca_path), media_type="image/png")
    raise HTTPException(status_code=404, detail="QCA visualization image not found")


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

            # Run QCA profiling on selected frame
            qca_res = deepsa_engine.analyze_qca(
                image_input=saved_frame_path,
                save_dir=angio_folder
            )

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
                "qca_metrics": qca_res.get("qca_metrics"),
            })

            angio.update({
                "dicom_path": dicom_path,
                "selected_frames": [str(saved_frame_path)],
                "selected_variant": selected_filename,
                "localization_result": qca_res.get("qca_metrics"),
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
