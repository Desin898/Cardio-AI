import os
import json
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Form, File, UploadFile, Query, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import settings
from backend.app.engines.prescreening_engine import prescreening_engine
from backend.app.engines.ten_year_risk_engine import ten_year_risk_engine
from backend.app.engines.ecg_cnn_engine import ecg_cnn_engine
from backend.app.engines.deepsa_engine import deepsa_engine

from backend.app.services.patient_service import patient_service
from backend.app.services.ecg_service import ecg_service
from backend.app.services.angiogram_service import angiogram_service

from backend.app.api.auth import router as auth_router
from backend.app.api.prescreening import router as prescreening_router
from backend.app.api.ecg import router as ecg_router
from backend.app.api.angiogram import router as angiogram_router
from backend.app.api.reports import router as reports_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
    logging.info("Initializing ML Engines on FastAPI startup...")
    try:
        prescreening_engine.load_models()
        ten_year_risk_engine.load_models()
    except Exception as e:
        logging.warning(f"Engine pre-loading warning: {e}")
    yield
    logging.info("FastAPI application shutdown.")

app = FastAPI(
    title=settings.APP_TITLE,
    version="2.0.0",
    description="Enterprise FastAPI backend for Coronary AI Detection System",
    lifespan=lifespan,
)

# CORS Middleware Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files & Setup Jinja2 Templates
static_dir = settings.PROJECT_ROOT / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

templates_dir = settings.PROJECT_ROOT / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Include API Routers under /api/v1
app.include_router(auth_router, prefix="/api/v1")
app.include_router(prescreening_router, prefix="/api/v1")
app.include_router(ecg_router, prefix="/api/v1")
app.include_router(angiogram_router, prefix="/api/v1")
app.include_router(reports_router, prefix="/api/v1")

#
# ALIAS ROUTES FOR BACKWARD COMPATIBILITY WITH EXISTING FRONTEND & FORM ACTIONS
#

@app.post("/predict")
def alias_predict(input_data: dict):
    from backend.app.schemas.patient import PatientScreeningInput
    validated = PatientScreeningInput(**input_data)
    from backend.app.api.prescreening import predict as run_predict
    return run_predict(validated)

@app.post("/predict_ten_year")
def alias_predict_ten_year(input_data: dict):
    from backend.app.schemas.patient import TenYearRiskInput
    validated = TenYearRiskInput(**input_data)
    from backend.app.api.prescreening import predict_ten_year as run_ten_year
    return run_ten_year(validated)

@app.post("/recommend")
def alias_recommend(input_data: dict):
    from backend.app.schemas.patient import PatientScreeningInput
    validated = PatientScreeningInput(**input_data)
    from backend.app.api.prescreening import recommend as run_recommend
    return run_recommend(validated)

@app.get("/get_screening_data/{session_id}")
async def alias_get_screening_data(session_id: str):
    from backend.app.api.reports import get_screening_data
    return await get_screening_data(session_id)

@app.get("/session_recommendations/{session_id}")
async def alias_session_recommendations(session_id: str):
    from backend.app.api.reports import session_recommendations
    return await session_recommendations(session_id)

@app.post("/ai_summary")
async def alias_ai_summary(input_data: dict):
    from backend.app.schemas.patient import AISummaryRequest
    validated = AISummaryRequest(**input_data)
    from backend.app.api.reports import generate_ai_summary_endpoint
    return await generate_ai_summary_endpoint(validated)


@app.post("/upload_ecg")
async def alias_upload_ecg(
    patient_id: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    notes: str = Form(""),
    ecg_type: str = Form("unknown"),
    screening_form_data: str = Form(""),
    ecg_file: UploadFile = File(...),
):
    from backend.app.api.ecg import upload_ecg
    return await upload_ecg(
        patient_id=patient_id,
        age=age,
        gender=gender,
        notes=notes,
        ecg_type=ecg_type,
        screening_form_data=screening_form_data,
        ecg_file=ecg_file,
    )

@app.get("/ecg_plot/{session_id}/{filename}")
async def alias_ecg_plot(session_id: str, filename: str):
    from backend.app.api.ecg import ecg_plot
    return await ecg_plot(session_id, filename)

@app.post("/upload_angiogram")
async def alias_upload_angiogram(
    session_id: str = Form(""),
    patient_id: str = Form(""),
    angio_type: str = Form("unknown"),
    doctor_notes: str = Form(""),
    angio_file: UploadFile = File(...),
):
    from backend.app.api.angiogram import upload_angiogram
    return await upload_angiogram(
        session_id=session_id,
        patient_id=patient_id,
        angio_type=angio_type,
        doctor_notes=doctor_notes,
        angio_file=angio_file,
    )

@app.get("/angiogram_image/{session_id}/{filename}")
async def alias_angiogram_image(session_id: str, filename: str):
    from backend.app.api.angiogram import angiogram_image
    return await angiogram_image(session_id, filename)

@app.post("/angiogram_confirm")
async def alias_angiogram_confirm(
    session_id: str = Form(""),
    patient_id: str = Form(""),
    selected_variant: str = Form(""),
):
    from backend.app.api.angiogram import angiogram_confirm
    return await angiogram_confirm(session_id, patient_id, selected_variant)

@app.get("/angiogram_frame/{patient_id}/{filename}")
async def alias_serve_preprocessed_frame(patient_id: str, filename: str):
    from backend.app.api.angiogram import serve_preprocessed_frame
    return await serve_preprocessed_frame(patient_id, filename)

#
# WEB TEMPLATE RENDERING ENDPOINTS
#

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/patient_login.html", response_class=HTMLResponse)
async def patient_login(request: Request):
    return templates.TemplateResponse(request=request, name="patient_login.html")

@app.get("/patient_register.html", response_class=HTMLResponse)
async def patient_register(request: Request):
    return templates.TemplateResponse(request=request, name="patient_register.html")

@app.get("/patient_dashboard.html", response_class=HTMLResponse)
@app.get("/patient_dashboard", response_class=HTMLResponse)
async def patient_dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="patient_dashboard.html")

@app.get("/upload_ecg.html", response_class=HTMLResponse)
@app.get("/upload_ecg", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse(request=request, name="upload_ecg.html")

@app.get("/pre_screening.html", response_class=HTMLResponse)
@app.get("/prescreening", response_class=HTMLResponse)
@app.get("/pre_screening", response_class=HTMLResponse)
async def pre_screening(request: Request):
    return templates.TemplateResponse(request=request, name="pre_screening.html")

@app.get("/pre_screening_results.html", response_class=HTMLResponse)
async def pre_screening_results(request: Request):
    return templates.TemplateResponse(request=request, name="pre_screening_results.html")

@app.get("/doctor_upload.html", response_class=HTMLResponse)
async def doctor_upload(request: Request):
    return templates.TemplateResponse(request=request, name="doctor_upload.html")

@app.get("/doctor_analysis_results.html", response_class=HTMLResponse)
async def doctor_analysis_results(request: Request):
    return templates.TemplateResponse(request=request, name="doctor_analysis_results.html")

@app.get("/doctor_login.html", response_class=HTMLResponse)
@app.get("/doctor_login", response_class=HTMLResponse)
async def doctor_login(request: Request, session_id: str = "", patient_id: str = ""):
    return templates.TemplateResponse(
        request=request,
        name="doctor_login.html",
        context={"session_id": session_id, "patient_id": patient_id, "error": None},
    )

@app.get("/doctor_dashboard", response_class=HTMLResponse)
@app.get("/doctor_dashboard.html", response_class=HTMLResponse)
async def doctor_dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="doctor_dashboard.html")

@app.get("/upload_angiogram", response_class=HTMLResponse)
@app.get("/angiogram_qca", response_class=HTMLResponse)
@app.get("/angiogram_qca.html", response_class=HTMLResponse)
@app.get("/angiogram_processing.html", response_class=HTMLResponse)
async def angiogram_upload_page(request: Request, session_id: str = "", patient_id: str = ""):
    return templates.TemplateResponse(
        request=request,
        name="angiogram_qca.html",
        context={"session_id": session_id, "patient_id": patient_id},
    )

@app.get("/angiogram_select/{session_id}", response_class=HTMLResponse)
@app.get("/angiogram_select.html", response_class=HTMLResponse)
async def angiogram_select(request: Request, session_id: str = ""):
    if not session_id:
        session_id = request.query_params.get("session_id", "")
    session_path = patient_service.find_session_path(session_id)
    if not session_path:
        raise HTTPException(status_code=404, detail="Session not found")

    metadata = patient_service.read_metadata(session_path / "metadata.json")
    patient_id = metadata.get("patient_id", "Unknown")
    variants = metadata.get("angiogram", {}).get("variants", [])

    for v in variants:
        v["url"] = f"/angiogram_image/{session_id}/{v['filename']}"

    return templates.TemplateResponse(
        request=request,
        name="angiogram_select.html",
        context={
            "session_id": session_id,
            "patient_id": patient_id,
            "variants": variants,
        },
    )

@app.get("/ecg_result/{session_id}", response_class=HTMLResponse)
@app.get("/ecg_result.html", response_class=HTMLResponse)
async def ecg_result(request: Request, session_id: str = ""):
    if not session_id:
        session_id = request.query_params.get("session_id", "")
    session_path = patient_service.find_session_path(session_id)
    if not session_path:
        raise HTTPException(status_code=404, detail="Session not found")

    metadata_file = session_path / "metadata.json"
    if not metadata_file.exists():
        raise HTTPException(status_code=404, detail="Metadata not found")

    metadata = patient_service.read_metadata(metadata_file)
    patient_id = metadata.get("patient_id", "Unknown")
    classification_raw = metadata.get("ecg", {}).get("classification_result")

    if not classification_raw:
        prediction = {"error": "No prediction data found"}
    else:
        try:
            prediction = (
                json.loads(classification_raw)
                if isinstance(classification_raw, str)
                else classification_raw
            )
        except Exception as e:
            prediction = {"error": f"JSON parse error: {str(e)}"}

    angio_info = metadata.get("angiogram", {})
    angiogram_uploaded = bool(angio_info.get("uploaded"))
    angio_selected = angio_info.get("selected_variant") if angio_info else None
    loc_result = angio_info.get("localization_result") if angio_info else None

    plot_url = None
    plot_path = settings.PROJECT_ROOT / "outputs" / "lead_activity_report.png"
    if plot_path.exists():
        plot_url = f"/ecg_plot/{session_id}/lead_activity_report.png"

    patient_data = metadata.get("screening_form_data")
    has_screening_data = bool(patient_data)

    ten_year_data = None
    if has_screening_data:
        ten_year_data = ten_year_risk_engine.predict(patient_data)

    return templates.TemplateResponse(
        request=request,
        name="ecg_result.html",
        context={
            "session_id": session_id,
            "patient_id": patient_id,
            "prediction": prediction,
            "plot_url": plot_url,
            "angiogram_uploaded": angiogram_uploaded,
            "angio_selected": angio_selected,
            "loc_result": loc_result,
            "patient_data": patient_data,
            "show_ten_year_risk": has_screening_data,
            "ten_year_model_available": (ten_year_risk_engine.ten_year_model is not None),
            "ten_year_data": ten_year_data,
            "has_screening_data": has_screening_data,
        },
    )

@app.get("/recommendations", response_class=HTMLResponse)
@app.get("/Recommendations.html", response_class=HTMLResponse)
async def recommendations_page(request: Request, session_id: str = "", source: str = "screening"):
    return templates.TemplateResponse(
        request=request,
        name="Recommendations.html",
        context={"session_id": session_id, "source": source},
    )

@app.get("/angiogram_processing", response_class=HTMLResponse)
async def angiogram_upload_form(request: Request):
    return templates.TemplateResponse(request=request, name="angiogram_processing.html")

@app.get("/angiogram_results", response_class=HTMLResponse)
async def angiogram_results(request: Request, patient_id: str = ""):
    if not patient_id:
        raise HTTPException(status_code=400, detail="Missing patient_id")

    patient_dir = settings.OUTPUT_ROOT / patient_id
    metadata_file = patient_dir / "metadata.json"
    if not metadata_file.exists():
        raise HTTPException(status_code=404, detail=f"Result folder not found for patient_id: {patient_id}")

    meta = patient_service.read_metadata(metadata_file)
    variants = meta.get("variants", [])
    selected_indices = meta.get("selected_frame_indices", [])

    for v in variants:
        v["url"] = f"/angiogram_frame/{patient_id}/{v['filename']}"

    return templates.TemplateResponse(
        request=request,
        name="angiogram_results1.html",
        context={
            "patient_id": patient_id,
            "variants": variants,
            "selected_indices": selected_indices,
        },
    )
