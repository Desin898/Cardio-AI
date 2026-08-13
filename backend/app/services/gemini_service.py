import os
import json
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from google import genai
from google.genai import types

from backend.app.core.config import settings

class AISummarySchema(BaseModel):
    clinical_summary: str = Field(
        description="A professional clinical summary of findings from the 3 engines, formatted as exactly 3 sentences."
    )
    patient_next_steps: str = Field(
        description="Patient-friendly next steps and lifestyle/clinical recommendations written in accessible, encouraging language."
    )

class GeminiService:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
        self.model = model or settings.GEMINI_MODEL or "gemini-1.5-pro"

    def _get_client(self) -> Optional[genai.Client]:
        key = self.api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            return None
        try:
            return genai.Client(api_key=key)
        except Exception as e:
            logging.error(f"Failed to initialize GenAI client: {e}")
            return None

    def generate_ai_summary(
        self,
        prescreening_output: Optional[Dict[str, Any]] = None,
        ecg_output: Optional[Dict[str, Any]] = None,
        deepsa_output: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Takes JSON outputs from PrescreeningEngine, ECGCNNEngine, and DeepSAEngine,
        and uses Gemini 1.5 Pro to generate a professional 3-sentence clinical summary
        and patient-friendly next steps.
        """
        prescreening_output = prescreening_output or {}
        ecg_output = ecg_output or {}
        deepsa_output = deepsa_output or {}

        client = self._get_client()
        if not client:
            logging.warning("GEMINI_API_KEY is not set or client initialization failed. Returning fallback summary.")
            return self._build_fallback_summary(
                prescreening_output, ecg_output, deepsa_output,
                error="GEMINI_API_KEY environment variable is not configured."
            )

        prompt = f"""
You are an expert cardiologist AI assistant. Analyze the following JSON diagnostic outputs from our three AI engines:

=== 1. PRESCREENING ENGINE OUTPUT ===
{json.dumps(prescreening_output, indent=2)}

=== 2. ECG CNN ENGINE OUTPUT ===
{json.dumps(ecg_output, indent=2)}

=== 3. DEEPSA ANGIOGRAM ENGINE OUTPUT ===
{json.dumps(deepsa_output, indent=2)}

REQUIREMENTS:
1. "clinical_summary": Must be EXACTLY 3 professional, medically precise sentences summarizing the patient's cardiovascular risk profile, ECG waveform findings, and angiogram/stenosis/vessel findings based on the provided engine outputs.
2. "patient_next_steps": Provide clear, patient-friendly next steps and lifestyle/action recommendations in simple, reassuring language that the patient can follow.
"""

        try:
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=AISummarySchema,
                    temperature=0.2,
                ),
            )

            if response and response.text:
                try:
                    data = json.loads(response.text)
                    return {
                        "success": True,
                        "clinical_summary": data.get("clinical_summary", "").strip(),
                        "patient_next_steps": data.get("patient_next_steps", "").strip(),
                        "model_used": self.model,
                        "error": None,
                    }
                except Exception as parse_err:
                    logging.warning(f"Error parsing Gemini response JSON: {parse_err}. Raw text: {response.text}")
                    return {
                        "success": True,
                        "clinical_summary": response.text.strip(),
                        "patient_next_steps": "Please discuss these detailed results with your attending physician.",
                        "model_used": self.model,
                        "error": None,
                    }

            return self._build_fallback_summary(
                prescreening_output, ecg_output, deepsa_output,
                error="Empty response received from Gemini API."
            )

        except Exception as e:
            logging.exception(f"Gemini API generation error: {e}")
            return self._build_fallback_summary(
                prescreening_output, ecg_output, deepsa_output,
                error=f"Gemini API error: {str(e)}"
            )

    def _build_fallback_summary(
        self,
        prescreening: Dict[str, Any],
        ecg: Dict[str, Any],
        deepsa: Dict[str, Any],
        error: str,
    ) -> Dict[str, Any]:
        risk = prescreening.get("prediction", prescreening.get("risk_status", "Undetermined"))
        ecg_pred = ecg.get("prediction", "Not available")
        vessel = ecg.get("suspected_vessel", deepsa.get("selected_variant", "Not localized"))

        sentence1 = f"Initial cardiovascular prescreening indicates a status of {risk}."
        sentence2 = f"ECG CNN analysis reveals findings consistent with {ecg_pred}."
        sentence3 = f"Angiogram assessment highlights potential vessel involvement in {vessel}."

        fallback_clinical = f"{sentence1} {sentence2} {sentence3}"
        fallback_patient = (
            "1. Schedule an appointment with your cardiologist to review these diagnostic findings.\n"
            "2. Maintain regular blood pressure and pulse monitoring.\n"
            "3. Follow a heart-healthy diet and adhere to recommended physical activity guidance."
        )

        return {
            "success": False,
            "clinical_summary": fallback_clinical,
            "patient_next_steps": fallback_patient,
            "model_used": self.model,
            "error": error,
        }

gemini_service = GeminiService()
