import os
import json
import logging
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from backend.app.services.gemini_service import gemini_service, GeminiService
from backend.main import app

def test_gemini_service_direct():
    print("\n--- Testing GeminiService Direct Call (Fallback) ---")
    prescreening_sample = {
        "prediction": "High Risk",
        "probability": 0.82,
        "top_factors": [
            {"factor": "sysBP", "value": 155, "effect": "increases risk"},
            {"factor": "totChol", "value": 240, "effect": "increases risk"}
        ]
    }
    ecg_sample = {
        "prediction": "Myocardial Infarction",
        "confidence": "88.5%",
        "risk_level": "EMERGENCY",
        "suspected_vessel": "LAD"
    }
    deepsa_sample = {
        "status": "completed",
        "selected_variant": "variant_01.png",
        "stenosis_severity": "Severe (75-90%)",
        "localized_vessel": "Left Anterior Descending (LAD)"
    }

    result = gemini_service.generate_ai_summary(
        prescreening_output=prescreening_sample,
        ecg_output=ecg_sample,
        deepsa_output=deepsa_sample,
    )

    print("Fallback Result:")
    print(json.dumps(result, indent=2))
    assert "clinical_summary" in result
    assert "patient_next_steps" in result
    print("Direct GeminiService fallback test passed!")

def test_gemini_service_mock_genai():
    print("\n--- Testing GeminiService with Gemini 1.5 Pro Mock Client ---")
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "clinical_summary": "Patient demonstrates high baseline cardiovascular risk driven by elevated systolic blood pressure and cholesterol. ECG CNN analysis indicates acute myocardial infarction with significant ST-segment changes. DeepSA angiogram assessment confirms severe stenosis localized to the left anterior descending artery.",
        "patient_next_steps": "1. Seek immediate emergency cardiovascular care for definitive intervention.\n2. Strictly adhere to emergency protocol medications as directed by your physician.\n3. Rest and avoid any strenuous physical exertion."
    })
    mock_client.models.generate_content.return_value = mock_response

    service = GeminiService(api_key="test_dummy_key", model="gemini-1.5-pro")

    with patch.object(service, "_get_client", return_value=mock_client):
        res = service.generate_ai_summary(
            prescreening_output={"prediction": "High Risk"},
            ecg_output={"prediction": "Myocardial Infarction"},
            deepsa_output={"stenosis": "Severe LAD"}
        )

        print("Mocked GenAI Result:")
        print(json.dumps(res, indent=2))
        assert res["success"] is True
        assert res["model_used"] == "gemini-1.5-pro"
        # Check exactly 3 sentences in clinical summary
        sentences = [s for s in res["clinical_summary"].split(".") if s.strip()]
        assert len(sentences) == 3
        print("Mocked GenAI Gemini 1.5 Pro test passed!")

def test_ai_summary_endpoint():
    print("\n--- Testing FastAPI /api/v1/reports/ai_summary Endpoint ---")
    client = TestClient(app)

    payload = {
        "prescreening_output": {
            "prediction": "Low Risk",
            "probability": 0.12
        },
        "ecg_output": {
            "prediction": "Normal Sinus Rhythm",
            "confidence": "96.0%",
            "risk_level": "LOW",
            "suspected_vessel": "N/A"
        },
        "deepsa_output": {
            "status": "normal",
            "notes": "No significant stenosis detected"
        }
    }

    response = client.post("/api/v1/reports/ai_summary", json=payload)
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))

    assert response.status_code == 200
    res_data = response.json()
    assert "clinical_summary" in res_data
    assert "patient_next_steps" in res_data
    print("Endpoint test passed!")

if __name__ == "__main__":
    test_gemini_service_direct()
    test_gemini_service_mock_genai()
    test_ai_summary_endpoint()
