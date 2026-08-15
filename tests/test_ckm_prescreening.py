import unittest
import warnings
from starlette.testclient import TestClient

from backend.main import app
from backend.app.schemas.patient import PatientScreeningInput, PrescreeningResponse
from backend.app.engines.prescreening_engine import prescreening_engine

class TestCKMPrescreeningEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Force load models on engine startup
        prescreening_engine.load_models()
        from backend.app.engines.ten_year_risk_engine import ten_year_risk_engine
        ten_year_risk_engine.load_models()
        cls.client = TestClient(app)

    def test_01_schema_validation_full_ckm_biomarkers(self):
        input_data = {
            "age": 55,
            "gender": "Male",
            "systolic_bp": 145.0,
            "diastolic_bp": 92.0,
            "bmi": 28.5,
            "current_smoker": True,
            "hba1c": 7.8,
            "hs_troponin": 18.5,
            "egfr": 70.0,
            "cholesterol_total": 235.0,
            "cholesterol_hdl": 42.0,
            "family_history_cad": True
        }
        validated = PatientScreeningInput(**input_data)
        self.assertEqual(validated.age, 55)
        self.assertEqual(validated.gender, "Male")
        self.assertTrue(validated.current_smoker)
        self.assertEqual(validated.hba1c, 7.8)
        self.assertTrue(validated.family_history_cad)

    def test_02_schema_validation_optional_missing_biomarkers(self):
        input_data = {
            "age": 42,
            "gender": "Female",
            "systolic_bp": 118.0,
            "diastolic_bp": 76.0,
            "bmi": 22.4,
            "current_smoker": False,
            "cholesterol_total": 180.0,
            "cholesterol_hdl": 58.0,
            "family_history_cad": False
        }
        validated = PatientScreeningInput(**input_data)
        self.assertIsNone(validated.hba1c)
        self.assertIsNone(validated.hs_troponin)
        self.assertIsNone(validated.egfr)

    def test_03_engine_missing_value_handling_and_calibration(self):
        # Data with missing biomarker inputs (None -> np.nan)
        data = {
            "age": 62,
            "gender": "Male",
            "systolic_bp": 155.0,
            "diastolic_bp": 96.0,
            "bmi": 31.0,
            "current_smoker": True,
            "hba1c": None,
            "hs_troponin": None,
            "egfr": None,
            "cholesterol_total": 250.0,
            "cholesterol_hdl": 38.0,
            "family_history_cad": True
        }
        res = prescreening_engine.predict(data)
        self.assertTrue(res["success"])
        self.assertIn("risk_category", res)
        self.assertIn(res["risk_category"], ["LOW", "MODERATE", "HIGH"])
        self.assertGreaterEqual(res["risk_probability"], 0.0)
        self.assertLessEqual(res["risk_probability"], 1.0)

    def test_04_risk_threshold_categorization(self):
        # Low risk input
        low_data = {
            "age": 28,
            "gender": "Female",
            "systolic_bp": 110.0,
            "diastolic_bp": 70.0,
            "bmi": 20.5,
            "current_smoker": False,
            "hba1c": 5.2,
            "hs_troponin": 5.0,
            "egfr": 105.0,
            "cholesterol_total": 160.0,
            "cholesterol_hdl": 65.0,
            "family_history_cad": False
        }
        low_res = prescreening_engine.predict(low_data)
        self.assertLess(low_res["risk_probability"], 0.65)
        self.assertIn(low_res["risk_category"], ["LOW", "MODERATE"])
        self.assertIn(low_res["recommended_next_path"], ["ROUTINE_MONITORING_LIFESTYLE", "CARDIAC_STRESS_TEST_ECHO"])

    def test_05_shap_explanation_json_structure(self):
        data = {
            "age": 60,
            "gender": "Male",
            "systolic_bp": 160.0,
            "diastolic_bp": 98.0,
            "bmi": 32.5,
            "current_smoker": True,
            "hba1c": 8.5,
            "hs_troponin": 24.0,
            "egfr": 60.0,
            "cholesterol_total": 260.0,
            "cholesterol_hdl": 35.0,
            "family_history_cad": True
        }
        res = prescreening_engine.predict(data)
        shap_list = res["shap_breakdown"]
        self.assertIsInstance(shap_list, list)
        self.assertGreater(len(shap_list), 0)

        top_driver = shap_list[0]
        self.assertIn("feature", top_driver)
        self.assertIn("value", top_driver)
        self.assertIn("impact", top_driver)
        self.assertIn("direction", top_driver)
        self.assertIn(top_driver["direction"], ["risk_increasing", "risk_decreasing"])
        self.assertTrue("%" in top_driver["impact"])

    def test_06_api_endpoint_predict_response(self):
        payload = {
            "age": 58,
            "gender": "Male",
            "systolic_bp": 142.0,
            "diastolic_bp": 88.0,
            "bmi": 27.8,
            "current_smoker": True,
            "hba1c": 7.2,
            "hs_troponin": 15.0,
            "egfr": 78.0,
            "cholesterol_total": 220.0,
            "cholesterol_hdl": 45.0,
            "family_history_cad": True
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            response = self.client.post("/api/v1/prescreening/predict", json=payload)
            self.assertEqual(response.status_code, 200)
            data = response.json()

            self.assertTrue(data["success"])
            self.assertIn("risk_category", data)
            self.assertIn("risk_probability", data)
            self.assertIn("probability_percentage", data)
            self.assertIn("shap_breakdown", data)
            self.assertIn("recommended_next_path", data)

            # Check zero unpickling warnings emitted during invocation
            unpickle_warnings = [
                item for item in w
                if "unpickle" in str(item.message).lower() 
                or "inconsistentversion" in str(item.message).lower()
                or "unpickling" in str(item.category).lower()
            ]
            if unpickle_warnings:
                print("Caught unpickle warnings:", [str(item.message) for item in unpickle_warnings])
            self.assertEqual(len(unpickle_warnings), 0)

    def test_07_web_workspace_endpoints_200_ok(self):
        for route in ["/pre_screening.html", "/pre_screening_results.html", "/upload_ecg.html", "/angiogram_qca.html", "/angiogram_qca"]:
            response = self.client.get(route)
            self.assertEqual(response.status_code, 200, f"Route {route} failed with status {response.status_code}")

if __name__ == "__main__":
    unittest.main()
