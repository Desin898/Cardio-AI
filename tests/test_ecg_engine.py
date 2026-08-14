import io
import unittest
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import GroupKFold
from fastapi.testclient import TestClient

from backend.main import app
from backend.app.engines.ecg_engine import ECGCNNEngine
from backend.app.schemas.ecg import ECGInferenceResult


class TestECGEngine(unittest.TestCase):
    """
    Automated unit tests to verify GroupKFold validation stability,
    anatomical lead-to-artery mapping logic (LAD, LCx, RCA), Softmax category
    probability normalization, and POST /api/v1/ecg/predict standalone file upload API endpoints.
    """

    def setUp(self):
        self.client = TestClient(app)

    def test_group_kfold_no_patient_leakage(self):
        """
        Verifies that GroupKFold strictly isolates patient_ids such that no patient
        appears in both the training and validation splits for any fold.
        """
        num_samples = 100
        num_patients = 20

        groups = np.array([f"patient_{i % num_patients}" for i in range(num_samples)])
        X = np.random.randn(num_samples, 12, 100)
        y = np.random.randint(0, 4, size=num_samples)

        gkf = GroupKFold(n_splits=5)

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
            train_patients = set(groups[train_idx])
            val_patients = set(groups[val_idx])

            overlap = train_patients.intersection(val_patients)
            self.assertEqual(len(overlap), 0, f"Patient leakage detected in fold {fold + 1}: {overlap}")
            self.assertEqual(len(train_patients) + len(val_patients), num_patients)

    def test_lead_to_artery_mapping_lad(self):
        """
        Verifies that high deviation in Septal/Anterior leads (V1, V2, V3, V4)
        maps to Left Anterior Descending (LAD).
        """
        engine = ECGCNNEngine()
        signal = np.random.normal(loc=0.0, scale=0.1, size=(12, 100))
        signal[6:10, :] += np.random.normal(loc=5.0, scale=2.0, size=(4, 100))

        suspected_artery, affected_leads, breakdown = engine.calculate_lead_deviation_scores(signal)

        self.assertEqual(suspected_artery, "Left Anterior Descending (LAD)")
        for lead in ["V1", "V2", "V3", "V4"]:
            self.assertIn(lead, affected_leads)
            self.assertGreater(breakdown[lead], breakdown["I"])

    def test_lead_to_artery_mapping_lcx(self):
        """
        Verifies that high deviation in Lateral leads (I, aVL, V5, V6)
        maps to Left Circumflex (LCx).
        """
        engine = ECGCNNEngine()
        signal = np.random.normal(loc=0.0, scale=0.1, size=(12, 100))
        lateral_indices = [0, 4, 10, 11]
        signal[lateral_indices, :] += np.random.normal(loc=5.0, scale=2.0, size=(4, 100))

        suspected_artery, affected_leads, breakdown = engine.calculate_lead_deviation_scores(signal)

        self.assertEqual(suspected_artery, "Left Circumflex (LCx)")
        for lead in ["I", "aVL", "V5", "V6"]:
            self.assertIn(lead, affected_leads)

    def test_lead_to_artery_mapping_rca(self):
        """
        Verifies that high deviation in Inferior leads (II, III, aVF)
        maps to Right Coronary Artery (RCA).
        """
        engine = ECGCNNEngine()
        signal = np.random.normal(loc=0.0, scale=0.1, size=(12, 100))
        inferior_indices = [1, 2, 5]
        signal[inferior_indices, :] += np.random.normal(loc=5.0, scale=2.0, size=(3, 100))

        suspected_artery, affected_leads, breakdown = engine.calculate_lead_deviation_scores(signal)

        self.assertEqual(suspected_artery, "Right Coronary Artery (RCA)")
        for lead in ["II", "III", "aVF"]:
            self.assertIn(lead, affected_leads)

    def test_softmax_probability_distribution_normalization(self):
        """
        Verifies that category_probabilities returns real normalized Softmax probabilities summing to 1.0.
        """
        engine = ECGCNNEngine()
        signal_vec = np.random.randn(1200).tolist()
        res = engine.predict({"signal_vector": signal_vec})

        schema_instance = ECGInferenceResult(**res)
        probs = schema_instance.category_probabilities

        # Verify all 4 cardiac classes exist in category_probabilities
        for cls_name in ["Normal", "Abnormal Heartbeat", "Active Myocardial Infarction", "History of MI"]:
            self.assertIn(cls_name, probs)

        # Verify probability sum equals 1.0 (with 0.01 floating point tolerance)
        prob_sum = sum(probs.values())
        self.assertAlmostEqual(prob_sum, 1.0, delta=0.02, msg=f"Probabilities sum to {prob_sum}, expected 1.0")

        # Verify confidence_score matches max class probability
        max_prob_class = max(probs, key=probs.get)
        self.assertAlmostEqual(probs[max_prob_class], schema_instance.confidence_score, delta=0.02)

    def test_api_predict_standalone_csv_file_upload(self):
        """
        Verifies POST /api/v1/ecg/predict returns HTTP 200 OK and valid softmax probabilities.
        """
        columns = [f"Lead{lead}_{i}" for lead in range(1, 13) for i in range(100)]
        data_row = np.random.randn(1200).tolist()
        df = pd.DataFrame([data_row], columns=columns)

        csv_bytes = io.BytesIO()
        df.to_csv(csv_bytes, index=False)
        csv_bytes.seek(0)

        response = self.client.post(
            "/api/v1/ecg/predict",
            files={"ecg_file": ("test_ecg_signal.csv", csv_bytes, "text/csv")}
        )

        self.assertEqual(response.status_code, 200, f"Expected 200 OK, got {response.status_code}: {response.text}")
        payload = response.json()
        self.assertIn("predicted_class", payload)
        self.assertIn("confidence_score", payload)
        self.assertIn("category_probabilities", payload)

        probs = payload["category_probabilities"]
        self.assertGreater(len(probs), 0)
        prob_sum = sum(probs.values())
        self.assertAlmostEqual(prob_sum, 1.0, delta=0.02)

    def test_api_predict_standalone_image_file_upload(self):
        """
        Verifies POST /api/v1/ecg/predict returns HTTP 200 OK for standalone image uploads.
        """
        img_array = np.uint8(np.random.randint(200, 255, size=(300, 400, 3)))
        img = Image.fromarray(img_array)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        response = self.client.post(
            "/api/v1/ecg/predict",
            files={"ecg_file": ("test_ecg_chart.png", img_bytes, "image/png")}
        )

        self.assertEqual(response.status_code, 200, f"Expected 200 OK, got {response.status_code}: {response.text}")
        payload = response.json()
        self.assertIn("predicted_class", payload)
        self.assertIn("confidence_score", payload)
        self.assertIn("category_probabilities", payload)

        probs = payload["category_probabilities"]
        self.assertGreater(len(probs), 0)
        prob_sum = sum(probs.values())
        self.assertAlmostEqual(prob_sum, 1.0, delta=0.02)


if __name__ == "__main__":
    unittest.main()
