import unittest
import numpy as np
from sklearn.model_selection import GroupKFold

from backend.app.engines.ecg_engine import ECGCNNEngine
from backend.app.schemas.ecg import ECGInferenceResult


class TestECGEngine(unittest.TestCase):
    """
    Automated unit tests to verify GroupKFold validation stability
    and anatomical lead-to-artery mapping logic (LAD, LCx, RCA).
    """

    def test_group_kfold_no_patient_leakage(self):
        """
        Verifies that GroupKFold strictly isolates patient_ids such that no patient
        appears in both the training and validation splits for any fold.
        """
        num_samples = 100
        num_patients = 20

        # 100 samples assigned across 20 distinct patient IDs (5 samples per patient)
        groups = np.array([f"patient_{i % num_patients}" for i in range(num_samples)])
        X = np.random.randn(num_samples, 12, 100)
        y = np.random.randint(0, 4, size=num_samples)

        gkf = GroupKFold(n_splits=5)

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
            train_patients = set(groups[train_idx])
            val_patients = set(groups[val_idx])

            # Assert no overlapping patient IDs between train and validation sets
            overlap = train_patients.intersection(val_patients)
            self.assertEqual(len(overlap), 0, f"Patient leakage detected in fold {fold + 1}: {overlap}")

            # Assert total patient coverage
            self.assertEqual(len(train_patients) + len(val_patients), num_patients)

    def test_lead_to_artery_mapping_lad(self):
        """
        Verifies that high deviation in Septal/Anterior leads (V1, V2, V3, V4)
        maps to Left Anterior Descending (LAD).
        """
        engine = ECGCNNEngine()

        # Base signal matrix (12 channels x 100 samples) with low noise
        signal = np.random.normal(loc=0.0, scale=0.1, size=(12, 100))

        # Inject high variance/ST elevation into Septal/Anterior leads V1, V2, V3, V4 (Indices 6, 7, 8, 9)
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

        # Inject high variance into Lateral leads: I (0), aVL (4), V5 (10), V6 (11)
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

        # Inject high variance into Inferior leads: II (1), III (2), aVF (5)
        inferior_indices = [1, 2, 5]
        signal[inferior_indices, :] += np.random.normal(loc=5.0, scale=2.0, size=(3, 100))

        suspected_artery, affected_leads, breakdown = engine.calculate_lead_deviation_scores(signal)

        self.assertEqual(suspected_artery, "Right Coronary Artery (RCA)")
        for lead in ["II", "III", "aVF"]:
            self.assertIn(lead, affected_leads)

    def test_ecg_engine_predict_payload_structure(self):
        """
        Verifies that ecg_engine.predict returns a compliant payload structure matching ECGInferenceResult schema.
        """
        engine = ECGCNNEngine()

        # Generate synthetic 12-lead signal vector (12 x 100 = 1200 values)
        signal_vec = np.random.randn(1200).tolist()
        res = engine.predict({"signal_vector": signal_vec})

        # Validate against Pydantic Schema
        schema_instance = ECGInferenceResult(**res)

        self.assertIn(schema_instance.predicted_class, ["Normal", "Abnormal Heartbeat", "Active Myocardial Infarction", "History of MI", "Error", "Unknown"])
        self.assertTrue(0.0 <= schema_instance.confidence_score <= 1.0)
        self.assertIsInstance(schema_instance.affected_leads, list)
        self.assertIsInstance(schema_instance.lead_analysis_breakdown, dict)
        self.assertIn(schema_instance.suspected_artery, [
            "Left Anterior Descending (LAD)",
            "Left Circumflex (LCx)",
            "Right Coronary Artery (RCA)",
            "N/A",
            "N/A - No acute blockage detected"
        ])


if __name__ == "__main__":
    unittest.main()
