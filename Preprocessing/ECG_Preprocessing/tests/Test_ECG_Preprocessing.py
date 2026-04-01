import unittest
import numpy as np
import cv2
from pathlib import Path

from Preprocessing.ECG_Preprocessing.ECG_Preprocessing_Functions import (
    preprocess_step1_image,
    crop_12_leads_from_gray,
    clean_lead_for_signal,
    extract_signal,
    resample_signal,
    process_single_ecg_image,
    TARGET_LEAD_LENGTH
)


class TestECGPreprocessing(unittest.TestCase):

    def setUp(self):

        base_dir = Path(__file__).resolve().parent.parent
        self.test_image_path = base_dir / "test_data" / "MI(15).jpg"

        self.test_img = cv2.imread(str(self.test_image_path))

        self.assertIsNotNone(self.test_img)

        self.gray_img = cv2.cvtColor(self.test_img, cv2.COLOR_BGR2GRAY)

    # TEST 1 — Image preprocessing
    def test_preprocess_step1_image(self):
        thr = preprocess_step1_image(self.test_img)

        self.assertIsNotNone(thr)
        self.assertEqual(len(thr.shape), 2)
        self.assertTrue(thr.dtype == np.uint8)

    # TEST 2 — Cropping into 12 leads
    def test_crop_12_leads(self):
        leads = crop_12_leads_from_gray(self.gray_img)

        self.assertEqual(len(leads), 12)
        self.assertIn("Lead_1", leads)
        self.assertIn("Lead_12", leads)

        for lead in leads.values():
            self.assertTrue(len(lead.shape) == 2)

    # TEST 3 — Lead cleaning
    def test_clean_lead_for_signal(self):
        lead = self.gray_img[0:200, 0:200]

        clean = clean_lead_for_signal(lead)

        self.assertEqual(clean.shape, lead.shape)
        self.assertTrue(clean.dtype == np.uint8)

    # TEST 4 — Signal extraction
    def test_extract_signal(self):
        binary = np.random.randint(0,2,(200,300))*255

        signal = extract_signal(binary)

        self.assertEqual(len(signal), 300)
        self.assertTrue(signal.min() >= 0)
        self.assertTrue(signal.max() <= 1)

    # TEST 5 — Signal resampling
    def test_resample_signal(self):
        sig = np.linspace(0,1,100)

        resampled = resample_signal(sig, 200)

        self.assertEqual(len(resampled), 200)

    # TEST 6 — Full pipeline
    def test_full_pipeline(self):

        result = process_single_ecg_image(self.test_image_path)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 12 * TARGET_LEAD_LENGTH)


if __name__ == "__main__":
    unittest.main()