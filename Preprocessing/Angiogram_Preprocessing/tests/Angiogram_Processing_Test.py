import unittest
import numpy as np
import pandas as pd
import cv2
from pathlib import Path

from Preprocessing.Angiogram_Preprocessing.Angiogram_DICOM_KeyFrame_Extraction import (
    validate_input,
    read_angiogram,
    extract_frames,
    compute_frame_quality_metrics,
    compute_diagnostic_score,
    select_top_k_with_spacing,
    enhanced_tophat_filter,
    preprocess_angiogram_frame,
    process_angiogram,
)


class TestAngiogramProcessing(unittest.TestCase):

    def setUp(self):
        # DICOM file path (no extension)
        self.dicom_path = Path(
            r"C:\Users\User\PycharmProjects\CM2603-DSGP-Group-3\Preprocessing"
            r"\Angiogram_Preprocessing\test_data\Patient-0001"
        )

        # Verify file exists before any test runs
        self.assertTrue(self.dicom_path.exists(), f"Test DICOM not found: {self.dicom_path}")

        # Synthetic float32 frame for image-processing unit tests
        self.frame = np.random.randint(0, 255, (512, 512), dtype=np.uint8).astype(np.float32)


    # Test 1: Input validation
    def test_validate_input(self):
        result = validate_input(self.dicom_path)
        self.assertTrue(result)

    # Test 2: Reading angiogram (DICOM)
    def test_read_angiogram_dicom(self):
        metadata, frames = read_angiogram(self.dicom_path)
        self.assertEqual(metadata["Source"], "DICOM")
        self.assertGreater(frames.shape[0], 0)


    # Test 3: Frame extraction
    def test_extract_frames(self):
        fake_stack = np.random.randint(0, 255, (10, 512, 512))
        frames = extract_frames(fake_stack)
        self.assertEqual(len(frames), 10)

    # Test 4: Frame quality metrics
    def test_compute_frame_quality_metrics(self):
        metrics = compute_frame_quality_metrics(self.frame)
        self.assertIn("mean_intensity", metrics)
        self.assertIn("contrast", metrics)
        self.assertIn("edge_strength", metrics)
        self.assertIn("noise", metrics)

    # Test 5: Diagnostic score
    def test_compute_diagnostic_score(self):
        # Identical frames -> score should be ~0 (1 - SSIM ≈ 1 - 1 = 0)
        score = compute_diagnostic_score(self.frame, self.frame)
        self.assertTrue(np.isfinite(score), "Score must be a finite number")
        self.assertAlmostEqual(score, 0.0, delta=0.05,
                               msg="Identical frames should yield a score near 0")

    # Test 6: Frame selection
    def test_select_top_k_with_spacing(self):
        df = pd.DataFrame({
            "frame_index":      list(range(10)),
            "diagnostic_score": np.linspace(1, 0, 10),
        })
        # min_frame_gap=1 ensures exactly k=3 frames are always selectable
        selected = select_top_k_with_spacing(df, k=3, min_frame_gap=1)
        self.assertEqual(len(selected), 3)

    # Test 7: Top-hat enhancement
    def test_enhanced_tophat_filter(self):
        img = self.frame.astype(np.uint8)
        enhanced = enhanced_tophat_filter(img)
        self.assertEqual(enhanced.shape, img.shape)
        self.assertEqual(enhanced.dtype, np.uint8)

    # Test 8: Frame preprocessing
    def test_preprocess_angiogram_frame(self):
        processed = preprocess_angiogram_frame(self.frame)
        self.assertEqual(processed.shape, (512, 512))
        self.assertEqual(processed.dtype, np.uint8)

    # Test 9: Full pipeline integration
    def test_process_angiogram_pipeline(self):
        result = process_angiogram(str(self.dicom_path))
        self.assertIn("patient_id", result)
        self.assertIn("output_directory", result)
        self.assertIn("selected_frame_indices", result)


if __name__ == "__main__":
    unittest.main()
