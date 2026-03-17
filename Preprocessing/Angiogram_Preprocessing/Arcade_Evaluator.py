"""
ARCADE Evaluator
----------------

Evaluates stenosis detection results using ARCADE dataset.

Computes:
- True Positives
- False Positives
- False Negatives
- Precision
- Recall
- F1 Score

Also saves visualization images comparing predictions with ground truth.
"""

import cv2
import numpy as np
from pathlib import Path

from Arcade_Loader import ArcadeLoader


class ArcadeEvaluator:

    def __init__(self, detector, iou_threshold=0.5, output_dir="outputs/arcade_eval"):
        """
        Parameters
        ----------
        detector : object
            Stenosis detection system (VesselAnalyzer)

        iou_threshold : float
            IoU threshold to determine TP

        output_dir : str
            Directory to save visualization outputs
        """

        self.detector = detector
        self.iou_threshold = iou_threshold

        self.tp = 0
        self.fp = 0
        self.fn = 0

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # IoU calculation
    def compute_iou(self, boxA, boxB):
        """
        Compute Intersection over Union
        """

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        inter_area = max(0, xB - xA) * max(0, yB - yA)

        boxA_area = boxA[2] * boxA[3]
        boxB_area = boxB[2] * boxB[3]

        union_area = boxA_area + boxB_area - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    # Draw visualization
    def visualize(self, image, gt_boxes, pred_boxes, image_id):
        """
        Save visualization showing GT vs predictions
        """

        vis = image.copy()

        # Draw ground truth boxes (GREEN)
        for box in gt_boxes:
            x, y, w, h = map(int, box)
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw predicted boxes (RED)
        for box in pred_boxes:
            x, y, w, h = map(int, box)
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)

        save_path = self.output_dir / f"eval_{image_id}.png"
        cv2.imwrite(str(save_path), vis)

    # Evaluate single sample
    def evaluate_sample(self, sample):

        image = sample["image"]
        gt_boxes = sample["bboxes"]
        image_id = sample["image_id"]

        # Run stenosis detector
        pred_boxes = self.detector.detect(image)

        matched_gt = set()

        for pred in pred_boxes:

            best_iou = 0
            best_gt_idx = -1

            for i, gt in enumerate(gt_boxes):

                iou = self.compute_iou(pred, gt)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            if best_iou >= self.iou_threshold and best_gt_idx not in matched_gt:
                self.tp += 1
                matched_gt.add(best_gt_idx)
            else:
                self.fp += 1

        self.fn += len(gt_boxes) - len(matched_gt)

        # Save visualization
        self.visualize(image, gt_boxes, pred_boxes, image_id)

    # Compute final metrics
    def compute_metrics(self):

        precision = self.tp / (self.tp + self.fp + 1e-6)
        recall = self.tp / (self.tp + self.fn + 1e-6)

        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        return {
            "true_positives": self.tp,
            "false_positives": self.fp,
            "false_negatives": self.fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    # Evaluate entire dataset
    def evaluate_dataset(self, samples):

        print("Starting ARCADE evaluation...\n")

        for i, sample in enumerate(samples):

            self.evaluate_sample(sample)

            if (i + 1) % 10 == 0:
                print(f"Processed {i+1} images")

        metrics = self.compute_metrics()

        print("\nEvaluation Results")
        print("-------------------")
        for k, v in metrics.items():
            print(f"{k}: {v}")

        return metrics


# Example usage
if __name__ == "__main__":

    # Dummy detector for testing
    class DummyDetector:
        def detect(self, image):
            return [[100, 150, 40, 60]]

    IMAGE_DIR = "path/to/preprocessed/images"
    ANNOTATION_FILE = "path/to/arcade/annotations/train.json"

    loader = ArcadeLoader(
        image_dir=IMAGE_DIR,
        annotation_path=ANNOTATION_FILE
    )

    samples = loader.load_samples()

    detector = DummyDetector()

    evaluator = ArcadeEvaluator(detector)

    evaluator.evaluate_dataset(samples)