import cv2
import numpy as np
from pathlib import Path

from arcade_loader import ArcadeLoader
from detector_pipeline import StenosisDetector


class ArcadeEvaluator:
    """
    Evaluates stenosis detection results using the ARCADE dataset.

    Computes:
    - True Positives
    - False Positives
    - False Negatives
    - Precision
    - Recall
    - F1 Score

    Also saves visualization images comparing predictions with ground truth.
    """

    def __init__(self, detector, iou_threshold=0.5, output_dir="outputs/arcade_eval"):
        self.detector = detector
        self.iou_threshold = iou_threshold

        self.tp = 0
        self.fp = 0
        self.fn = 0

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compute_iou(self, boxA, boxB):
        """
        Compute Intersection over Union for [x, y, w, h] boxes.
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        inter_area = max(0, xB - xA) * max(0, yB - yA)

        boxA_area = max(0, boxA[2]) * max(0, boxA[3])
        boxB_area = max(0, boxB[2]) * max(0, boxB[3])

        union_area = boxA_area + boxB_area - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def visualize(self, image, gt_boxes, pred_boxes, image_id):
        """
        Save visualization showing GT vs predictions.
        Green = ground truth
        Red = prediction
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

    def evaluate_sample(self, sample):
        """
        Evaluate one sample against ground truth.
        """
        image = sample["image"]
        gt_boxes = sample["bboxes"]
        image_id = sample["image_id"]

        # Run real stenosis detector
        pred_boxes = self.detector.detect(image)

        matched_gt = set()

        for pred in pred_boxes:
            best_iou = 0.0
            best_gt_idx = -1

            for i, gt in enumerate(gt_boxes):
                iou = self.compute_iou(pred, gt)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            matched = False

            if best_gt_idx != -1 and best_gt_idx not in matched_gt:
                if best_iou >= self.iou_threshold or self.point_in_box(pred, gt_boxes[best_gt_idx]):
                    matched = True

            if matched:
                self.tp += 1
                matched_gt.add(best_gt_idx)
            else:
                self.fp += 1

        self.fn += len(gt_boxes) - len(matched_gt)

        self.visualize(image, gt_boxes, pred_boxes, image_id)

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

    def evaluate_dataset(self, samples):
        print("Starting ARCADE evaluation...\n")

        for i, sample in enumerate(samples):
            self.evaluate_sample(sample)

            print(
                f"Processed {i + 1}/{len(samples)} | "
                f"TP={self.tp}, FP={self.fp}, FN={self.fn}"
            )

        metrics = self.compute_metrics()

        print("\nEvaluation Results")
        print("-------------------")
        for k, v in metrics.items():
            print(f"{k}: {v}")

        return metrics

    def point_in_box(self, pred_box, gt_box):
        """
        Check whether predicted box center lies inside ground-truth box.
        Boxes are [x, y, w, h].
        """
        px = pred_box[0] + pred_box[2] / 2.0
        py = pred_box[1] + pred_box[3] / 2.0

        gx, gy, gw, gh = gt_box
        return (gx <= px <= gx + gw) and (gy <= py <= gy + gh)


if __name__ == "__main__":
    IMAGE_DIR = r"C:\Users\desin\OneDrive\Desktop\2nd Year\DSGP\Angiogram\archive (2)\arcade\stenosis\train\images"
    ANNOTATION_FILE = r"C:\Users\desin\OneDrive\Desktop\2nd Year\DSGP\Angiogram\archive (2)\arcade\stenosis\train\annotations\train.json"

    loader = ArcadeLoader(
        image_dir=IMAGE_DIR,
        annotation_path=ANNOTATION_FILE
    )

    # Small subset first for integration testing
    samples = loader.load_samples()[:5]

    detector = StenosisDetector()

    evaluator = ArcadeEvaluator(detector=detector, iou_threshold=0.5)

    evaluator.evaluate_dataset(samples)