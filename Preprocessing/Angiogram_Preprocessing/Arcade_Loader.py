"""
ARCADE Dataset Loader
---------------------

Loads:
1. Top-hat enhanced angiogram images
2. ARCADE annotation JSON

Outputs structured samples for evaluation.

"""

import json
import cv2
import numpy as np
from pathlib import Path


class ArcadeLoader:
    """
    Loader for ARCADE angiogram dataset
    """

    def __init__(self, image_dir, annotation_path):
        """
        Parameters
        ----------
        image_dir : str
            Directory containing top-hat enhanced angiogram images

        annotation_path : str
            Path to ARCADE JSON annotation file
        """

        self.image_dir = Path(image_dir)
        self.annotation_path = Path(annotation_path)

        # Load annotations
        with open(self.annotation_path, "r") as f:
            self.data = json.load(f)

        # Store useful sections
        self.images = self.data["images"]
        self.annotations = self.data["annotations"]

        # Map image_id -> annotations
        self.annotation_map = self._build_annotation_map()

        print(f"Loaded {len(self.images)} images")
        print(f"Loaded {len(self.annotations)} annotations")

    # Build annotation lookup
    def _build_annotation_map(self):
        """
        Creates a dictionary mapping image_id -> annotation list
        """

        annotation_map = {}

        for ann in self.annotations:
            image_id = ann["image_id"]

            if image_id not in annotation_map:
                annotation_map[image_id] = []

            annotation_map[image_id].append(ann)

        return annotation_map

    def load_image(self, file_name):
        """
        Load angiogram image
        """

        # Extract only the filename (removes 'images/' if present)
        file_name = Path(file_name).name

        image_path = self.image_dir / file_name

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(str(image_path))

        return image

    # Extract bounding boxes
    def extract_bboxes(self, annotations):
        """
        Extract bounding boxes from annotations
        """

        boxes = []

        for ann in annotations:

            if "bbox" in ann:
                x, y, w, h = ann["bbox"]
                boxes.append([int(x), int(y), int(w), int(h)])

        return boxes

    # Extract segmentation polygons
    def extract_polygons(self, annotations):
        """
        Extract segmentation polygons if available
        """

        polygons = []

        for ann in annotations:

            if "segmentation" in ann:
                segmentation = ann["segmentation"]

                for poly in segmentation:
                    polygon = np.array(poly).reshape(-1, 2)
                    polygons.append(polygon)

        return polygons

    # Load dataset samples
    def load_samples(self):
        """
        Load dataset samples

        Returns
        -------
        list of dict
        """

        samples = []

        for img in self.images:

            image_id = img["id"]
            file_name = img["file_name"]

            # Load image
            image = self.load_image(file_name)

            # Get annotations
            anns = self.annotation_map.get(image_id, [])

            boxes = self.extract_bboxes(anns)
            polygons = self.extract_polygons(anns)

            sample = {
                "image_id": image_id,
                "file_name": file_name,
                "image": image,
                "bboxes": boxes,
                "polygons": polygons
            }

            samples.append(sample)

        return samples

    # Visualization (for validation)
    def visualize_sample(self, sample, save_path=None):
        """
        Draw ground truth boxes on image
        """

        image = sample["image"].copy()

        for bbox in sample["bboxes"]:

            x, y, w, h = map(int, bbox)

            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        if save_path:
            cv2.imwrite(str(save_path), image)

        return image


# Example usage
if __name__ == "__main__":
    from pathlib import Path

    IMAGE_DIR = Path("path/to/preprocessed/images")
    ANNOTATION_FILE = "path/to/arcade/annotations/train.json"

    loader = ArcadeLoader(
        image_dir=IMAGE_DIR,
        annotation_path=ANNOTATION_FILE
    )

    samples = loader.load_samples()

    print(f"Loaded {len(samples)} samples")

    # visualize first sample
    sample = samples[0]

    loader.visualize_sample(
        sample,
        save_path="sample_visualization.png"
    )

    print("Visualization saved")

