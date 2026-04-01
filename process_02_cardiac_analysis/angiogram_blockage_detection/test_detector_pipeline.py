import cv2
from pathlib import Path
from inference.detector_pipeline import StenosisDetector

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_PATH = PROJECT_ROOT / "test_image.png"
OUTPUT_PATH = PROJECT_ROOT / "test_detector_overlay.png"

detector = StenosisDetector()

image = cv2.imread(str(IMAGE_PATH))
if image is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

result = detector.detect(image, return_debug=True)

print("Predicted boxes:", result["pred_boxes"])
print("Number of boxes:", len(result["pred_boxes"]))
print("Detected blockages:", result["blockages"])

overlay = image.copy()
for box in result["pred_boxes"]:
    x, y, w, h = box
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imwrite(str(OUTPUT_PATH), overlay)
print(f"Overlay saved to: {OUTPUT_PATH}")