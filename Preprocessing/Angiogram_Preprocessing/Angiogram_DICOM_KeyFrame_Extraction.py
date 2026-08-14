import cv2
import numpy as np
import pandas as pd
import json
from pathlib import Path
import pydicom
from datetime import datetime
from skimage.metrics import structural_similarity as ssim


# -----------------------------
# 1. Validation
# -----------------------------
def validate_input(file_path: Path) -> bool:
    try:
        ds = pydicom.dcmread(file_path)
        if hasattr(ds, "PixelData"):
            return True
    except:
        pass

    file_path = Path(file_path)
    if file_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        img = cv2.imread(str(file_path))
        if img is not None:
            return True

    cap = cv2.VideoCapture(str(file_path))
    if cap.isOpened():
        cap.release()
        return True

    return False


# -----------------------------
# 2. Read DICOM / MP4 / Image
# -----------------------------
def read_angiogram(file_path: Path):
    try:
        ds = pydicom.dcmread(file_path)
        if hasattr(ds, "PixelData"):
            frames = ds.pixel_array.astype(np.float32)
            if len(frames.shape) == 2:
                frames = np.expand_dims(frames, axis=0)

            metadata = {
                "Source": "DICOM",
                "Modality": ds.get("Modality", "Unknown"),
                "NumberOfFrames": frames.shape[0]
            }
            return metadata, frames
    except:
        pass

    file_path = Path(file_path)
    if file_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            frames = np.expand_dims(img.astype(np.float32), axis=0)
            metadata = {
                "Source": "IMAGE",
                "Modality": "XA (Image)",
                "NumberOfFrames": 1
            }
            return metadata, frames

    cap = cv2.VideoCapture(str(file_path))
    if cap.isOpened():
        frame_list = []
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if frame.ndim == 2:
                gray = frame
            elif frame.ndim == 3 and frame.shape[2] == 1:
                gray = frame[:, :, 0]
            elif frame.ndim == 3 and frame.shape[2] >= 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            frame_list.append(gray.astype(np.float32))
        cap.release()

        if frame_list:
            frames = np.array(frame_list)
            metadata = {
                "Source": "MP4",
                "Modality": "XA (Simulated)",
                "NumberOfFrames": frames.shape[0]
            }
            return metadata, frames

    raise ValueError("Unsupported or unreadable file format.")


# -----------------------------
# 3. Extract Frames
# -----------------------------
def extract_frames(frame_array: np.ndarray):
    return [frame_array[i] for i in range(frame_array.shape[0])]


# -----------------------------
# 4. Frame Quality Metrics
# -----------------------------
def compute_frame_quality_metrics(frame: np.ndarray):
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    mean_intensity = np.mean(gray)
    contrast = np.std(gray)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength = np.mean(np.sqrt(sobel_x**2 + sobel_y**2))

    noise = np.std(gray - cv2.GaussianBlur(gray, (5, 5), 0))

    return {
        "mean_intensity": mean_intensity,
        "contrast": contrast,
        "edge_strength": edge_strength,
        "noise": noise
    }


def compute_diagnostic_score(frame, reference):
    data_range = frame.max() - frame.min() + 1e-6
    score, _ = ssim(reference, frame, full=True, data_range=data_range)
    return 1 - score


def normalize_frames(frames: np.ndarray):
    normalized = []
    for frame in frames:
        norm = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        normalized.append(norm.astype(np.uint8))
    return np.array(normalized)


def compute_sharpness(frame):
    return cv2.Laplacian(frame, cv2.CV_64F).var()


def select_best_frames(frames: np.ndarray, top_k=3):
    sharpness_scores = [compute_sharpness(f) for f in frames]
    sorted_indices = np.argsort(sharpness_scores)[::-1]
    return sorted_indices[:top_k]


def process_angiogram(file_path: Path, output_root: str = "output"):
    file_path = Path(file_path)
    patient_id = file_path.stem
    patient_output_dir = Path(output_root) / patient_id
    patient_output_dir.mkdir(parents=True, exist_ok=True)

    metadata, frames = read_angiogram(file_path)

    if frames.shape[0] == 1:
        selected_indices = [0]
    else:
        selected_indices = select_best_frames(frames, top_k=min(3, frames.shape[0]))

    variants = []
    for rank, idx in enumerate(selected_indices):
        frame = frames[idx]
        norm = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

        filename = f"{patient_id}_frame_{idx}_rank_{rank + 1}.png"
        filepath = patient_output_dir / filename
        cv2.imwrite(str(filepath), norm)

        metrics = compute_frame_quality_metrics(norm)
        variants.append({
            "rank": rank + 1,
            "frame_index": int(idx),
            "filename": filename,
            "path": str(filepath),
            "label": f"Keyframe #{rank + 1} (Frame {idx})",
            "sharpness": float(compute_sharpness(norm)),
            "contrast": float(metrics["contrast"]),
            "edge_strength": float(metrics["edge_strength"])
        })

    pipeline_metadata = {
        "patient_id": patient_id,
        "processed_at": datetime.now().isoformat(),
        "source": metadata.get("Source", "Unknown"),
        "number_of_original_frames": int(metadata.get("NumberOfFrames", 1)),
        "selected_frame_indices": [int(i) for i in selected_indices],
        "output_directory": str(patient_output_dir),
        "variants": variants
    }

    metadata_path = patient_output_dir / "pipeline_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(pipeline_metadata, f, indent=4)

    return pipeline_metadata