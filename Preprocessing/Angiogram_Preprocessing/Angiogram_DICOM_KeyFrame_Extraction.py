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

    cap = cv2.VideoCapture(str(file_path))
    if cap.isOpened():
        cap.release()
        return True

    return False


# -----------------------------
# 2. Read DICOM / MP4
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

    cap = cv2.VideoCapture(str(file_path))
    if cap.isOpened():
        frame_list = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_list.append(gray.astype(np.float32))
        cap.release()

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


# -----------------------------
# 5. Frame Selection
# -----------------------------
def select_top_k_with_spacing(df, k=5, min_frame_gap=3):
    selected_frames = []
    selected_indices = []

    for _, row in df.iterrows():
        frame_idx = row["frame_index"]

        if all(abs(frame_idx - idx) >= min_frame_gap for idx in selected_indices):
            selected_frames.append(row)
            selected_indices.append(frame_idx)

        if len(selected_frames) == k:
            break

    return pd.DataFrame(selected_frames)


# -----------------------------
# 6. Preprocessing
# -----------------------------
def enhanced_tophat_filter(img, kernel_ratio=0.04, apply_clahe=True):
    img = img.astype(np.uint8)

    k = int(img.shape[1] * kernel_ratio)
    k = max(15, k | 1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    white = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    black = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

    enhanced = cv2.add(img, white)
    enhanced = cv2.subtract(enhanced, black)

    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(enhanced)

    return enhanced


def preprocess_angiogram_frame(frame: np.ndarray, target_size=(512, 512)):
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)

    normalized = cv2.normalize(
        resized, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    enhanced = enhanced_tophat_filter(normalized)
    return enhanced


# -----------------------------
# 7. Main Entry Function
# -----------------------------
def process_angiogram(file_path: str, output_root="Preprocessed_Angiogram_Output"):
    """
    Run the full angiogram preprocessing pipeline.

    Parameters
    ----------
    file_path : str
        Path to a DICOM file or MP4 video.
    output_root : str or Path
        Root directory where the patient sub-folder will be created.

    Returns
    -------
    dict with keys:
        patient_id              – unique ID for this run
        output_directory        – absolute path to the patient output folder
        selected_frame_indices  – list of original frame indices that were kept
        variants                – list of dicts, one per saved frame:
                                    { "label":    "Frame 1 (original index 20)",
                                      "filename": "frame_01.png" }
                                  Consumed directly by metadata['angiogram']['variants']
                                  in the Flask doctor portal flow.
    """
    file_path = Path(file_path)

    if not validate_input(file_path):
        raise ValueError("Invalid angiogram file.")

    metadata, frames = read_angiogram(file_path)
    frame_list = extract_frames(frames)

    frame_metrics = []

    # Baseline reference frame (first frame of angiogram)
    reference = frame_list[0]

    for idx, frame in enumerate(frame_list):
        metrics = compute_frame_quality_metrics(frame)
        metrics["frame_index"] = idx

        # SSIM-based diagnostic score
        metrics["diagnostic_score"] = compute_diagnostic_score(frame, reference)

        frame_metrics.append(metrics)

    metrics_df = pd.DataFrame(frame_metrics).sort_values(
        by="diagnostic_score", ascending=False
    ).reset_index(drop=True)

    key_frames_df = select_top_k_with_spacing(metrics_df)

    selected_indices = key_frames_df["frame_index"].astype(int).tolist()
    selected_frames = [frame_list[idx] for idx in selected_indices]

    preprocessed_frames = [
        preprocess_angiogram_frame(frame)
        for frame in selected_frames
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    patient_id = f"{file_path.stem}_{timestamp}"

    output_root = Path(output_root)
    patient_dir = output_root / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)

    # Save frames and build variants list
    variants = []
    for i, frame in enumerate(preprocessed_frames, start=1):
        filename = f"frame_{i:02d}.png"
        cv2.imwrite(str(patient_dir / filename), frame)
        variants.append({
            "label":    f"Frame {i} (original index {selected_indices[i - 1]})",
            "filename": filename,
        })

    # Save pipeline metadata
    metadata_output = {
        "patient_id":               patient_id,
        "source":                   metadata["Source"],
        "number_of_original_frames": metadata["NumberOfFrames"],
        "selected_frame_indices":   selected_indices,
        "variants":                 variants,
    }

    with open(patient_dir / "metadata.json", "w") as f:
        json.dump(metadata_output, f, indent=4)

    return {
        "patient_id":             patient_id,
        "output_directory":       str(patient_dir),
        "selected_frame_indices": selected_indices,
        "variants":               variants,   # ← consumed by Flask upload_angiogram route
    }