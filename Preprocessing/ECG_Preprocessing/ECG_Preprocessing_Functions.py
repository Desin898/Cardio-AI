import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import glob


# ---------- STEP A: image preprocessing (grayscale, denoise, threshold) ----------
def preprocess_step1_image(img):
    """Take BGR image (cv2.imread) and return preprocessed binary image (threshold)."""
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    thr = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 7
    )
    return thr


# ---------- STEP B: smart crop into 12 leads ----------
def crop_12_leads_from_gray(gray_img):
    """Input: gray image (2D). Output: dict Lead_1..Lead_12 -> crop images (grayscale)."""
    h, w = gray_img.shape[:2]
    top_margin = int(0.18 * h)
    lead_height = int((h - top_margin) / 4)
    lead_width = int(w / 3)

    leads = {}
    idx = 1
    for row in range(4):
        for col in range(3):
            y1 = top_margin + row * lead_height
            y2 = top_margin + (row + 1) * lead_height
            x1 = col * lead_width
            x2 = (col + 1) * lead_width
            crop = gray_img[y1:y2, x1:x2].copy()
            leads[f"Lead_{idx}"] = crop
            idx += 1
    return leads


# ---------- STEP C: clean lead (blur, threshold, morphology) ----------
def clean_lead_for_signal(lead_img):
    """Return single cleaned binary image for signal extraction (uint8)."""
    gray = lead_img if len(lead_img.shape) == 2 else cv2.cvtColor(lead_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        41, 5
    )
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return clean


# ---------- STEP D: extract 1D signal from cleaned lead ----------
def extract_signal(clean_img):
    """Return normalized 1D signal (length = width of clean_img). Values in [0,1]."""
    clean = clean_img.astype(np.uint8)
    h, w = clean.shape
    signal = []

    for x in range(w):
        column = clean[:, x]
        pts = np.where(column > 0)[0]
        if len(pts) == 0:
            signal.append(signal[-1] if len(signal) > 0 else h // 2)
        else:
            y = int(np.mean(pts))
            signal.append(y)

    sig = np.array(signal, dtype=float)

    if sig.max() - sig.min() < 1e-8:
        return np.zeros_like(sig)

    return (sig - sig.min()) / (sig.max() - sig.min())


# ---------- STEP E: resample to fixed length ----------
def resample_signal(sig, target_len):
    """Resample 1D vector sig to length target_len via linear interpolation."""
    if len(sig) == target_len:
        return sig
    if len(sig) == 0:
        return np.zeros(target_len, dtype=float)

    old_x = np.linspace(0, 1, num=len(sig))
    new_x = np.linspace(0, 1, num=target_len)
    return np.interp(new_x, old_x, sig)


# ---------- small helper to write images if requested ----------
def save_intermediate_images(image_dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, img in image_dict.items():
        cv2.imwrite(str(out_dir / f"{name}.png"), img)


# Fixed target lead length (computed once from training dataset)
TARGET_LEAD_LENGTH = 737


# ---------- Single-image pipeline function ----------
def process_single_ecg_image(
    img_path,
    target_len=TARGET_LEAD_LENGTH,
    save_images=False,
    save_folder=None
):
    """
    Process one ECG image file path.
    Returns: flattened vector (1D numpy) length = 12 * target_len
    If save_images True and save_folder provided, saves intermediate images there.
    """

    p = Path(img_path)
    bgr = cv2.imread(str(p))

    if bgr is None:
        print("WARN: cannot read", img_path)
        return None

    # Step 1: preprocess
    thr = preprocess_step1_image(bgr)

    if save_images and save_folder:
        save_intermediate_images(
            {"step1_threshold": thr},
            Path(save_folder) / p.stem
        )

    # Step 2: crop using original grayscale
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    leads = crop_12_leads_from_gray(gray)

    if save_images and save_folder:
        save_intermediate_images(
            {f"crop_{k}": v for k, v in leads.items()},
            Path(save_folder) / p.stem / "crops"
        )

    # Step 3: clean → extract → resample
    all_lead_signals = []

    for i in range(1, 13):
        lead = leads[f"Lead_{i}"]
        clean = clean_lead_for_signal(lead)

        sig = extract_signal(clean)
        sig_rs = resample_signal(sig, target_len)
        all_lead_signals.append(sig_rs)

        if save_images and save_folder:
            save_intermediate_images(
                {f"clean_lead_{i}": clean},
                Path(save_folder) / p.stem / "cleaned"
            )

    flattened = np.concatenate(all_lead_signals, axis=0)
    return flattened