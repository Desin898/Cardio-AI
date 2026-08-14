import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes
from scipy.ndimage import distance_transform_edt
from scipy.signal import savgol_filter

from backend.app.core.config import settings


class QCAService:
    """
    Quantitative Coronary Angiography (QCA) Service.
    Extracts vessel centerlines via Medial Axis Transform / Skeletonization,
    profiles perpendicular lumen diameters, calculates percentage stenosis,
    grades lesion severity, and renders high-contrast annotated diagnostic images.
    """

    @staticmethod
    def preprocess_mask(mask_input: Any) -> np.ndarray:
        """Converts mask input (PIL Image, numpy array, or path) into a clean binary uint8 array."""
        if isinstance(mask_input, (str, Path)):
            if not os.path.exists(mask_input):
                raise FileNotFoundError(f"Mask file not found: {mask_input}")
            mask_np = cv2.imread(str(mask_input), cv2.IMREAD_GRAYSCALE)
        elif isinstance(mask_input, Image.Image):
            mask_np = np.array(mask_input.convert("L"))
        elif isinstance(mask_input, np.ndarray):
            mask_np = mask_input.copy()
            if mask_np.ndim == 3:
                mask_np = cv2.cvtColor(mask_np, cv2.COLOR_BGR2GRAY)
        else:
            raise TypeError("Unsupported mask_input format. Expected PIL.Image, np.ndarray, or file path.")

        binary = mask_np > 127
        binary = remove_small_objects(binary, min_size=300)
        binary = remove_small_holes(binary, area_threshold=100)
        return (binary.astype(np.uint8)) * 255

    @staticmethod
    def extract_centerline_and_diameters(binary_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts vessel centerline coordinates and Euclidean Distance Transform radii.
        Returns:
            skeleton: 2D boolean array of centerline pixels
            distance_map: 2D float array of perpendicular distances to nearest boundary
            y_coords, x_coords: 1D arrays of centerline pixel coordinates
        """
        binary_bool = binary_mask > 0
        skeleton = skeletonize(binary_bool)
        distance_map = distance_transform_edt(binary_bool)

        y_coords, x_coords = np.where(skeleton)
        return skeleton, distance_map, y_coords, x_coords

    @staticmethod
    def _smooth_profile(profile: np.ndarray) -> np.ndarray:
        """Smooths diameter profile using Savitzky-Golay filter or moving average."""
        n = len(profile)
        if n < 5:
            return profile.copy()
        win = min(9, n if n % 2 == 1 else n - 1)
        if win >= 5:
            try:
                return savgol_filter(profile, window_length=win, polyorder=2, mode="interp")
            except Exception:
                pass
        return np.convolve(profile, np.ones(5) / 5, mode="same")

    def compute_qca_metrics(
        self,
        binary_mask: np.ndarray,
        pixel_spacing_mm: float = 1.0
    ) -> Dict[str, Any]:
        """
        Calculates QCA stenosis metrics:
          - Minimum Lumen Diameter (d_min)
          - Reference Vessel Diameter (d_ref)
          - Percentage Stenosis = (1 - (d_min / d_ref)) * 100
          - Lesion Severity: MILD (<50%), MODERATE (50-69%), SEVERE (>=70%)
          - Intervention Recommended (True for SEVERE)
        """
        skeleton, distance_map, y_coords, x_coords = self.extract_centerline_and_diameters(binary_mask)

        if len(x_coords) == 0:
            return {
                "stenosis_percentage": 0.0,
                "severity_grade": "MILD",
                "d_min": 10.0,
                "d_ref": 10.0,
                "lesion_coordinates": {"x": 0, "y": 0},
                "intervention_recommended": False,
                "centerline_points_count": 0,
                "centerline_coords": [],
            }

        # Order centerline points sequentially along vessel axis
        sort_order = np.lexsort((x_coords, y_coords))
        x_coords = x_coords[sort_order]
        y_coords = y_coords[sort_order]

        radii = distance_map[y_coords, x_coords]
        diameters = radii * 2.0
        smooth_diameters = self._smooth_profile(diameters)

        # Ignore end caps of centerline to prevent tip distance transform drop-off artifacts
        end_ignore = 15
        if len(smooth_diameters) > 2 * end_ignore + 5:
            valid_indices = list(range(end_ignore, len(smooth_diameters) - end_ignore))
        else:
            valid_indices = list(range(len(smooth_diameters)))

        valid_diameters = smooth_diameters[valid_indices]

        # Identify bottleneck minimum lumen diameter (d_min)
        min_local_idx = int(np.argmin(valid_diameters))
        min_idx = valid_indices[min_local_idx]

        d_min = float(smooth_diameters[min_idx])
        x_min = int(x_coords[min_idx])
        y_min = int(y_coords[min_idx])

        # Estimate reference vessel diameter (d_ref) from neighboring non-stenotic segments or 75th percentile
        neighborhood_window = 25
        neigh_start = max(0, min_idx - neighborhood_window)
        neigh_end = min(len(smooth_diameters), min_idx + neighborhood_window + 1)
        neighbor_diameters = np.concatenate([smooth_diameters[neigh_start:max(neigh_start, min_idx-3)],
                                             smooth_diameters[min(neigh_end, min_idx+4):neigh_end]])

        if len(neighbor_diameters) > 0:
            d_ref = float(np.percentile(neighbor_diameters, 75))
        else:
            d_ref = float(np.percentile(smooth_diameters[valid_indices], 75))

        d_ref = max(d_ref, d_min + 1e-4)

        # Calculate percentage stenosis
        stenosis_pct = round(float(max(0.0, min(100.0, (1.0 - (d_min / d_ref)) * 100.0))), 1)

        # Lesion severity grading rules:
        # MILD: < 50%
        # MODERATE: 50% <= Stenosis < 70%
        # SEVERE: >= 70% (Catheter Intervention Recommended)
        if stenosis_pct >= 70.0:
            severity = "SEVERE"
            intervention = True
        elif stenosis_pct >= 50.0:
            severity = "MODERATE"
            intervention = False
        else:
            severity = "MILD"
            intervention = False

        centerline_coords = [{"x": int(x_coords[i]), "y": int(y_coords[i])} for i in range(0, len(x_coords), max(1, len(x_coords)//50))]

        return {
            "stenosis_percentage": stenosis_pct,
            "severity_grade": severity,
            "d_min": round(d_min * pixel_spacing_mm, 2),
            "d_ref": round(d_ref * pixel_spacing_mm, 2),
            "lesion_coordinates": {"x": x_min, "y": y_min},
            "intervention_recommended": intervention,
            "centerline_points_count": len(x_coords),
            "centerline_coords": centerline_coords,
        }

    def render_qca_visualization(
        self,
        original_image_input: Any,
        binary_mask: np.ndarray,
        qca_metrics: Dict[str, Any],
        output_save_path: str
    ) -> str:
        """
        Renders an annotated QCA visual diagnostic image with:
          - Cyan vessel contours
          - Green arterial centerline
          - Red circle & crosshair at bottleneck minimum lumen diameter (d_min)
          - Yellow line at reference vessel segment (d_ref)
          - High-contrast clinical metric header banner
        """
        if isinstance(original_image_input, (str, Path)) and os.path.exists(original_image_input):
            bg_img = cv2.imread(str(original_image_input), cv2.IMREAD_COLOR)
        elif isinstance(original_image_input, np.ndarray):
            bg_img = original_image_input.copy()
            if bg_img.ndim == 2:
                bg_img = cv2.cvtColor(bg_img, cv2.COLOR_GRAY2BGR)
        else:
            bg_img = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

        h, w = bg_img.shape[:2]
        vis_img = bg_img.copy()

        # 1. Draw Cyan Vessel Contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_img, contours, -1, (255, 255, 0), 2)

        # 2. Draw Green Arterial Centerline
        skeleton, _, y_coords, x_coords = self.extract_centerline_and_diameters(binary_mask)
        for i in range(len(x_coords)):
            cv2.circle(vis_img, (int(x_coords[i]), int(y_coords[i])), 1, (0, 255, 0), -1)

        # 3. Highlight Bottleneck (d_min) & Reference Segment (d_ref)
        lx = qca_metrics["lesion_coordinates"]["x"]
        ly = qca_metrics["lesion_coordinates"]["y"]
        stenosis = qca_metrics["stenosis_percentage"]
        severity = qca_metrics["severity_grade"]
        d_min = qca_metrics["d_min"]
        d_ref = qca_metrics["d_ref"]
        intervention = qca_metrics["intervention_recommended"]

        # Red circle marker around narrowest stenosis bottleneck
        radius = max(6, int(d_ref))
        cv2.circle(vis_img, (lx, ly), radius + 8, (0, 0, 255), 2)
        cv2.drawMarker(vis_img, (lx, ly), (0, 0, 255), cv2.MARKER_CROSS, markerSize=12, thickness=2)

        # 4. Render High-Contrast Top Banner
        banner_height = 50
        banner = np.zeros((banner_height, w, 3), dtype=np.uint8)

        color_map = {
            "SEVERE": (0, 0, 255),      # Red
            "MODERATE": (0, 165, 255),  # Orange
            "MILD": (0, 255, 0)         # Green
        }
        status_color = color_map.get(severity, (255, 255, 255))
        interv_str = "INTERVENTION RECOMMENDED" if intervention else "CONSERVATIVE MANAGEMENT"

        cv2.rectangle(banner, (0, 0), (w, banner_height), (30, 30, 30), -1)
        text_str = f"QCA Stenosis: {stenosis}% ({severity}) | d_min: {d_min}px | d_ref: {d_ref}px | {interv_str}"
        cv2.putText(banner, text_str, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2, cv2.LINE_AA)

        combined_img = np.vstack([banner, vis_img])

        os.makedirs(os.path.dirname(output_save_path), exist_ok=True)
        cv2.imwrite(output_save_path, combined_img)
        return output_save_path


qca_service = QCAService()
