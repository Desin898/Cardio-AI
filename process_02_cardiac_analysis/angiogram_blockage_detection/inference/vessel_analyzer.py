import numpy as np
from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes
from scipy.ndimage import distance_transform_edt
from scipy.signal import savgol_filter
from skan import Skeleton, summarize


class VesselAnalyzer:
    """
    Geometry-based vessel analyzer for stenosis localization.
    Updated to use segment-based narrowing detection instead of
    only isolated local-minimum detection.
    """

    def __init__(
        self,
        mask_pil,
        min_object_size=500,
        min_hole_size=150,
        min_branch_length=50,
        end_ignore_pixels=12,
        local_window=10,
        inner_gap=4,
        min_stenosis_percent=15.0,
        duplicate_distance=30,
        min_lesion_length_pixels=3,
        border_margin=15,
        lesion_threshold_ratio=0.90,
        min_branch_avg_diameter=2.0
    ):
        if hasattr(mask_pil, "convert"):
            mask_np = np.array(mask_pil.convert("L"))
        elif isinstance(mask_pil, np.ndarray):
            mask_np = mask_pil.copy()
        else:
            mask_np = np.array(mask_pil)

        binary = mask_np > 127
        binary = remove_small_objects(binary, min_size=min_object_size)
        binary = remove_small_holes(binary, area_threshold=min_hole_size)

        self.binary_mask = binary.astype(np.uint8)
        self.skeleton = None
        self.distance_map = None

        self.img_h, self.img_w = self.binary_mask.shape

        self.min_branch_length = min_branch_length
        self.end_ignore_pixels = end_ignore_pixels
        self.local_window = local_window
        self.inner_gap = inner_gap
        self.min_stenosis_percent = min_stenosis_percent
        self.duplicate_distance = duplicate_distance
        self.min_lesion_length_pixels = min_lesion_length_pixels
        self.border_margin = border_margin
        self.lesion_threshold_ratio = lesion_threshold_ratio
        self.min_branch_avg_diameter = min_branch_avg_diameter

    def get_vessel_geometry(self):
        self.skeleton = skeletonize(self.binary_mask).astype(bool)
        self.distance_map = distance_transform_edt(self.binary_mask)
        return self.skeleton, self.distance_map

    def _smooth_profile(self, diameters):
        n = len(diameters)

        if n < 5:
            return diameters.copy()

        win = min(9, n if n % 2 == 1 else n - 1)
        if win >= 5:
            try:
                return savgol_filter(diameters, window_length=win, polyorder=2, mode="interp")
            except Exception:
                pass

        return np.convolve(diameters, np.ones(5) / 5, mode="same")

    def _classify_stenosis(self, percent):
        if percent < 50:
            return "Mild"
        elif percent < 70:
            return "Moderate"
        return "Severe"

    def _compute_local_reference(self, profile, idx):
        left_start = max(0, idx - self.local_window)
        left_end = max(0, idx - self.inner_gap)
        right_start = min(len(profile), idx + self.inner_gap + 1)
        right_end = min(len(profile), idx + self.local_window + 1)

        left_ref = profile[left_start:left_end]
        right_ref = profile[right_start:right_end]

        ref_candidates = np.concatenate([left_ref, right_ref]) if len(left_ref) > 0 and len(right_ref) > 0 else profile
        if len(ref_candidates) == 0:
            return None

        d_ref = float(np.median(ref_candidates))

        if d_ref <= 0:
            return None

        return d_ref

    def _compute_confidence(self, sten_pct, lesion_length, d_ref, d_min):
        severity_score = min(sten_pct / 100.0, 1.0)
        length_score = min(lesion_length / 15.0, 1.0)
        drop = max(d_ref - d_min, 0.0)
        drop_score = min(drop / max(d_ref, 1e-6), 1.0)

        confidence = 0.5 * severity_score + 0.3 * length_score + 0.2 * drop_score
        return round(float(confidence), 3)

    def _is_duplicate(self, x, y, blockages):
        for b in blockages:
            bx, by = b["coords"]
            if np.linalg.norm(np.array([x, y]) - np.array([bx, by])) < self.duplicate_distance:
                return True
        return False

    def _is_near_border(self, x, y):
        return (
            x < self.border_margin or
            y < self.border_margin or
            x > (self.img_w - self.border_margin) or
            y > (self.img_h - self.border_margin)
        )

    def _expand_lesion_segment(self, profile, center_idx, d_ref):
        threshold = self.lesion_threshold_ratio * d_ref

        left = center_idx
        while left - 1 >= 0 and profile[left - 1] < threshold:
            left -= 1

        right = center_idx
        while right + 1 < len(profile) and profile[right + 1] < threshold:
            right += 1

        return left, right

    def _find_best_segment_center(self, profile, left, right):
        segment = profile[left:right + 1]
        rel_idx = int(np.argmin(segment))
        return left + rel_idx

    def calculate_stenosis(self):
        if self.skeleton is None or np.sum(self.skeleton) == 0:
            return []

        skel_obj = Skeleton(self.skeleton)
        summary = summarize(skel_obj)

        if len(summary) == 0:
            return []

        major_branches = summary[summary["branch-distance"] > self.min_branch_length]

        blockages = []

        for branch_id, row in major_branches.iterrows():
            path_coords = skel_obj.path_coordinates(branch_id).astype(int)

            if len(path_coords) < (2 * self.end_ignore_pixels + 3):
                continue

            radii = self.distance_map[path_coords[:, 0], path_coords[:, 1]]
            diameters = radii * 2.0
            smooth_diameters = self._smooth_profile(diameters)

            if np.mean(smooth_diameters) < self.min_branch_avg_diameter:
                continue

            start_idx = self.end_ignore_pixels
            end_idx = len(smooth_diameters) - self.end_ignore_pixels

            if end_idx <= start_idx:
                continue

            i = start_idx
            while i < end_idx:
                d_here = float(smooth_diameters[i])

                d_ref = self._compute_local_reference(smooth_diameters, i)
                if d_ref is None:
                    i += 1
                    continue

                sten_pct = (1.0 - (d_here / d_ref)) * 100.0

                if sten_pct < self.min_stenosis_percent or sten_pct > 98:
                    i += 1
                    continue

                left_idx, right_idx = self._expand_lesion_segment(smooth_diameters, i, d_ref)
                lesion_length = right_idx - left_idx + 1

                if lesion_length < self.min_lesion_length_pixels:
                    i += 1
                    continue

                center_idx = self._find_best_segment_center(smooth_diameters, left_idx, right_idx)
                d_min = float(smooth_diameters[center_idx])

                d_ref_center = self._compute_local_reference(smooth_diameters, center_idx)
                if d_ref_center is None:
                    i = right_idx + 1
                    continue

                sten_pct_center = (1.0 - (d_min / d_ref_center)) * 100.0

                if sten_pct_center < self.min_stenosis_percent or sten_pct_center > 98:
                    i = right_idx + 1
                    continue

                y, x = path_coords[center_idx]

                if self._is_near_border(int(x), int(y)):
                    i = right_idx + 1
                    continue

                if self._is_duplicate(x, y, blockages):
                    i = right_idx + 1
                    continue

                lesion_radius = max(int(round(max(d_ref_center, d_min) / 2.0)), 8)
                confidence = self._compute_confidence(
                    sten_pct_center, lesion_length, d_ref_center, d_min
                )

                blockages.append({
                    "coords": (int(x), int(y)),
                    "percentage": round(float(sten_pct_center), 1),
                    "severity": self._classify_stenosis(sten_pct_center),
                    "branch_id": int(branch_id),
                    "d_min": round(d_min, 2),
                    "d_ref": round(d_ref_center, 2),
                    "lesion_length": int(lesion_length),
                    "lesion_radius": int(lesion_radius),
                    "confidence": confidence,
                    "segment_start_idx": int(left_idx),
                    "segment_end_idx": int(right_idx)
                })

                i = right_idx + 1

        blockages.sort(key=lambda b: (b["percentage"], b["confidence"], b["lesion_length"]), reverse=True)
        return blockages