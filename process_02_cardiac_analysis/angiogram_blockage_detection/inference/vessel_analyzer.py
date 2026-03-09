import numpy as np
from skimage.morphology import skeletonize, remove_small_objects, remove_small_holes
from scipy.ndimage import distance_transform_edt
from scipy.signal import savgol_filter
from skan import Skeleton, summarize


class VesselAnalyzer:
    """
    Geometry-based vessel analyzer for stenosis localization.
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
        min_stenosis_percent=30.0,
        duplicate_distance=30,
        lesion_half_window=3,
        min_lesion_length_pixels=5
    ):
        mask_np = np.array(mask_pil.convert("L"))

        # Binary threshold
        binary = mask_np > 127

        # Clean small noise
        binary = remove_small_objects(binary, min_size=min_object_size)

        # Fill small holes in vessel lumen region
        binary = remove_small_holes(binary, area_threshold=min_hole_size)

        self.binary_mask = binary.astype(np.uint8)
        self.skeleton = None
        self.distance_map = None

        # Parameters
        self.min_branch_length = min_branch_length
        self.end_ignore_pixels = end_ignore_pixels
        self.local_window = local_window
        self.inner_gap = inner_gap
        self.min_stenosis_percent = min_stenosis_percent
        self.duplicate_distance = duplicate_distance
        self.lesion_half_window = lesion_half_window
        self.min_lesion_length_pixels = min_lesion_length_pixels

    def get_vessel_geometry(self):
        """
        Generates vessel skeleton and distance map.
        """
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
        if percent < 40:
            return "Mild"
        elif percent < 70:
            return "Moderate"
        return "Severe"

    def _is_local_minimum(self, profile, idx):
        if idx <= 0 or idx >= len(profile) - 1:
            return False
        return profile[idx] < profile[idx - 1] and profile[idx] <= profile[idx + 1]

    def _compute_local_reference(self, profile, idx):
        """
        Compute local reference diameter using nearby proximal/distal windows.
        """
        left_start = idx - self.local_window
        left_end = idx - self.inner_gap
        right_start = idx + self.inner_gap + 1
        right_end = idx + self.local_window + 1

        if left_start < 0 or right_end > len(profile):
            return None

        left_ref = profile[left_start:left_end]
        right_ref = profile[right_start:right_end]

        if len(left_ref) == 0 or len(right_ref) == 0:
            return None

        ref_candidates = np.concatenate([left_ref, right_ref])
        d_ref = float(np.median(ref_candidates))

        if d_ref <= 0:
            return None

        return d_ref

    def _estimate_lesion_length(self, profile, idx, d_ref):
        """
        Estimate how many nearby pixels stay below 90% of reference diameter.
        """
        threshold = 0.9 * d_ref
        count = 1

        # expand left
        j = idx - 1
        while j >= 0 and profile[j] < threshold:
            count += 1
            j -= 1

        # expand right
        j = idx + 1
        while j < len(profile) and profile[j] < threshold:
            count += 1
            j += 1

        return count

    def _compute_confidence(self, sten_pct, lesion_length, d_ref, d_min):
        """
        Simple confidence score from 0 to 1.
        """
        severity_score = min(sten_pct / 100.0, 1.0)
        length_score = min(lesion_length / 12.0, 1.0)

        # Bigger actual drop -> better confidence
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

            if len(path_coords) < (2 * self.end_ignore_pixels + 5):
                continue

            radii = self.distance_map[path_coords[:, 0], path_coords[:, 1]]
            diameters = radii * 2.0
            smooth_diameters = self._smooth_profile(diameters)

            start_idx = self.end_ignore_pixels
            end_idx = len(smooth_diameters) - self.end_ignore_pixels

            if end_idx <= start_idx:
                continue

            for i in range(start_idx, end_idx):
                d_min = float(smooth_diameters[i])

                if not self._is_local_minimum(smooth_diameters, i):
                    continue

                d_ref = self._compute_local_reference(smooth_diameters, i)
                if d_ref is None:
                    continue

                sten_pct = (1.0 - (d_min / d_ref)) * 100.0

                if sten_pct < self.min_stenosis_percent or sten_pct > 95:
                    continue

                lesion_length = self._estimate_lesion_length(smooth_diameters, i, d_ref)
                if lesion_length < self.min_lesion_length_pixels:
                    continue

                y, x = path_coords[i]

                if self._is_duplicate(x, y, blockages):
                    continue

                lesion_radius = max(int(round(d_ref / 2.0)), 6)
                confidence = self._compute_confidence(sten_pct, lesion_length, d_ref, d_min)

                blockages.append({
                    "coords": (int(x), int(y)),
                    "percentage": round(float(sten_pct), 1),
                    "severity": self._classify_stenosis(sten_pct),
                    "branch_id": int(branch_id),
                    "d_min": round(d_min, 2),
                    "d_ref": round(d_ref, 2),
                    "lesion_length": int(lesion_length),
                    "lesion_radius": int(lesion_radius),
                    "confidence": confidence
                })

        blockages.sort(key=lambda b: (b["percentage"], b["confidence"]), reverse=True)
        return blockages