import numpy as np
import cv2
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import distance_transform_edt
from skan import Skeleton, summarize


class VesselAnalyzer:
    def __init__(self, mask_pil):
        mask_np = np.array(mask_pil.convert('L'))
        # 1. CLEAN THE MASK FIRST: Remove tiny disconnected noise bits
        cleaned_mask = remove_small_objects((mask_np > 127), min_size=500)
        self.binary_mask = cleaned_mask.astype(np.uint8)
        self.skeleton = None
        self.distance_map = None

    def get_vessel_geometry(self):
        self.skeleton = skeletonize(self.binary_mask).astype(bool)
        self.distance_map = distance_transform_edt(self.binary_mask)
        return self.skeleton, self.distance_map

    def calculate_stenosis(self):
        if self.skeleton is None or np.sum(self.skeleton) == 0:
            return []

        skel_obj = Skeleton(self.skeleton)
        summary = summarize(skel_obj)

        # 2. FILTER BRANCHES: Only look at major segments (>40 pixels)
        # This stops the 'False Positive' circles on tiny side-vessels
        major_branches = summary[summary['branch-distance'] > 40]

        blockages = []
        for index, row in major_branches.iterrows():
            path_coords = skel_obj.path_coordinates(index).astype(int)
            radii = self.distance_map[path_coords[:, 0], path_coords[:, 1]]
            diameters = radii * 2

            # 3. SMOOTH THE DIAMETER CURVE: Removes 'jaggies'
            smooth_diameters = np.convolve(diameters, np.ones(5) / 5, mode='same')

            # 4. SLIDING REFERENCE: Compare narrow part to healthy part nearby
            for i in range(10, len(smooth_diameters) - 10):
                d_min = smooth_diameters[i]
                # Compare to a local reference (average of 10 pixels before/after)
                d_ref = np.mean(np.concatenate([smooth_diameters[i - 10:i - 5], smooth_diameters[i + 5:i + 10]]))

                if d_ref > 0:
                    sten_pct = (1 - (d_min / d_ref)) * 100

                    # Only flag if it's a real 'dip' and significant (>50%)
                    if sten_pct > 50:
                        # Ensure we don't pick 20 points for the same blockage
                        is_duplicate = any(np.linalg.norm(
                            np.array([path_coords[i][1], path_coords[i][0]]) - np.array(b['coords'])) < 30 for b in
                                           blockages)
                        if not is_duplicate:
                            blockages.append({
                                'coords': (path_coords[i][1], path_coords[i][0]),
                                'percentage': round(sten_pct, 1)
                            })
        return blockages