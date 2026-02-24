import numpy as np
import cv2
import torch
import torchvision.transforms as T
from PIL import Image, ImageChops, ImageOps
from skimage import filters, morphology


def apply_tophat(img, fsize=50):
    """
    Removes background noise (ribs, spine) to highlight vessels[cite: 2, 3].
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img).astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (fsize, fsize))
    wth = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel).astype(np.float32)
    bth = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
    dst = (img + wth - bth).clip(0, 255).astype(np.uint8)
    return Image.fromarray(dst)


def make_mask(img, remove_size=2000, local_kernel=21, hole_max_size=100):
    """
    Advanced morphological cleaning based on DeepSA logic[cite: 109, 110, 111].
    Uses Yen thresholding and small object removal to ensure clinical accuracy.
    """
    image = np.array(img.convert('L'))

    # 1. Primary Thresholding (Yen's method) [cite: 109, 110]
    thresh = filters.threshold_yen(image)
    seg1 = (image >= thresh).astype('bool')

    # 2. Local Thresholding to catch finer details
    thresh_local = filters.threshold_local(image, local_kernel)
    seg2 = (image >= thresh_local).astype('bool')

    # 3. Intersection of thresholds to isolate significant vessels
    inter = (seg1 & seg2).astype('uint8') * 255

    # 4. Final Cleanup: Morphological closing and noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dst = cv2.morphologyEx(inter, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Remove small noise 'islands'
    cleaned = morphology.remove_small_objects(dst.astype(bool), remove_size)

    # Convert to image for further processing
    return Image.fromarray((cleaned * 255).astype('uint8'))


def clear_mask(mask, area_max=3000):
    """
    Removes specific non-vessel objects based on area/perimeter ratios.
    This prevents the 'messy' artifacts seen in noisy backgrounds.
    """
    sp = np.array(mask)
    conts, _ = cv2.findContours(sp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cont in conts:
        area = cv2.contourArea(cont)
        perimeter = cv2.arcLength(cont, True)
        # Clinical ratio to distinguish between vessel segments and noise
        ratio = area / (perimeter + 1e-5)
        if ratio > 2.5 and area < area_max:
            cv2.fillPoly(sp, [cont], (0, 0, 0))
    return Image.fromarray(sp)


def fusion_predict(model, x, device='cpu'):
    """
    Standard model forward pass to get raw vessel prediction[cite: 61, 62].
    """
    model.to(device)
    with torch.no_grad():
        pred = model.model(x)
    return pred


def get_skeleton(mask_img):
    """
    Converts a clinical binary mask into a 1-pixel wide centerline (skeleton).
    This creates the 'State Space' for search algorithms.
    """
    mask_array = np.array(mask_img) > 0
    skeleton = morphology.skeletonize(mask_array)
    return (skeleton * 255).astype(np.uint8)