import numpy as np
import cv2
import torch
import torchvision.transforms as T
from PIL import Image, ImageOps
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

def make_mask(img, remove_size=2000):
    """
    Cleans the model output to create a crisp binary mask[cite: 109, 110].
    """
    image = np.array(img.convert('L'))
    # Thresholding to separate vessel from background [cite: 109, 110]
    thresh = filters.threshold_yen(image)
    seg = (image >= thresh).astype('bool')
    # Remove small noise particles [cite: 111]
    seg = morphology.remove_small_objects(seg, remove_size)
    return (seg * 255).astype('uint8')

def fusion_predict(model, x, device='cpu'):
    """
    Passes the image through the network to get the vessel prediction[cite: 100, 105].
    """
    model.to(device)
    with torch.no_grad():
        # The 'model' method in your UNet class handles the forward pass [cite: 61, 62]
        pred = model.model(x)
    return pred