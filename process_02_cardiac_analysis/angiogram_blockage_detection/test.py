import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageChops, ImageOps
import cv2

from models import UNet
from datasets import tophat
from utils import fusion_predict, make_mask, clear_mask

# ----------------------------------------------------------
#                CONFIGURATION
# ----------------------------------------------------------

SIZE = 512
CKPT = "ckpt/fscad_36249.ckpt"          # Path to DeepSA vessel extraction model weights
INPUT_IMAGE = "example2.png"           # Target image for processing
DEVICE = "cpu"                         # Set to 'cuda' if GPU is available

# ----------------------------------------------------------
#            PREPROCESSING TRANSFORMS
# ----------------------------------------------------------

# Transform 1: Resizing and Top-Hat morphological enhancement [cite: 87]
tfmc1 = T.Compose([
    T.Resize(SIZE),
    T.Lambda(lambda img: tophat(img, 50)),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

# Transform 2: Standard resizing without morphological enhancement [cite: 87]
tfmc2 = T.Compose([
    T.Resize(SIZE),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

# ----------------------------------------------------------
#                LOAD MODEL
# ----------------------------------------------------------

print("\nLoading DeepSA Model...")

netE = UNet(1, 1, 32, bilinear=True)
checkpoint = torch.load(CKPT, map_location=DEVICE)

# Clean state dict keys by removing the "module." prefix (used in DataParallel training) [cite: 88]
new_dict = {k.replace("module.", ""): v for k, v in checkpoint["netE"].items()}
netE.load_state_dict(new_dict)
netE.to(DEVICE)
netE.eval()

print("Model loaded successfully.\n")

# ----------------------------------------------------------
#            LOAD INPUT IMAGE
# ----------------------------------------------------------

print(f"Loading input image: {INPUT_IMAGE}")

# Convert input to grayscale (L mode) for the UNet encoder [cite: 88]
img = Image.open(INPUT_IMAGE).convert("L")

# ----------------------------------------------------------
#            PROCESS THROUGH DeepSA PIPELINE
# ----------------------------------------------------------

x1 = tfmc1(img)
x2 = tfmc2(img)

print("Running DeepSA fusion prediction...")

# Generate two scale outputs using fusion_predict
_, out1 = fusion_predict(
    netE, ["none"], x1, multiangle=False, denoise=4,
    size=SIZE, cutoff=0.4, pad=0, netE=True
)

_, out2 = fusion_predict(
    netE, ["none"], x2, multiangle=False, denoise=4,
    size=SIZE, cutoff=0.4, pad=0, netE=True
)

# Merge multi-scale outputs using Maximum Intensity Projection (MIP) logic
merged = Image.fromarray(
    np.expand_dims(
        np.max(np.concatenate((np.array(out1), np.array(out2)), axis=2), axis=2),
        2
    ).repeat(3, axis=2)
)

# ----------------------------------------------------------
#                MASK & SEGMENTATION
# ----------------------------------------------------------

print("Extracting segmentation mask...")

# Create binary mask and remove small noise objects to isolate vessels
seg_mask = make_mask(merged, remove_size=2000, local_kernel=21, hole_max_size=100)
segmented = clear_mask(seg_mask)

# ----------------------------------------------------------
#                CREATE SUBTRACTION IMAGE
# ----------------------------------------------------------

print("Creating subtraction angiography output...")

# Invert image and reduce intensity to highlight arterial structures
sub_img = ImageChops.invert(merged)
sub_img = T.ToTensor()(sub_img)
sub_img = sub_img * (2 ** (-0.5))  # Constant intensity reduction
sub_img = T.ToPILImage()(sub_img)

# ----------------------------------------------------------
#                SAVE OUTPUTS
# ----------------------------------------------------------

sub_img.save("output_subtraction.png")
segmented.save("output_segmentation.png")

print("\nDONE!")
print("Saved:")
print(" - output_subtraction.png  (artery enhanced)")
print(" - output_segmentation.png (binary vessel mask)\n")