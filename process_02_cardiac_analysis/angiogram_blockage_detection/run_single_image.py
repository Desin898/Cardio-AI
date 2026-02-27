import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageChops
import cv2

from models import UNet
from datasets import tophat
from utils import fusion_predict, make_mask, clear_mask


# ----------------------------------------------------------
#                CONFIGURATION
# ----------------------------------------------------------

SIZE = 512
CKPT = "ckpt/fscad_36249.ckpt"      # Path to the DeepSA pre-trained weights
INPUT_IMAGE = "example7.png"       # Target angiogram frame for processing
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------
#            PREPROCESSING TRANSFORMS
# ----------------------------------------------------------

# Transform set 1: Includes morphological Top-Hat enhancement for fine detail [cite: 80, 81]
tfmc1 = T.Compose([
    T.Resize(SIZE),
    T.Lambda(lambda img: tophat(img, 50)),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

# Transform set 2: Standard resizing for broader structural extraction [cite: 81]
tfmc2 = T.Compose([
    T.Resize(SIZE),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])


# ----------------------------------------------------------
#                LOAD MODEL
# ----------------------------------------------------------

print("\nInitializing DeepSA Inference Engine...")

# Initialize UNet with 1 input channel (grayscale) and 1 output channel (mask) [cite: 81]
netE = UNet(1, 1, 32, bilinear=True)
checkpoint = torch.load(CKPT, map_location=DEVICE)

# Standardize state dict keys by removing multi-GPU 'module.' prefixes [cite: 81, 82]
new_state = {k.replace("module.", ""): v for k, v in checkpoint["netE"].items()}
netE.load_state_dict(new_state)

netE.to(DEVICE)
netE.eval()

print("Model weights loaded successfully.\n")


# ----------------------------------------------------------
#                LOAD INPUT IMAGE
# ----------------------------------------------------------

print(f"Opening source image: {INPUT_IMAGE}")
# Convert to single-channel 'L' (Grayscale) mode [cite: 82]
img = Image.open(INPUT_IMAGE).convert("L")


# ----------------------------------------------------------
#            PROCESS THROUGH DeepSA PIPELINE
# ----------------------------------------------------------

x1 = tfmc1(img)
x2 = tfmc2(img)

print("Executing dual-scale fusion inference...")

# Prediction 1: Enhanced via Top-Hat transform [cite: 82]
_, out1 = fusion_predict(
    netE, ["none"], x1,
    multiangle=False, denoise=4, size=SIZE, cutoff=0.4, pad=0, netE=True
)

# Prediction 2: Raw image scale
_, out2 = fusion_predict(
    netE, ["none"], x2,
    multiangle=False, denoise=4, size=SIZE, cutoff=0.4, pad=0, netE=True
)


# ----------------------------------------------------------
#            MERGE TWO OUTPUT SCALES
# ----------------------------------------------------------

# Combine both predictions using Maximum Intensity Projection (MIP)
merged = Image.fromarray(
    np.expand_dims(
        np.max(np.concatenate((np.array(out1), np.array(out2)), axis=2), axis=2),
        2
    ).repeat(3, axis=2)
)


# ----------------------------------------------------------
#                SEGMENTATION MASK
# ----------------------------------------------------------

print("Generating binary vessel mask...")

# Apply morphological operations to extract clean vessel boundaries
seg_mask = make_mask(
    merged, remove_size=2000, local_kernel=21, hole_max_size=100
)

# Clean mask to remove isolated artifacts or non-vascular noise
segmented = clear_mask(seg_mask)


# ----------------------------------------------------------
#                FINAL SUBTRACTION IMAGE
# ----------------------------------------------------------

print("Rendering enhanced subtraction visualization...")

# Produce high-contrast output for visual clinical inspection
sub_img = ImageChops.invert(merged)
sub_img = T.ToTensor()(sub_img)
sub_img = sub_img * (2 ** -0.5)        # Reduce overall intensity for clarity
sub_img = T.ToPILImage()(sub_img)


# ----------------------------------------------------------
#                SAVE RESULTS
# ----------------------------------------------------------

sub_img.save("output_subtraction.png")
segmented.save("output_segmentation.png")

print("\n--------------------------------")
print("INFERENCE COMPLETE!")
print("Resulting artifacts:")
print(" - output_subtraction.png (Enhanced arteries)")
print(" - output_segmentation.png (Binary vessel map)")
print("--------------------------------\n")