import gradio as gr
import torch
from models import UNet
from datasets import tophat
import torchvision.transforms as T
from torchvision.utils import make_grid
import numpy as np
from PIL import Image, ImageChops, ImageOps
from utils import fusion_predict, make_mask, clear_mask
from pathlib import Path
import cv2
from skimage import morphology

# Configuration constants
SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pre-processing transforms: Scale 1 includes Top-Hat enhancement [cite: 21, 80, 81]
tfmc1 = T.Compose([
    T.Resize(SIZE),
    T.Lambda(lambda img: tophat(img, 50)),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

# Pre-processing transforms: Scale 2 is standard resizing [cite: 21, 22]
tfmc2 = T.Compose([
    T.Resize(SIZE),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

# Path to the pre-trained model checkpoint [cite: 22, 80]
ckpts = ['ckpt/fscad_36249.ckpt']

# Initialize and load the UNet model [cite: 22]
netE = UNet(1, 1, 32, bilinear=True)
checkpoint = torch.load(ckpts[0], map_location=DEVICE)
# Remove 'module.' prefix if the checkpoint was saved from a DataParallel model [cite: 22, 81, 82]
new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['netE'].items()}
netE.load_state_dict(new_state_dict)
netE.to(DEVICE)
netE.eval()


def predict(img, auto_tresh, options):
    """
    Main prediction pipeline for Deep Subtraction Angiography.
    Processes the input image to generate an enhanced subtraction image and a binary vessel mask. [cite: 23, 28, 29]
    """
    if auto_tresh:
        # Determine options based on UI checkboxes [cite: 22, 23]
        multiangle = True if "Multiangle" in options else False
        pad = 50 if "Pad margin" in options else 0

        img = img.convert('L')
        x1 = tfmc1(img)
        x2 = tfmc2(img)

        # Dual-scale fusion prediction [cite: 24]
        _, out1 = fusion_predict(netE, ["none"], x1, multiangle=multiangle, denoise=4, size=SIZE, cutoff=0.4, pad=pad,
                                 netE=True)
        _, out2 = fusion_predict(netE, ["none"], x2, multiangle=False, denoise=4, size=SIZE, cutoff=0.4, pad=pad,
                                 netE=True)

        # Merge the two scales using maximum intensity projection [cite: 24]
        out_merge = Image.fromarray(
            np.expand_dims(np.max(np.concatenate((np.array(out1), np.array(out2)), axis=2), axis=2), 2).repeat(3, 2))

        # Generate the vessel segmentation mask [cite: 24, 25]
        mask_merge = make_mask(out_merge, remove_size=2000, local_kernel=21, hole_max_size=100)
        out_merge = T.functional.adjust_gamma(out_merge, 2)
        seg_img = clear_mask(mask_merge)

        # Create the deep subtraction visual output [cite: 25]
        sub_img = ImageChops.invert(out_merge)
        sub_img = T.ToTensor()(sub_img)
        sub_img = sub_img * (2 ** (-0.5))  # Adjust intensity [cite: 25]
        sub_img = T.ToPILImage()(sub_img)
    else:
        # Simplified inference mode [cite: 25, 26]
        img = img.convert('L')
        x = tfmc1(img)
        input = x.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred_y = netE(input)

        # Process the subtraction image output [cite: 26]
        sub_img = make_grid(pred_y, normalize=True)
        sub_img = (sub_img.permute(1, 2, 0).detach().cpu().numpy() * 255).astype('uint8')
        sub_img = cv2.fastNlMeansDenoising(sub_img, None, 3, 7, 21)  # Denoising [cite: 26, 27]

        sub_img = T.ToPILImage()(sub_img)
        sub_img = ImageOps.autocontrast(sub_img, cutoff=1)
        sub_img = T.functional.adjust_gamma(sub_img, 2)

        sub_img = ImageChops.invert(sub_img)
        sub_img = T.ToTensor()(sub_img)
        sub_img = sub_img * (2 ** (-0.5))
        sub_img = T.ToPILImage()(sub_img)

        # Process the binary segmentation mask [cite: 28]
        seg_img = torch.sign(pred_y)
        seg_img = ((seg_img.cpu().detach() + 1) / 2).numpy().astype(bool)
        seg_img = morphology.remove_small_objects(seg_img, 500)  # Remove noise [cite: 28]
        seg_img = (seg_img * 255).astype('uint8')
        seg_img = torch.from_numpy(seg_img / 255)
        seg_img = T.ToPILImage()(seg_img[0])

    return sub_img, seg_img


# Gradio Interface Setup
title = "DeepSA: Deep Subtraction Angiography"
description = "Vessel Enhancement and Segmentation Model for Coronary Angiograms."
article = "<p style='text-align: center'>Re-implemented for DSGP Group 3 Project</p>"

# Load example images if available in the directory
examples = list(Path("data/frames").glob("*.png"))
examples = [[str(e)] for e in examples]

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Input Angiogram"),
        gr.Checkbox(value=True, label="Auto-Thresholding"),
        gr.CheckboxGroup(["Multiangle", "Pad margin"], label="Advanced Options")
    ],
    outputs=[
        gr.Image(type="pil", label='Enhanced Subtraction'),
        gr.Image(type="pil", label='Vessel Segmentation')
    ],
    title=title,
    description=description,
    article=article,
    examples=examples
)

if __name__ == "__main__":
    # Launch the local server
    demo.launch(server_name='0.0.0.0', share=False)