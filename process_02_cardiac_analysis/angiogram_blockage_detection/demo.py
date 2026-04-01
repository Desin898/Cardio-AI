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
from inference.vessel_analyzer import VesselAnalyzer

# Configuration constants
SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pre-processing transforms
tfmc1 = T.Compose([
    T.Resize(SIZE),
    T.Lambda(lambda img: tophat(img, 50)),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

tfmc2 = T.Compose([
    T.Resize(SIZE),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

# Path to the pre-trained model checkpoint
ckpts = ["ckpt/fscad_36249.ckpt"]

# Initialize and load the UNet model
netE = UNet(1, 1, 32, bilinear=True)
checkpoint = torch.load(ckpts[0], map_location=DEVICE)
new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['netE'].items()}
netE.load_state_dict(new_state_dict)
netE.to(DEVICE)
netE.eval()


def get_severity_color(severity):
    """
    Return RGB color for severity class.
    """
    if severity == "Severe":
        return (255, 0, 0)      # Red
    elif severity == "Moderate":
        return (255, 255, 0)    # Yellow
    return (0, 255, 255)        # Cyan for Mild


def draw_info_box(image, text_lines, start_x=10, start_y=10):
    """
    Draw a small summary panel on the image.
    """
    overlay = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    thickness = 1
    line_height = 18

    max_width = 0
    for line in text_lines:
        (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_width = max(max_width, tw)

    box_w = max_width + 14
    box_h = len(text_lines) * line_height + 10

    cv2.rectangle(
        overlay,
        (start_x, start_y),
        (start_x + box_w, start_y + box_h),
        (0, 0, 0),
        thickness=-1
    )

    y = start_y + 16
    for line in text_lines:
        cv2.putText(
            overlay,
            line,
            (start_x + 6, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )
        y += line_height

    return overlay


def draw_stenosis_overlay(sub_img, detected_blockages, top_k=3):
    """
    Draw stenosis localization overlay on subtraction image.
    """
    overlay = np.array(sub_img.convert("RGB")).copy()

    # Keep strongest lesions only
    detected_blockages = detected_blockages[:top_k]

    summary_lines = [f"Detected lesions: {len(detected_blockages)}"]

    for idx, blockage in enumerate(detected_blockages, start=1):
        x, y = blockage["coords"]
        percentage = blockage["percentage"]
        severity = blockage.get("severity", "Unknown")
        confidence = blockage.get("confidence", 0.0)
        lesion_radius = blockage.get("lesion_radius", 8)

        color = get_severity_color(severity)

        # Draw lesion circle
        cv2.circle(overlay, (x, y), lesion_radius, color, 2)

        # Draw center point
        cv2.circle(overlay, (x, y), 2, color, -1)

        # Label text
        label = f"{percentage}% {severity} | C:{confidence:.2f}"

        # Keep text inside image
        text_x = min(x + 10, overlay.shape[1] - 180)
        text_y = max(y - 10, 20)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

        # Background box
        cv2.rectangle(
            overlay,
            (text_x - 2, text_y - th - 4),
            (text_x + tw + 2, text_y + 2),
            (0, 0, 0),
            thickness=-1
        )

        cv2.putText(
            overlay,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA
        )

        summary_lines.append(f"{idx}. {percentage}% {severity}")

    overlay = draw_info_box(overlay, summary_lines, start_x=10, start_y=10)
    return Image.fromarray(overlay)


def predict(img, auto_tresh, options):
    """
    Main prediction pipeline for Deep Subtraction Angiography.
    """
    if img is None:
        return None, None, None

    if auto_tresh:
        multiangle = "Multiangle" in options
        pad = 50 if "Pad margin" in options else 0

        img_L = img.convert("L")
        x1 = tfmc1(img_L)
        x2 = tfmc2(img_L)

        # Dual-scale fusion prediction
        _, out1 = fusion_predict(
            netE, ["none"], x1,
            multiangle=multiangle,
            denoise=4,
            size=SIZE,
            cutoff=0.4,
            pad=pad,
            netE=True
        )

        _, out2 = fusion_predict(
            netE, ["none"], x2,
            multiangle=False,
            denoise=4,
            size=SIZE,
            cutoff=0.4,
            pad=pad,
            netE=True
        )

        # Merge the two scales
        out_merge = Image.fromarray(
            np.expand_dims(
                np.max(np.concatenate((np.array(out1), np.array(out2)), axis=2), axis=2),
                2
            ).repeat(3, 2)
        )

        # Vessel segmentation mask
        mask_merge = make_mask(out_merge, remove_size=2000, local_kernel=21, hole_max_size=100)
        out_merge_gamma = T.functional.adjust_gamma(out_merge, 2)
        seg_img = clear_mask(mask_merge)

        # Deep subtraction output
        sub_img = ImageChops.invert(out_merge_gamma)
        sub_img = T.ToTensor()(sub_img)
        sub_img = sub_img * (2 ** (-0.5))
        sub_img = T.ToPILImage()(sub_img)

    else:
        # Simplified inference mode
        img_L = img.convert("L")
        x = tfmc1(img_L)
        input_tensor = x.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred_y = netE(input_tensor)

        # Subtraction image
        sub_img = make_grid(pred_y, normalize=True)
        sub_img = (sub_img.permute(1, 2, 0).detach().cpu().numpy() * 255).astype("uint8")
        sub_img = cv2.fastNlMeansDenoising(sub_img, None, 3, 7, 21)

        sub_img = T.ToPILImage()(sub_img)
        sub_img = ImageOps.autocontrast(sub_img, cutoff=1)
        sub_img = T.functional.adjust_gamma(sub_img, 2)

        sub_img = ImageChops.invert(sub_img)
        sub_img = T.ToTensor()(sub_img)
        sub_img = sub_img * (2 ** (-0.5))
        sub_img = T.ToPILImage()(sub_img)

        # Binary segmentation mask
        seg_img_tensor = torch.sign(pred_y)
        seg_img_np = ((seg_img_tensor.cpu().detach() + 1) / 2).numpy().astype(bool)
        seg_img_np = morphology.remove_small_objects(seg_img_np, 500)
        seg_img_np = (seg_img_np * 255).astype("uint8")
        seg_img = T.ToPILImage()(seg_img_np[0])

    # --- STENOSIS ANALYSIS ---
    analyzer = VesselAnalyzer(seg_img)
    analyzer.get_vessel_geometry()
    detected_blockages = analyzer.calculate_stenosis()

    # --- OVERLAY VISUALIZATION ---
    if len(detected_blockages) > 0:
        final_overlay = draw_stenosis_overlay(sub_img, detected_blockages, top_k=3)
    else:
        no_lesion_img = np.array(sub_img.convert("RGB")).copy()
        no_lesion_img = draw_info_box(
            no_lesion_img,
            ["Detected lesions: 0", "No significant stenosis candidate found"],
            start_x=10,
            start_y=10
        )
        final_overlay = Image.fromarray(no_lesion_img)

    return sub_img, seg_img, final_overlay


# Gradio Interface Setup
title = "DeepSA: Enhanced Cardiac Analysis"
description = "Subtraction, Segmentation, and Automated Stenosis Pinpointing for Coronary Angiograms."
article = "<p style='text-align: center'>Re-implemented for DSGP Group 3 Project</p>"

# Load example images if available
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
        gr.Image(type="pil", label="1. Enhanced Subtraction"),
        gr.Image(type="pil", label="2. Vessel Segmentation"),
        gr.Image(type="pil", label="3. Stenosis Localization (Blockage %)")
    ],
    title=title,
    description=description,
    article=article,
    examples=examples
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", share=False)