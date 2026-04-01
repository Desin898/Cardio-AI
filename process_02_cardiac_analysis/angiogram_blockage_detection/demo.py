import gradio as gr
import torch
import requests
import io
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

# ============================================================
#  Configuration
# ============================================================
SIZE   = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

ckpts = ["ckpt/fscad_36249.ckpt"]
netE  = UNet(1, 1, 32, bilinear=True)
checkpoint     = torch.load(ckpts[0], map_location=DEVICE)
new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['netE'].items()}
netE.load_state_dict(new_state_dict)
netE.to(DEVICE)
netE.eval()


# ============================================================
#  Helper drawing utilities
# ============================================================

def get_severity_color(severity):
    if severity == "Severe":
        return (255, 0, 0)
    elif severity == "Moderate":
        return (255, 255, 0)
    return (0, 255, 255)


def draw_info_box(image, text_lines, start_x=10, start_y=10):
    overlay     = image.copy()
    font        = cv2.FONT_HERSHEY_SIMPLEX
    font_scale  = 0.42
    thickness   = 1
    line_height = 18

    max_width = 0
    for line in text_lines:
        (tw, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_width   = max(max_width, tw)

    box_w = max_width + 14
    box_h = len(text_lines) * line_height + 10

    cv2.rectangle(overlay, (start_x, start_y),
                  (start_x + box_w, start_y + box_h), (0, 0, 0), -1)

    y = start_y + 16
    for line in text_lines:
        cv2.putText(overlay, line, (start_x + 6, y), font,
                    font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_height

    return overlay


def draw_stenosis_overlay(sub_img, detected_blockages, top_k=3, draw_boxes=True):
    overlay            = np.array(sub_img.convert("RGB")).copy()
    detected_blockages = detected_blockages[:top_k]
    summary_lines      = [f"Detected lesions: {len(detected_blockages)}"]

    for idx, blockage in enumerate(detected_blockages, start=1):
        x, y          = blockage["coords"]
        percentage    = blockage["percentage"]
        severity      = blockage.get("severity", "Unknown")
        confidence    = blockage.get("confidence", 0.0)
        lesion_radius = blockage.get("lesion_radius", 8)
        lesion_length = blockage.get("lesion_length", 0)
        color         = get_severity_color(severity)

        cv2.circle(overlay, (x, y), lesion_radius, color, 2)
        cv2.circle(overlay, (x, y), 3, color, -1)

        if draw_boxes:
            box_r = max(int(lesion_radius * 3), 18)
            x1 = max(0, x - box_r)
            y1 = max(0, y - box_r)
            x2 = min(overlay.shape[1] - 1, x + box_r)
            y2 = min(overlay.shape[0] - 1, y + box_r)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        label  = f"{percentage}% {severity} | C:{confidence:.2f}"

        text_x = x + 10
        text_y = y - 10

        if text_x > overlay.shape[1] - 190:
            text_x = max(10, x - 190)
        if text_y < 20:
            text_y = min(overlay.shape[0] - 10, y + 20)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(overlay,
                      (text_x - 3, text_y - th - 5),
                      (text_x + tw + 3, text_y + 3),
                      (0, 0, 0), -1)
        cv2.putText(overlay, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        summary_lines.append(
            f"{idx}. {percentage}% {severity} | L:{lesion_length} | C:{confidence:.2f}"
        )

    overlay = draw_info_box(overlay, summary_lines, start_x=10, start_y=10)
    return Image.fromarray(overlay)


# ============================================================
#  Image fetch helper
# ============================================================

def load_image_from_url(url: str):
    """Fetch image from Flask server and return as PIL Image."""
    if not url or not url.strip():
        return None
    try:
        resp = requests.get(url.strip(), timeout=20)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        print(f"[DeepSA] Successfully loaded image from: {url}")
        return img
    except Exception as exc:
        print(f"[DeepSA] load_image_from_url failed: {exc}")
        return None


# ============================================================
#  Page-load handler  ← THE CORE FIX
#
#  Gradio calls this automatically on every page load and
#  injects the live HTTP request as `request: gr.Request`.
#  We read ?image= directly in Python — no DOM manipulation,
#  no synthetic events, no timing hacks — and return the PIL
#  Image straight into the input_image widget through
#  Gradio's own data pipeline.
# ============================================================

def on_page_load(request: gr.Request):
    image_url = request.query_params.get("image", "").strip()
    if not image_url:
        print("[DeepSA] No ?image= param on page load.")
        return None
    print(f"[DeepSA] Page load — fetching: {image_url}")
    return load_image_from_url(image_url)


# ============================================================
#  Main prediction pipeline
# ============================================================

def predict(img, auto_tresh, options):
    if img is None:
        return None, None, None

    if auto_tresh:
        multiangle = "Multiangle" in options
        pad        = 50 if "Pad margin" in options else 0

        img_L = img.convert("L")
        x1    = tfmc1(img_L)
        x2    = tfmc2(img_L)

        _, out1 = fusion_predict(netE, ["none"], x1,
                                 multiangle=multiangle, denoise=4,
                                 size=SIZE, cutoff=0.4, pad=pad, netE=True)
        _, out2 = fusion_predict(netE, ["none"], x2,
                                 multiangle=False, denoise=4,
                                 size=SIZE, cutoff=0.4, pad=pad, netE=True)

        out_merge = Image.fromarray(
            np.expand_dims(
                np.max(np.concatenate(
                    (np.array(out1), np.array(out2)), axis=2), axis=2),
                2
            ).repeat(3, 2)
        )

        mask_merge      = make_mask(out_merge, remove_size=2000,
                                    local_kernel=21, hole_max_size=100)
        out_merge_gamma = T.functional.adjust_gamma(out_merge, 2)
        seg_img         = clear_mask(mask_merge)

        sub_img = ImageChops.invert(out_merge_gamma)
        sub_img = T.ToTensor()(sub_img)
        sub_img = sub_img * (2 ** (-0.5))
        sub_img = T.ToPILImage()(sub_img)

    else:
        img_L        = img.convert("L")
        x            = tfmc1(img_L)
        input_tensor = x.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred_y = netE(input_tensor)

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

        seg_img_tensor = torch.sign(pred_y)
        seg_img_np     = ((seg_img_tensor.cpu().detach() + 1) / 2).numpy().astype(bool)
        seg_img_np     = morphology.remove_small_objects(seg_img_np, 500)
        seg_img_np     = (seg_img_np * 255).astype("uint8")
        seg_img        = T.ToPILImage()(seg_img_np[0])

    analyzer           = VesselAnalyzer(seg_img)
    analyzer.get_vessel_geometry()
    detected_blockages = analyzer.calculate_stenosis()

    if len(detected_blockages) > 0:
        final_overlay = draw_stenosis_overlay(sub_img, detected_blockages, top_k=3, draw_boxes=True)
    else:
        no_lesion_img = np.array(sub_img.convert("RGB")).copy()
        no_lesion_img = draw_info_box(
            no_lesion_img,
            [
                "Detected lesions: 0",
                "No strong stenosis candidate found",
                "Review vessel segmentation for context"
            ],
            start_x=10, start_y=10
        )
        final_overlay = Image.fromarray(no_lesion_img)

    return sub_img, seg_img, final_overlay


# ============================================================
#  Gradio Blocks UI
# ============================================================

with gr.Blocks(title="DeepSA: Enhanced Cardiac Analysis") as demo:

    gr.Markdown(
        "# DeepSA: Enhanced Cardiac Analysis\n"
        "Subtraction, Segmentation, and Automated Stenosis Pinpointing "
        "for Coronary Angiograms."
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Input Angiogram")
            auto_thresh = gr.Checkbox(value=True, label="Auto-Thresholding")
            adv_options = gr.CheckboxGroup(
                ["Multiangle", "Pad margin"], label="Advanced Options"
            )
            submit_btn = gr.Button("Analyze", variant="primary")

        with gr.Column(scale=2):
            out_sub = gr.Image(type="pil", label="1. Enhanced Subtraction")
            out_seg = gr.Image(type="pil", label="2. Vessel Segmentation")
            out_loc = gr.Image(type="pil", label="3. Stenosis Localization (Blockage %)")

    submit_btn.click(
        fn=predict,
        inputs=[input_image, auto_thresh, adv_options],
        outputs=[out_sub, out_seg, out_loc],
    )

    examples_list = list(Path("data/frames").glob("*.png"))
    if examples_list:
        gr.Examples(
            examples=[[str(e)] for e in examples_list],
            inputs=[input_image],
        )

    gr.Markdown(
        "<p style='text-align:center'>Re-implemented for DSGP Group 3 Project</p>"
    )

    # ── Auto-load the selected frame on page visit ──────────────────
    #
    #  demo.load() fires every time the browser loads the page.
    #  Gradio automatically injects the live HTTP request object when
    #  the function signature includes `request: gr.Request`, so we
    #  can read ?image= directly in Python.
    #
    #  The PIL Image returned by on_page_load() goes straight into
    #  `input_image` through Gradio's own internal data pipeline —
    #  exactly as if the user had uploaded it manually.
    # ────────────────────────────────────────────────────────────────
    demo.load(
        fn=on_page_load,
        inputs=[],
        outputs=[input_image],
    )


# ============================================================
#  Entry point
# ============================================================

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
    )