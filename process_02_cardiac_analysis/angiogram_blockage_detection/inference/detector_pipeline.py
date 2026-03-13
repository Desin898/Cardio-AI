import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageChops, ImageOps
import cv2
from skimage import morphology

from models import UNet
from datasets import tophat
from utils import fusion_predict, make_mask, clear_mask
from inference.vessel_analyzer import VesselAnalyzer


class StenosisDetector:
    """
    Full stenosis detection pipeline:
    Input image -> DeepSA enhancement -> vessel segmentation ->
    VesselAnalyzer -> predicted stenosis bounding boxes
    """

    def __init__(self, ckpt_path="ckpt/fscad_36249.ckpt", size=512, device=None):
        self.size = size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_path = ckpt_path

        # Preprocessing transforms
        self.tfmc1 = T.Compose([
            T.Resize(self.size),
            T.Lambda(lambda img: tophat(img, 50)),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])

        self.tfmc2 = T.Compose([
            T.Resize(self.size),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])

        # Load model once
        self.netE = UNet(1, 1, 32, bilinear=True)
        checkpoint = torch.load(self.ckpt_path, map_location=self.device)
        new_state_dict = {
            k.replace("module.", ""): v
            for k, v in checkpoint["netE"].items()
        }
        self.netE.load_state_dict(new_state_dict)
        self.netE.to(self.device)
        self.netE.eval()

    def _prepare_pil_image(self, image):

        if isinstance(image, Image.Image):
            pil_img = image.convert("L")
            orig_w, orig_h = pil_img.size
            return pil_img, orig_w, orig_h

        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                pil_img = Image.fromarray(image).convert("L")
                orig_h, orig_w = image.shape[:2]
                return pil_img, orig_w, orig_h

            if image.ndim == 3:
                orig_h, orig_w = image.shape[:2]
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb).convert("L")
                return pil_img, orig_w, orig_h

        raise TypeError("Unsupported image type. Expected PIL.Image or numpy.ndarray.")

    def _run_deepsa_pipeline(self, pil_img, auto_thresh=True, multiangle=False, pad_margin=False):

        pad = 50 if pad_margin else 0

        if auto_thresh:
            x1 = self.tfmc1(pil_img)
            x2 = self.tfmc2(pil_img)

            _, out1 = fusion_predict(
                self.netE, ["none"], x1,
                multiangle=multiangle,
                denoise=4,
                size=self.size,
                cutoff=0.4,
                pad=pad,
                netE=True
            )

            _, out2 = fusion_predict(
                self.netE, ["none"], x2,
                multiangle=False,
                denoise=4,
                size=self.size,
                cutoff=0.4,
                pad=pad,
                netE=True
            )

            out_merge = Image.fromarray(
                np.expand_dims(
                    np.max(
                        np.concatenate((np.array(out1), np.array(out2)), axis=2),
                        axis=2
                    ),
                    2
                ).repeat(3, axis=2)
            )

            mask_merge = make_mask(
                out_merge,
                remove_size=2000,
                local_kernel=21,
                hole_max_size=100
            )

            out_merge_gamma = T.functional.adjust_gamma(out_merge, 2)
            seg_img = clear_mask(mask_merge)

            sub_img = ImageChops.invert(out_merge_gamma)
            sub_img = T.ToTensor()(sub_img)
            sub_img = sub_img * (2 ** (-0.5))
            sub_img = T.ToPILImage()(sub_img)

        else:
            x = self.tfmc1(pil_img)
            input_tensor = x.unsqueeze(0).to(self.device)

            with torch.no_grad():
                pred_y = self.netE(input_tensor)

            sub_img = morphology  # placeholder removed below intentionally
            sub_img = pred_y[0].detach().cpu()

            sub_img = ((sub_img + 1) / 2).clamp(0, 1)
            sub_img = T.ToPILImage()(sub_img)
            sub_img = ImageOps.autocontrast(sub_img, cutoff=1)
            sub_img = T.functional.adjust_gamma(sub_img, 2)

            sub_img = ImageChops.invert(sub_img)
            sub_img = T.ToTensor()(sub_img)
            sub_img = sub_img * (2 ** (-0.5))
            sub_img = T.ToPILImage()(sub_img)

            seg_img_tensor = torch.sign(pred_y)
            seg_img_np = ((seg_img_tensor.cpu().detach() + 1) / 2).numpy().astype(bool)
            seg_img_np = morphology.remove_small_objects(seg_img_np, 500)
            seg_img_np = (seg_img_np * 255).astype("uint8")
            seg_img = T.ToPILImage()(seg_img_np[0])

        return sub_img, seg_img

    def _blockages_to_boxes(self, blockages, det_w, det_h, orig_w, orig_h):
        """
        Convert VesselAnalyzer outputs into [x, y, w, h] boxes.
        Scales boxes from detector size back to original image size.
        """
        boxes = []

        scale_x = orig_w / float(det_w)
        scale_y = orig_h / float(det_h)

        for b in blockages:
            x, y = b["coords"]
            r = int(b.get("lesion_radius", 8))

            x1 = max(0, x - r)
            y1 = max(0, y - r)
            x2 = min(det_w - 1, x + r)
            y2 = min(det_h - 1, y + r)

            # scale back to original image size
            sx1 = int(round(x1 * scale_x))
            sy1 = int(round(y1 * scale_y))
            sx2 = int(round(x2 * scale_x))
            sy2 = int(round(y2 * scale_y))

            sw = max(1, sx2 - sx1)
            sh = max(1, sy2 - sy1)

            boxes.append([sx1, sy1, sw, sh])

        return boxes

    def detect(self, image, return_debug=False):
        """
        Main method expected by ArcadeEvaluator.

        Parameters
        ----------
        image : numpy.ndarray or PIL.Image
            Input angiogram image

        return_debug : bool
            If True, also returns extra debug outputs

        Returns
        -------
        list
            Predicted bounding boxes in [x, y, w, h] format
        """
        pil_img, orig_w, orig_h = self._prepare_pil_image(image)

        sub_img, seg_img = self._run_deepsa_pipeline(
            pil_img,
            auto_thresh=True,
            multiangle=False,
            pad_margin=False
        )

        analyzer = VesselAnalyzer(seg_img)
        analyzer.get_vessel_geometry()
        blockages = analyzer.calculate_stenosis()

        det_w, det_h = seg_img.size
        pred_boxes = self._blockages_to_boxes(
            blockages,
            det_w=det_w,
            det_h=det_h,
            orig_w=orig_w,
            orig_h=orig_h
        )

        if return_debug:
            return {
                "pred_boxes": pred_boxes,
                "blockages": blockages,
                "sub_img": sub_img,
                "seg_img": seg_img
            }

        return pred_boxes