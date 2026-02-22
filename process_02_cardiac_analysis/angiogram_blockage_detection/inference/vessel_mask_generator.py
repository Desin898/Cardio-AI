import torch
import torchvision.transforms as T
from PIL import Image
import os
import sys

# FIX: Tells Python to find 'models' in the folder above this one (inference -> angiogram_blockage_detection)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.models import UNet
import utils

# --- CONFIGURATION ---
SIZE = 512
DEVICE = "cpu"

# FIX: Update this path to where you just pasted the file in Step 1
CHECKPOINT_PATH = os.path.abspath("../models/fscad_36249.ckpt")
DATA_FOLDER = "../data/"
OUTPUT_FOLDER = "../outputs/"

# Optimized Transform using utils logic
transform = T.Compose([
    T.Resize((SIZE, SIZE)),
    T.Lambda(lambda img: utils.apply_tophat(img, 50)),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])


def load_vessel_model(path):
    # Initialize architecture
    model = UNet(1, 1, 32, bilinear=True, res=False)

    if not os.path.exists(path):
        print(f"CRITICAL ERROR: Checkpoint file NOT FOUND at: {path}")
        print("Please ensure you copied fscad_36249.ckpt to your 'models' folder.")
        return None

    checkpoint = torch.load(path, map_location=DEVICE)
    # Clean state dict keys from 'netE' as required by DeepSA logic
    new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['netE'].items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    return model


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    print("--- Initializing Vessel Segmentation ---")
    net = load_vessel_model(CHECKPOINT_PATH)

    if net:
        # Images currently in your data folder
        samples = ["140.png", "148.png", "285.png"]

        for img_name in samples:
            img_path = os.path.join(DATA_FOLDER, img_name)
            if not os.path.exists(img_path):
                print(f"Warning: {img_name} not found in data folder. Skipping...")
                continue

            print(f"Generating mask for: {img_name}")
            raw_img = Image.open(img_path).convert("L")
            input_tensor = transform(raw_img).unsqueeze(0)

            with torch.no_grad():
                # Inference
                output = utils.fusion_predict(net, input_tensor, device=DEVICE)
                # Cleanup using morphological yen thresholding and small object removal
                mask_array = utils.make_mask(T.ToPILImage()(output[0]), remove_size=1000)
                mask_img = Image.fromarray(mask_array)

            save_path = os.path.join(OUTPUT_FOLDER, f"vessel_mask_{img_name}")
            mask_img.save(save_path)
            print(f"Successfully saved to: {save_path}")