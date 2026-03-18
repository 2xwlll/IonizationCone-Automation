# scripts/2d/predict_2d.py

import torch
import numpy as np
from pathlib import Path
from astropy.io import fits
import matplotlib.pyplot as plt

from src.models.model_2d import UNet

# ===== CONFIG =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "results/unet_model_2d.pth"

INPUT_DIR = Path("data/2d/raw/real")  # or synthetic
OUTPUT_DIR = Path("data/2d/predicted/real")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (128, 128)
THRESHOLD = 0.5  # Prediction cutoff

# ===== Load Model =====
print("🧠 Loading model...")
model = UNet(in_channels=1, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ===== Helper =====
def load_and_preprocess_fits(path):
    data = fits.getdata(path).astype(np.float32)
    data = np.nan_to_num(data)
    if data.shape != IMG_SIZE:
        from skimage.transform import resize
        data = resize(data, IMG_SIZE, preserve_range=True, anti_aliasing=True)
    data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
    tensor = torch.tensor(data).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    return tensor

def save_mask(mask_tensor, save_path):
    mask_np = mask_tensor.squeeze().cpu().numpy().astype(np.float32)
    fits.writeto(save_path, mask_np, overwrite=True)

# ===== Run Prediction =====
fits_files = list(INPUT_DIR.glob("*.fits"))
print(f"🔍 Found {len(fits_files)} FITS files in {INPUT_DIR}")

for i, img_path in enumerate(fits_files):
    input_tensor = load_and_preprocess_fits(img_path).to(DEVICE)

    with torch.no_grad():
        pred = model(input_tensor)
        pred = torch.sigmoid(pred)
        pred_mask = (pred > THRESHOLD).float()

    # Save mask
    filename = img_path.stem
    save_path = OUTPUT_DIR / f"{filename}_mask.fits"
    save_mask(pred_mask, save_path)

    # Optional plot preview for first 3
    if i < 3:
        plt.figure(figsize=(10, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(input_tensor.squeeze().cpu(), cmap="gray")
        plt.title("Input")

        plt.subplot(1, 2, 2)
        plt.imshow(pred_mask.squeeze().cpu(), cmap="plasma")
        plt.title("Predicted Mask")

        plt.suptitle(f"Prediction: {filename}")
        plt.show()

    print(f"[{i+1}/{len(fits_files)}] ✅ Saved prediction: {save_path}")

