# scripts/2d/infer_real_nircam.py

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from astropy.io import fits
from skimage.transform import resize

# =========================
# ADD REPO ROOT TO PYTHONPATH
# =========================
# This ensures Python can find src/ no matter where the script is run from
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

# =========================
# IMPORT YOUR MODEL
# =========================
from src.machine_learning.models.model_2d import UNet

# =========================
# CONFIG
# =========================
FITS_PATH = "data/2d/raw/real/MAST_2025-12-19T2002/JWST/jw03707-o128_t002_nircam_clear-f335m/jw03707-o128_t002_nircam_clear-f335m_i2d.fits"
MODEL_PATH = "results/unet_best_2d.pth"
OUT_DIR = "results/real_inference"

IMAGE_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# LOAD FITS
# =========================
with fits.open(FITS_PATH) as hdul:
    img = hdul[1].data if len(hdul) > 1 else hdul[0].data

img = img.astype(np.float32)

# =========================
# PREPROCESS
# =========================
# Handle NaNs
img[np.isnan(img)] = 0.0

# Normalize (same philosophy as synthetic)
img -= img.min()
img /= (img.max() + 1e-8)

# Resize to UNet input
img_resized = resize(
    img,
    (IMAGE_SIZE, IMAGE_SIZE),
    preserve_range=True,
    anti_aliasing=True
).astype(np.float32)

# Shape: (1, 1, H, W)
tensor = torch.from_numpy(img_resized)[None, None, :, :].to(DEVICE)

# =========================
# LOAD MODEL
# =========================
model = UNet(in_channels=1, out_channels=1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# =========================
# INFERENCE
# =========================
with torch.no_grad():
    pred = model(tensor)
    pred = torch.sigmoid(pred)
    pred = pred.cpu().numpy()[0, 0]

# =========================
# SAVE PRODUCTS
# =========================
# Save mask FITS
fits.writeto(
    f"{OUT_DIR}/predicted_mask.fits",
    pred.astype(np.float32),
    overwrite=True
)

# =========================
# FIGURE
# =========================
plt.figure(figsize=(6, 6))
plt.imshow(img_resized, cmap="gray", origin="lower")
plt.contour(pred, levels=[0.5], colors="red", linewidths=1)
plt.title("NGC 1068 – UNet Ionization Cone Prediction")
plt.axis("off")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/ngc1068_unet_overlay.png", dpi=300)
plt.close()

print("Inference complete.")
print(f"Results saved to: {OUT_DIR}")

