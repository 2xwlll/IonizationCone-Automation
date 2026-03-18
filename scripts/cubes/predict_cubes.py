
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import torch
from src.model import UNet
from src.ionization.real_dataset import RealJWSTDataset

# ===== CONFIG =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "results/unet_model.pth"
REAL_DATA_PATH = "data/processed_3d"
OUTPUT_MASK_DIR = Path("data/predicted_masks_real")
BATCH_SIZE = 1

OUTPUT_MASK_DIR.mkdir(parents=True, exist_ok=True)

# ===== Load Model =====
model = UNet(in_channels=16, out_channels=1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ===== Load Data =====
dataset = RealJWSTDataset(image_dir=REAL_DATA_PATH)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===== Inference Loop =====
print(f"🚀 Running inference on {len(dataset)} cubes...")

for i, (cube, fname) in enumerate(loader):
    cube = cube.to(DEVICE)  # [1, 16, 128, 128]
    
    with torch.no_grad():
        output = model(cube)  # [1, 1, 128, 128]
        output = torch.sigmoid(output)
        pred_mask = output.squeeze().cpu().numpy()  # -> (128, 128)

    # Save mask
    mask_name = OUTPUT_MASK_DIR / fname[0].replace(".npy", "_predmask.npy")
    np.save(mask_name, pred_mask)
    print(f"[{i+1}/{len(dataset)}] ✅ Saved mask: {mask_name}")

    # Optional visualization
    if i < 3:
        mid_channel = cube[0, 8].cpu().numpy()  # Show 9th slice (index 8)
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(mid_channel, cmap="gray")
        plt.title("Middle Slice of Cube")
        plt.subplot(1, 2, 2)
        plt.imshow(pred_mask, cmap="plasma")
        plt.title("Predicted Mask")
        plt.show()

print("✅ Inference complete.")

