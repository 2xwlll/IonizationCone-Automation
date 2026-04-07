#!/usr/bin/env python3

import os
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from src.machine_learning.datasets.ionization_dataset import IonizationConeDataset2D
from src.machine_learning.models.model_2d import UNet


# --------------------------
# CONFIG
# --------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_IMG_DIR = "data/2d/derived_from_cubes/synthetic/processed/norm_v1/val/images"
DEFAULT_MASK_DIR = "data/2d/derived_from_cubes/synthetic/processed/norm_v1/val/masks"


# --------------------------
# ARGUMENTS
# --------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, default=None, help="Path to model .pth (auto if None)")
parser.add_argument("--data_img", type=str, default=None)
parser.add_argument("--data_mask", type=str, default=None)
parser.add_argument("--num_samples", type=int, default=5)
parser.add_argument("--save_dir", type=str, default="viz_outputs")

args = parser.parse_args()


# --------------------------
# AUTO-FIND LATEST RUN
# --------------------------
def get_latest_model():
    base = Path("results/2d/unet")

    if not base.exists():
        raise FileNotFoundError(f"Missing directory: {base}")

    runs = sorted([p for p in base.iterdir() if p.is_dir()])

    if not runs:
        raise FileNotFoundError("No training runs found.")

    latest_run = runs[-1]
    model_path = latest_run / "models" / "best.pth"

    if not model_path.exists():
        raise FileNotFoundError(f"No model found at: {model_path}")

    return str(model_path), latest_run


# --------------------------
# RESOLVE PATHS
# --------------------------
if args.model is None:
    args.model, run_dir = get_latest_model()
    print(f"\nAuto mode enabled")
    print(f"Using run: {run_dir.name}")
    print(f"Model: {args.model}\n")

if args.data_img is None:
    args.data_img = DEFAULT_IMG_DIR

if args.data_mask is None:
    args.data_mask = DEFAULT_MASK_DIR


# --------------------------
# LOAD MODEL
# --------------------------
model = UNet(in_channels=1, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load(args.model, map_location=DEVICE))
model.eval()


# --------------------------
# LOAD DATA
# --------------------------
dataset = IonizationConeDataset2D(
    image_dir=args.data_img,
    mask_dir=args.data_mask
)

loader = DataLoader(dataset, batch_size=1, shuffle=True)


# --------------------------
# OUTPUT DIR
# --------------------------
os.makedirs(args.save_dir, exist_ok=True)


# --------------------------
# VISUALIZATION LOOP
# --------------------------
count = 0

print("Running visualization...\n")

with torch.no_grad():
    for imgs, masks in loader:
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        preds = torch.sigmoid(model(imgs))

        img = imgs[0].cpu().squeeze()
        mask = masks[0].cpu().squeeze()
        pred = (preds[0].cpu().squeeze() > 0.5).float()
        error = torch.abs(pred - mask)

        # --------------------------
        # PLOT
        # --------------------------
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))

        axes[0].imshow(img, cmap="gray")
        axes[0].set_title("Image")

        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Mask")

        axes[2].imshow(pred, cmap="gray")
        axes[2].set_title("Prediction")

        axes[3].imshow(error, cmap="hot")
        axes[3].set_title("Error")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()

        save_path = os.path.join(args.save_dir, f"sample_{count:03d}.png")
        plt.savefig(save_path)
        plt.close()

        print(f"Saved {save_path}")

        count += 1
        if count >= args.num_samples:
            break

print("\nDone.")
