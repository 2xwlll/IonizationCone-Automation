#!/usr/bin/env python3

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.machine_learning.datasets.ionization_dataset import IonizationConeDataset2D
from src.machine_learning.models.model_2d import UNet

# --------------------------
# CONFIG
# --------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_RESULTS_DIR = "results/2d/unet"
DATASET_PATH = "data/2d/synthetic_bicone/val"

BATCH_SIZE = 4
NUM_SAMPLES = 6
SAVE_DIR = "results/visualizations"

os.makedirs(SAVE_DIR, exist_ok=True)

# --------------------------
# FIND LATEST RUN
# --------------------------
def get_latest_run(base_dir):
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"No such directory: {base_dir}")

    runs = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("run_")
    ]

    if not runs:
        raise ValueError("No run folders found.")

    runs.sort()
    latest = runs[-1]

    return os.path.join(base_dir, latest)

# --------------------------
# LOAD DATASET
# --------------------------
def load_dataset():
    dataset = IonizationConeDataset2D(
        image_dir=os.path.join(DATASET_PATH, "images"),
        mask_dir=os.path.join(DATASET_PATH, "masks"),
        normalize=False
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False   # IMPORTANT: reproducibility
    )

    return dataset, loader

# --------------------------
# LOAD MODEL
# --------------------------
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    return model

# --------------------------
# VISUALIZATION
# --------------------------
@torch.no_grad()
def visualize(model, dataset):
    # fixed indices → consistent comparison across runs
    indices = list(range(NUM_SAMPLES))

    imgs = torch.stack([dataset[i][0] for i in indices]).to(DEVICE)
    masks = torch.stack([dataset[i][1] for i in indices]).to(DEVICE)

    logits = model(imgs)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    imgs = imgs.cpu()
    masks = masks.cpu()
    probs = probs.cpu()
    preds = preds.cpu()

    for i in range(NUM_SAMPLES):
        fig, axes = plt.subplots(1, 3, figsize=(12, 5))

        # Input
        axes[0].imshow(imgs[i][0], cmap="gray")
        axes[0].set_title("Input")

        # Ground truth
        axes[1].imshow(masks[i][0], cmap="gray")
        axes[1].set_title("Ground Truth")

        # Probability map (VERY useful)
        im = axes[2].imshow(probs[i][0], cmap="viridis")
        axes[2].set_title("Prediction (Probability)")
        fig.colorbar(im, ax=axes[2], fraction=0.046)

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"sample_{i}.png"))
        plt.close()

# --------------------------
# MAIN
# --------------------------
def main():
    print("\n--- VISUALIZATION SCRIPT ---\n")

    # find latest run
    run_dir = get_latest_run(BASE_RESULTS_DIR)
    model_path = os.path.join(run_dir, "models", "best.pth")

    print(f"Using run: {run_dir}")
    print(f"Model path: {model_path}")

    # load model
    print("\nLoading model...")
    model = load_model(model_path)

    # load dataset
    print("Loading dataset...")
    dataset, loader = load_dataset()

    print(f"Dataset size: {len(dataset)} samples")

    # visualize
    print("\nGenerating visualizations...")
    visualize(model, dataset)

    print(f"\nSaved to: {SAVE_DIR}")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
