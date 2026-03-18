#!/usr/bin/env python3 :)

"""
train_2d.py

Train a 2D U-Net for ionization cone segmentation on real + synthetic data.
Automatically checks dataset directories, validates image/mask counts,
tracks training/validation loss + dice score, saves best model, and plots curves.
"""

import os
import torch
from torch import optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


from src.machine_learning.datasets.ionization_dataset import IonizationConeDataset2D
from src.machine_learning.models.model_2d import UNet
from src.machine_learning.losses.combined_BCE_Dice import BCEDiceLoss


# --------------------------
# CONFIG
# --------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASETS = [
    ("data/2d/processed/real", "data/2d/masks/real"),
    ("data/2d/processed/synthetic", "data/2d/masks/synthetic")
]
MODEL_SAVE_PATH = "results/unet_best_2d.pth"
BATCH_SIZE = 1
EPOCHS = 200
LR = 1e-3
VAL_SPLIT = 0.1
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# --------------------------
# HELPER FUNCTIONS
# --------------------------
def dice_coefficient(preds, targets, threshold=0.5, eps=1e-6):
    """Computes Dice score for predicted masks."""
    preds = (torch.sigmoid(preds) > threshold).float()
    targets = targets.float()
    intersection = (preds * targets).sum()
    return (2. * intersection + eps) / (preds.sum() + targets.sum() + eps)

def check_dataset_paths(datasets):
    """Verify dataset directories and matching image/mask counts."""
    valid_datasets = []
    print("\n🔍 Checking dataset paths...")
    for data_root, mask_root in datasets:
        if not os.path.isdir(data_root):
            print(f"⚠️  Data directory missing: {data_root}")
            continue
        if not os.path.isdir(mask_root):
            print(f"⚠️  Mask directory missing: {mask_root}")
            continue
        images = [f for f in os.listdir(data_root) if f.endswith((".fits", ".png", ".jpg"))]
        masks  = [f for f in os.listdir(mask_root) if f.endswith((".fits", ".png", ".jpg"))]
        if len(images) != len(masks):
            print(f"⚠️  Skipping {data_root} — image/mask count mismatch ({len(images)} vs {len(masks)})")
            continue
        print(f"✅ Dataset OK: {data_root} ({len(images)} samples)")
        valid_datasets.append((data_root, mask_root))
    return valid_datasets

# --------------------------
# DATASET LOADING
# --------------------------
valid_datasets = check_dataset_paths(DATASETS)
if not valid_datasets:
    raise RuntimeError("No valid datasets found! Add images/masks before training.")

datasets = []
for data_root, mask_root in valid_datasets:  
    datasets.append(
        IonizationConeDataset2D(
            image_dir=data_root,
            mask_dir=mask_root
        )
    )


full_dataset = ConcatDataset(datasets)
val_size = int(VAL_SPLIT * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)

print(f"\nLoaded {len(full_dataset)} samples "
      f"({train_size} train / {val_size} val)")

# --------------------------
# MODEL, LOSS, OPTIMIZER
# --------------------------
model = UNet(in_channels=1, out_channels=1).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = BCEDiceLoss(bce_weight=1.0, dice_weight=3.0, pos_weight=20.0).to(DEVICE)

# --------------------------
# TRAINING LOOP
# --------------------------
best_val_loss = float("inf")
history = {"train_loss": [], "val_loss": [], "val_dice": []}

print("\nStarting training...")
for epoch in range(EPOCHS):
    # --- Training ---
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        preds = model(images)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_train_loss = running_loss / len(train_dataset)

    # --- Validation ---
    model.eval()
    val_loss, val_dice = 0.0, 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            preds = model(images)
            val_loss += criterion(preds, masks).item() * images.size(0)
            val_dice += dice_coefficient(preds, masks).item() * images.size(0)

    epoch_val_loss = val_loss / len(val_dataset)
    epoch_val_dice = val_dice / len(val_dataset)

    # --- Logging ---
    history["train_loss"].append(epoch_train_loss)
    history["val_loss"].append(epoch_val_loss)
    history["val_dice"].append(epoch_val_dice)

    print(f"📉 Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {epoch_train_loss:.4f} | "
          f"Val Loss: {epoch_val_loss:.4f} | "
          f"Val Dice: {epoch_val_dice:.4f}")

    # --- Save best model ---
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        os.makedirs("results", exist_ok=True)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Best model saved at {MODEL_SAVE_PATH}")

# --------------------------
# PLOT TRAINING CURVES
# --------------------------
plt.figure(figsize=(8,5))
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.plot(history["val_dice"], label="Val Dice", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Loss / Dice")
plt.title("Training Curves (2D U-Net)")
plt.legend()
plt.savefig("results/training_curves.png")
plt.close()

print("\nTraining curves saved to results/training_curves.png")
print(f"Best model: {MODEL_SAVE_PATH} (Val Loss {best_val_loss:.4f})")

print("Image min/max:", images.min().item(), images.max().item())
print("Mask min/max:", masks.min().item(), masks.max().item())
print("Pred min/max:", preds.min().item(), preds.max().item())

