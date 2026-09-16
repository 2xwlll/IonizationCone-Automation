#!/usr/bin/env python3

import os
import datetime
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random
import difflib
from src.machine_learning.losses.combined_BCE_Dice import BCEDiceLoss
from src.machine_learning.datasets.ionization_dataset import IonizationConeDataset2D
from src.machine_learning.models.model_2d import UNet

# --------------------------
# RUN DIRS
# --------------------------
RUN_NAME = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")

BASE_RESULTS_DIR = os.path.join("results", "2d", "unet", RUN_NAME)
MODEL_DIR        = os.path.join(BASE_RESULTS_DIR, "models")
PLOT_DIR         = os.path.join(BASE_RESULTS_DIR, "plots")
SAMPLE_DIR       = os.path.join(BASE_RESULTS_DIR, "samples")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

# --------------------------
# CONFIG
# --------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_CONFIG = {
    "root":      "data/2d",
    "name_hint": "synthetic_oiii_realistic",
    "cutoff":    0.5
}

BATCH_SIZE  = 8
EPOCHS      = 30
LR          = 1e-4
POS_WEIGHT  = 20.0   # cone pixels weighted 20x — fixes class imbalance
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# --------------------------
# FUZZY DATASET RESOLVER
# --------------------------
def fuzzy_find_dataset(root, hint, cutoff=0.5):
    if not os.path.exists(root):
        raise FileNotFoundError(f"Dataset root not found: {root}")

    candidates = [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ]

    matches = difflib.get_close_matches(hint, candidates, n=1, cutoff=cutoff)

    if not matches:
        raise ValueError(
            f"No dataset match for '{hint}' in {root}\n"
            f"Available datasets: {candidates}"
        )

    return matches[0]

# --------------------------
# REPRODUCIBILITY
# --------------------------
def worker_init_fn(worker_id):
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)

# --------------------------
# METRICS
# --------------------------
def dice_coefficient(preds, targets, eps=1e-6):
    preds   = (torch.sigmoid(preds) > 0.5).float()
    targets = targets.float()

    dice_per_sample = []
    for p, t in zip(preds, targets):
        intersection = (p * t).sum()
        dice = (2. * intersection + eps) / (p.sum() + t.sum() + eps)
        dice_per_sample.append(dice)

    return torch.stack(dice_per_sample).mean()

# --------------------------
# DATA LOADING
# --------------------------
def load_dataset(path_img, path_mask):
    return IonizationConeDataset2D(
        image_dir=path_img,
        mask_dir=path_mask,
        normalize=False
    )

def build_loaders():
    root    = DATASET_CONFIG["root"]
    hint    = DATASET_CONFIG["name_hint"]
    cutoff  = DATASET_CONFIG["cutoff"]

    dataset_name = fuzzy_find_dataset(root, hint, cutoff)
    base_path    = os.path.join(root, dataset_name)

    train_set = load_dataset(
        os.path.join(base_path, "train/images"),
        os.path.join(base_path, "train/masks")
    )
    val_set = load_dataset(
        os.path.join(base_path, "val/images"),
        os.path.join(base_path, "val/masks")
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print(f"\nUsing dataset: {dataset_name} (hint='{hint}')")
    print(f"Train: {len(train_set)} | Val: {len(val_set)}\n")

    return train_loader, val_loader, len(train_set), len(val_set)

# --------------------------
# CHECKPOINTING
# --------------------------
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "best.pth")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "checkpoint.pth")

def save_checkpoint(model, optimizer, epoch, best_val_loss):
    torch.save({
        "model":          model.state_dict(),
        "optimizer":      optimizer.state_dict(),
        "epoch":          epoch,
        "best_val_loss":  best_val_loss
    }, CHECKPOINT_PATH)

def load_checkpoint(model, optimizer):
    if not os.path.exists(CHECKPOINT_PATH):
        return 0, float("inf")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"Resuming from epoch {ckpt['epoch']+1}")

    return ckpt["epoch"], ckpt["best_val_loss"]

# --------------------------
# TRAIN / EVAL
# --------------------------
def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0

    for imgs, masks in tqdm(loader, leave=False):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()
        preds = model(imgs)
        loss  = loss_fn(preds, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0
    total_dice = 0

    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        preds       = model(imgs)
        loss        = loss_fn(preds, masks)

        total_loss += loss.item()
        total_dice += dice_coefficient(preds, masks).item()

    return total_loss / len(loader), total_dice / len(loader)

# --------------------------
# MAIN
# --------------------------
def main():
    train_loader, val_loader, n_train, n_val = build_loaders()

    # debug check — confirm data looks right before training
    imgs, masks = next(iter(train_loader))
    print(f"IMG  shape={imgs.shape}  min={imgs.min():.4f}  max={imgs.max():.4f}")
    print(f"MASK shape={masks.shape} unique={torch.unique(masks).tolist()}")
    print(f"Device: {DEVICE}\n")

    model     = UNet(in_channels=1, out_channels=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # pos_weight fixes class imbalance — missing a cone pixel costs
    # POS_WEIGHT times more than a background pixel
    loss_fn = BCEDiceLoss()

    start_epoch, best_val_loss = load_checkpoint(model, optimizer)

    history = {"train": [], "val": [], "dice": []}

    for epoch in range(start_epoch, EPOCHS):

        train_loss          = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_loss, val_dice  = evaluate(model, val_loader, loss_fn)

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        history["dice"].append(val_dice)

        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"Train {train_loss:.4f} | "
            f"Val {val_loss:.4f} | "
            f"Dice {val_dice:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> saved best model (val_loss={val_loss:.4f})")

        save_checkpoint(model, optimizer, epoch, best_val_loss)

    # --------------------------
    # PLOT
    # --------------------------
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(history["train"], label="Train Loss", color="steelblue")
    ax1.plot(history["val"],   label="Val Loss",   color="orange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(history["dice"], label="Val Dice", color="green", linestyle="--")
    ax2.set_ylabel("Dice")
    ax2.legend(loc="upper right")

    plt.title("Training Curves")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "training_curves.png"), dpi=150)
    plt.close()

    print(f"\nDone. Results saved to: {BASE_RESULTS_DIR}")

if __name__ == "__main__":
    main()
