#!/usr/bin/env python3

import os
import datetime
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random
import difflib

from src.machine_learning.datasets.ionization_dataset import IonizationConeDataset2D
from src.machine_learning.models.model_2d import UNet
from src.machine_learning.losses.combined_BCE_Dice import BCEDiceLoss

# --------------------------
# RUN DIRS
# --------------------------
RUN_NAME = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")

BASE_RESULTS_DIR = os.path.join("results", "2d", "unet", RUN_NAME)
MODEL_DIR = os.path.join(BASE_RESULTS_DIR, "models")
PLOT_DIR = os.path.join(BASE_RESULTS_DIR, "plots")
SAMPLE_DIR = os.path.join(BASE_RESULTS_DIR, "samples")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

# --------------------------
# CONFIG
# --------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_CONFIG = {
    "root": "data/2d",
    "name_hint": "synthetic_bicone",
    "cutoff": 0.5  # fuzziness knob
}

BATCH_SIZE = 2
EPOCHS = 10
LR = 1e-3
SEED = 42

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
    preds = (torch.sigmoid(preds) > 0.5).float()
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
    root = DATASET_CONFIG["root"]
    hint = DATASET_CONFIG["name_hint"]
    cutoff = DATASET_CONFIG["cutoff"]

    dataset_name = fuzzy_find_dataset(root, hint, cutoff)
    base_path = os.path.join(root, dataset_name)

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

    print(f"\nUsing dataset: {dataset_name} (hint='{hint}')\n")

    return train_loader, val_loader, len(train_set), len(val_set)

# --------------------------
# CHECKPOINTING
# --------------------------
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "best.pth")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "checkpoint.pth")

def save_checkpoint(model, optimizer, scaler, epoch, best_val_loss):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss
    }, CHECKPOINT_PATH)

def load_checkpoint(model, optimizer, scaler):
    if not os.path.exists(CHECKPOINT_PATH):
        return 0, float("inf")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])

    return ckpt["epoch"], ckpt["best_val_loss"]

# --------------------------
# TRAIN / EVAL
# --------------------------
def train_one_epoch(model, loader, optimizer, loss_fn, scaler):
    model.train()
    total_loss = 0

    from contextlib import nullcontext
    amp_context = autocast if DEVICE == "cuda" else nullcontext

    for imgs, masks in tqdm(loader, leave=False):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()

        with amp_context():
            preds = model(imgs)
            loss = loss_fn(preds, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()

    total_loss = 0
    total_dice = 0

    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        preds = model(imgs)
        loss = loss_fn(preds, masks)

        total_loss += loss.item()
        total_dice += dice_coefficient(preds, masks).item()

    return total_loss / len(loader), total_dice / len(loader)

# --------------------------
# MAIN
# --------------------------
def main():
    train_loader, val_loader, n_train, n_val = build_loaders()

    print(f"Train: {n_train} | Val: {n_val}")

    #DEBUG HERE (early, before training)
    imgs, masks = next(iter(train_loader))
    print("IMG min/max:", imgs.min().item(), imgs.max().item())
    print("MASK unique:", torch.unique(masks))

    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler()

    loss_fn = nn.BCEWithLogitsLoss()

    start_epoch, best_val_loss = load_checkpoint(model, optimizer, scaler)

    history = {"train": [], "val": [], "dice": []}

    for epoch in range(start_epoch, EPOCHS):

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, scaler)
        val_loss, val_dice = evaluate(model, val_loader, loss_fn)

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        history["dice"].append(val_dice)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train {train_loss:.4f} | "
            f"Val {val_loss:.4f} | "
            f"Dice {val_dice:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

        save_checkpoint(model, optimizer, scaler, epoch, best_val_loss)

    # --------------------------
    # PLOT
    # --------------------------
    plt.figure()
    plt.plot(history["train"], label="Train Loss")
    plt.plot(history["val"], label="Val Loss")
    plt.plot(history["dice"], label="Val Dice")
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, "training_curves.png"))
    plt.close()

    print("\nDone.")
    print(f"Saved to: {BASE_RESULTS_DIR}")
    imgs, masks = next(iter(train_loader))
    print(imgs.min().item(), imgs.max().item())

if __name__ == "__main__":
    main()
