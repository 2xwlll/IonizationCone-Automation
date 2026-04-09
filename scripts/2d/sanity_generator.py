#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import argparse
import json

# ─────────────────────────────────────────────
# CONFIG (CLI)
# ─────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="synthetic_bicone")
args = parser.parse_args()

BASE_DIR = Path("data/2d") / args.name

GRID = 64
N_SAMPLES = 200

# ─────────────────────────────────────────────
# RESET DATASET (SAFE)
# ─────────────────────────────────────────────

def reset():
    if BASE_DIR.exists():
        assert "2d" in str(BASE_DIR), "Refusing to delete outside data/2d"
        print(f"Resetting dataset at: {BASE_DIR}")
        shutil.rmtree(BASE_DIR)

    for split in ["train", "val", "test"]:
        (BASE_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (BASE_DIR / split / "masks").mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# GEOMETRY
# ─────────────────────────────────────────────

def axis(phi, theta):
    phi = np.radians(phi)
    theta = np.radians(theta)
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ], dtype=np.float32)

def cone(grid, axis_vec, opening_deg, radius):
    c = grid // 2
    z, y, x = np.mgrid[0:grid, 0:grid, 0:grid]

    v = np.stack([x - c, y - c, z - c], axis=-1)
    dist = np.linalg.norm(v, axis=-1) + 1e-8
    v_unit = v / dist[..., None]

    cosang = np.sum(v_unit * axis_vec, axis=-1)

    return (cosang >= np.cos(np.radians(opening_deg))) & (dist <= radius)

# ─────────────────────────────────────────────
# ORIENTATION SAMPLING (KEY CHANGE)
# ─────────────────────────────────────────────

def sample_theta():
    mode = np.random.choice(
        ["toward", "away", "edge", "uniform"],
        p=[0.3, 0.3, 0.2, 0.2]
    )

    if mode == "toward":
        u = np.random.uniform(0.3, 1.0)
    elif mode == "away":
        u = np.random.uniform(-1.0, -0.3)
    elif mode == "edge":
        u = np.random.uniform(-0.3, 0.3)
    else:
        u = np.random.uniform(-1.0, 1.0)

    return np.degrees(np.arccos(u))

# ─────────────────────────────────────────────
# TRUNCATION
# ─────────────────────────────────────────────

def truncate(vol, p=0.2):
    if np.random.random() > p:
        return vol

    cut = np.random.randint(GRID // 4, GRID // 2)

    if np.random.random() < 0.5:
        vol[:cut, :, :] = 0
    else:
        vol[cut:, :, :] = 0

    return vol

# ─────────────────────────────────────────────
# PROJECTION
# ─────────────────────────────────────────────

def project(vol):
    return vol.max(axis=0).astype(np.float32)

# ─────────────────────────────────────────────
# SAMPLE GENERATION
# ─────────────────────────────────────────────

def maybe_blank():
    return np.random.random() < 0.1

def make_sample():
    # Blank case
    if maybe_blank():
        img = np.zeros((GRID, GRID), dtype=np.float32)
        mask = np.zeros((GRID, GRID), dtype=np.float32)
        return img, mask

    r = GRID // 2 - 2
    opening = 25

    phi = np.random.uniform(0, 360)
    theta = sample_theta()

    ax1 = axis(phi, theta)
    ax2 = -ax1  # clean opposite cone

    v1 = cone(GRID, ax1, opening, r)
    v2 = cone(GRID, ax2, opening, r)

    vol = v1 | v2
    vol = truncate(vol, p=0.2)

    image = project(vol)
    image = image / (image.max() + 1e-8)

    mask = project(vol).astype(np.float32)
    mask = (mask > 0).astype(np.float32)

    return image, mask

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

def generate():
    return [make_sample() for _ in range(N_SAMPLES)]

def save(samples):
    for i, (img, mask) in enumerate(samples):
        split = "train" if i < 160 else "val" if i < 180 else "test"

        np.save(BASE_DIR / split / "images" / f"{i:04d}.npy", img)
        np.save(BASE_DIR / split / "masks" / f"{i:04d}.npy", mask)

# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def viz(samples):
    plt.figure(figsize=(8, 8))
    for i in range(9):
        img, _ = samples[i]
        plt.subplot(3, 3, i + 1)
        plt.imshow(img, cmap="inferno")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────────
# METADATA
# ─────────────────────────────────────────────

def save_metadata():
    meta = {
        "grid": GRID,
        "n_samples": N_SAMPLES,
        "blank_fraction": 0.1,
        "truncation_prob": 0.2,
        "orientation_modes": ["toward", "away", "edge", "uniform"],
        "description": "Sanity-check synthetic bicone dataset with orientation diversity"
    }

    with open(BASE_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    reset()
    samples = generate()
    viz(samples)
    save(samples)
    save_metadata()

    print(f"\nDONE — dataset ready at: {BASE_DIR}\n")
