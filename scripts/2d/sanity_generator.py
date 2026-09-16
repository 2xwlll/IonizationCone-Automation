#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import argparse
import json

# =========================================================
# CONFIG
# =========================================================

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="synthetic_oiii_realistic")
args = parser.parse_args()

BASE_DIR = Path("data/2d") / args.name

GRID = 128
N_SAMPLES = 1000

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

# =========================================================
# RESET DATASET
# =========================================================

def reset():
    if BASE_DIR.exists():
        assert "2d" in str(BASE_DIR)
        print(f"Resetting dataset: {BASE_DIR}")
        shutil.rmtree(BASE_DIR)

    for split in ["train", "val", "test"]:
        (BASE_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (BASE_DIR / split / "masks").mkdir(parents=True, exist_ok=True)

# =========================================================
# GEOMETRY HELPERS
# =========================================================

def axis(phi, theta):
    phi = np.radians(phi)
    theta = np.radians(theta)
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ], dtype=np.float32)

def pixel_vectors(grid):
    c = grid // 2
    y, x = np.mgrid[0:grid, 0:grid]
    z = np.full_like(x, c)

    v = np.stack([x - c, y - c, z], axis=-1)
    norm = np.linalg.norm(v, axis=-1) + 1e-8
    return v / norm[..., None]

# =========================================================
# PHYSICAL EMISSION MODEL (CORE IDEA)
# =========================================================

def cone_illumination(v_dir, axis_vec, opening_angle):
    cosang = np.sum(v_dir * axis_vec, axis=-1)
    angle = np.arccos(np.clip(cosang, -1, 1))

    # soft cone, NOT binary
    return np.exp(-(angle / np.radians(opening_angle))**2)

def radial_decay(shape, r0=None):
    grid = shape[0]
    c = grid // 2
    y, x = np.mgrid[0:grid, 0:grid]
    r = np.sqrt((x - c)**2 + (y - c)**2)

    r0 = r0 or (grid * 0.3)
    return np.exp(-(r / r0))

def clumpy_gas(grid, n_blobs=25):
    img = np.zeros((grid, grid), dtype=np.float32)

    for _ in range(n_blobs):
        y = np.random.randint(0, grid)
        x = np.random.randint(0, grid)
        amp = np.random.uniform(0.2, 1.0)
        sigma = np.random.uniform(2, 10)

        yy, xx = np.mgrid[0:grid, 0:grid]
        blob = np.exp(-((xx-x)**2 + (yy-y)**2) / (2*sigma**2))
        img += amp * blob

    return img / (img.max() + 1e-8)

def psf_blur(img):
    # cheap Gaussian blur (no scipy dependency)
    kernel = np.array([[1,2,1],
                       [2,4,2],
                       [1,2,1]], dtype=np.float32)
    kernel /= kernel.sum()

    pad = np.pad(img, 1, mode="reflect")
    out = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i,j] = np.sum(pad[i:i+3, j:j+3] * kernel)

    return out

# =========================================================
# SAMPLE GENERATION
# =========================================================

def maybe_blank():
    return np.random.rand() < 0.12

def sample_params():
    return {
        "has_agn": np.random.rand() > 0.15,
        "n_cones": np.random.choice([0,1,2], p=[0.2,0.4,0.4]),
        "opening": np.random.uniform(15, 45),
        "asymmetry": np.random.uniform(0.5, 1.5),
        "clumpiness": np.random.uniform(0.5, 2.0),
        "noise": np.random.uniform(0.01, 0.05)
    }

def generate_sample():
    params = sample_params()
    grid = GRID

    if maybe_blank() or not params["has_agn"]:
        img = clumpy_gas(grid, n_blobs=10)
        mask = np.zeros((grid, grid), dtype=np.float32)
        return img, mask

    vdir = pixel_vectors(grid)

    emission = np.zeros((grid, grid), dtype=np.float32)
    mask = np.zeros((grid, grid), dtype=np.float32)

    n_cones = params["n_cones"]

    for _ in range(n_cones):
        phi = np.random.uniform(0, 360)
        theta = np.random.uniform(0, 180)
        axis_vec = axis(phi, theta)

        illum = cone_illumination(vdir, axis_vec, params["opening"])

        rdecay = radial_decay((grid, grid))

        cone_field = illum * rdecay

        emission += cone_field
        mask += illum

    # physical components
    gas = clumpy_gas(grid, n_blobs=int(20 * params["clumpiness"]))
    emission += 0.6 * gas

    emission = psf_blur(emission)

    noise = np.random.normal(0, params["noise"], emission.shape)
    emission += noise

    emission = np.clip(emission, 0, None)
    emission /= (emission.max() + 1e-8)

    mask = mask / (mask.max() + 1e-8)

    return emission.astype(np.float32), mask.astype(np.float32)

# =========================================================
# DATASET BUILD
# =========================================================

def generate():
    return [generate_sample() for _ in range(N_SAMPLES)]

def save(samples):
    for i, (img, mask) in enumerate(samples):

        if i < int(N_SAMPLES * TRAIN_SPLIT):
            split = "train"
        elif i < int(N_SAMPLES * (TRAIN_SPLIT + VAL_SPLIT)):
            split = "val"
        else:
            split = "test"

        np.save(BASE_DIR / split / "images" / f"{i:05d}.npy", img)
        np.save(BASE_DIR / split / "masks" / f"{i:05d}.npy", mask)

# =========================================================
# VISUAL CHECK
# =========================================================

def viz(samples):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        img, mask = samples[i]
        plt.subplot(3, 3, i + 1)
        plt.imshow(img, cmap="inferno")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# =========================================================
# METADATA
# =========================================================

def save_metadata():
    meta = {
        "grid": GRID,
        "samples": N_SAMPLES,
        "model": "illumination_field + clumpy gas + PSF blur",
        "mask_type": "soft cone illumination field",
        "note": "designed for UNet generalization to JWST/HST OIII morphology"
    }

    with open(BASE_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    reset()
    samples = generate()
    viz(samples[:9])
    save(samples)
    save_metadata()

    print(f"\nDONE → {BASE_DIR}\n")
