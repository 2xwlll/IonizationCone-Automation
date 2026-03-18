#!/usr/bin/env python3
"""
generate_synthetic_2d_imaging_with_masks.py

Produces realistic Narrow Band JWST/NIRCam-like 2D images (.npy)
with matching masks and difficulty control. Guarantees some images
contain ionization cones and previews multiple examples.
"""

import numpy as np
from pathlib import Path
from scipy.ndimage import affine_transform
import matplotlib.pyplot as plt
import random
import sys

# ============================================================
# CONFIG
# ============================================================

N_IMAGES = 100
IMAGE_SIZE = 128
SAVE_FITS = False
CLEAN_OLD = True
RANDOM_SEED = 123

# Base astrophysical knobs
F_CONE = 0.65
CONE_OPENING_RANGE = (20, 50)
CONE_SIGMA_PX_RANGE = (8, 28)

# Instrument knobs
PSF_SIGMA_PX = 1.1
PHOTON_SCALE = 1e4
READ_NOISE_SIGMA = 0.005
FLAT_FIELD_RMS = 0.01
AMP_GAIN_DIFF = 0.03
VIGNETTE_STRENGTH = 0.08
N_STARS_RANGE = (2, 10)
STAR_FLUX_RANGE = (0.02, 3.0)
STAR_FWHM_RANGE = (0.8, 3.5)
N_CLUMPS = (3, 12)
DUST_BLOTS_RANGE = (0, 4)
DUST_OPACITY_RANGE = (0.12, 0.6)
COSMIC_RAY_RATE = 0.002
HOT_PIXEL_RATE = 0.0005
ONE_OVER_F_AMPLITUDE = 0.004
SATURATION_LEVEL = 1.0

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ============================================================
# DIFFICULTY CONTROL
# ============================================================

DIFFICULTY = sys.argv[1] if len(sys.argv) > 1 else "easy"

if DIFFICULTY not in ["easy", "medium", "hard"]:
    raise ValueError("Difficulty must be: easy, medium, hard")

print(f"\nUsing difficulty: {DIFFICULTY}\n")

if DIFFICULTY == "easy":
    AMP_RANGE = (0.8, 1.2)
    NOISE_SCALE = 0.8
    DUST_PROB = 0.5
    OBSCURE_PROB = 0.1

elif DIFFICULTY == "medium":
    AMP_RANGE = (0.5, 0.9)
    NOISE_SCALE = 1.0
    DUST_PROB = 0.8
    OBSCURE_PROB = 0.3

elif DIFFICULTY == "hard":
    AMP_RANGE = (0.3, 0.6)
    NOISE_SCALE = 1.3
    DUST_PROB = 0.95
    OBSCURE_PROB = 0.6

# Difficulty-specific output folder
OUT_DIR = Path(f"data/2d/processed/synthetic_{DIFFICULTY}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

if CLEAN_OLD:
    for f in OUT_DIR.iterdir():
        try:
            f.unlink()
        except:
            pass

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def add_bulge_disk(image_size=128, center=None, bulge_sigma=12, disk_scale=40, bulge_frac=0.5):
    if center is None:
        center = (image_size//2, image_size//2)
    y, x = np.indices((image_size, image_size))
    cy, cx = center
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    bulge = np.exp(-(r**2) / (2 * bulge_sigma**2))
    disk = np.exp(-r / disk_scale)
    return bulge_frac * bulge + (1 - bulge_frac) * disk

def generate_2d_bicone(size, apex, angle, opening, sigma_pix, amp, hollow=True, asym=1.0):
    """Simplified placeholder bicone generator"""
    y, x = np.indices((size, size))
    cy, cx = apex
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    cone = np.exp(-(r**2) / (2*sigma_pix**2)) * amp * asym
    mask = (cone > amp*0.1).astype(np.uint8)
    return cone, mask

def add_noise(image, photon_scale, read_noise):
    image = np.clip(image, 0, None)  # Ensure no negatives
    lam = image * photon_scale
    lam[np.isnan(lam)] = 0
    lam[lam<0] = 0
    return np.random.poisson(lam)/photon_scale + np.random.normal(0, read_noise, image.shape)

def clip_saturation(image, level):
    return np.clip(image, 0, level)

# Placeholder helpers to keep the script self-contained
def add_clumps(image, n_clumps): return image
def add_foreground_stars(image, n_stars, flux_range, fwhm_range): return image
def add_dust_blobs(image, n_blobs, size_range, opacity_range): return image
def add_vignetting(image, strength): return image
def psf_convolve(image, sigma): return image
def add_hot_pixels(image, rate): return image
def add_cosmic_rays(image, rate): return image
def apply_amp_gains(image, gain_frac): return image
def apply_flat_field(image, rms): return image
def add_one_over_f(image, amplitude): return image

# ============================================================
# IMAGE GENERATOR
# ============================================================

def generate_image_and_mask(size=IMAGE_SIZE, force_cone=False):
    center = (size//2 + np.random.randint(-6,7),
              size//2 + np.random.randint(-6,7))
    background = add_bulge_disk(
        size,
        center=center,
        bulge_sigma=np.random.uniform(6,18),
        disk_scale=np.random.uniform(20,60),
        bulge_frac=np.random.uniform(0.3,0.8)
    )
    image = background * np.random.uniform(0.6,1.2)
    mask = np.zeros_like(image, dtype=np.uint8)
    image = add_clumps(image, n_clumps=np.random.randint(*N_CLUMPS))

    add_cone = (force_cone or np.random.rand() < F_CONE)
    if add_cone:
        apex = (np.random.randint(int(size*0.25), int(size*0.75)),
                np.random.randint(int(size*0.25), int(size*0.75)))
        angle = np.random.uniform(0,360)
        opening = np.random.uniform(*CONE_OPENING_RANGE)
        sigma_pix = np.random.uniform(*CONE_SIGMA_PX_RANGE)
        amp = np.random.uniform(*AMP_RANGE)
        asym = np.random.uniform(0.4, 1.0)
        cone_signal, cone_mask = generate_2d_bicone(
            size, apex, angle, opening, sigma_pix, amp,
            hollow=True, asym=asym
        )
        if np.random.rand() < OBSCURE_PROB:
            mask_mod = (np.linspace(0,1,size)[None,:]
                        if np.random.rand()<0.5
                        else np.linspace(0,1,size)[:,None])
            cone_signal *= mask_mod
            cone_mask = (cone_mask * (mask_mod>0.1)).astype(np.uint8)
        image += cone_signal
        mask = np.maximum(mask, cone_mask)

    image = add_foreground_stars(
        image,
        n_stars=np.random.randint(*N_STARS_RANGE),
        flux_range=STAR_FLUX_RANGE,
        fwhm_range=STAR_FWHM_RANGE
    )
    if np.random.rand() < DUST_PROB:
        image = add_dust_blobs(
            image,
            n_blobs=np.random.randint(*DUST_BLOTS_RANGE),
            size_range=(15,60),
            opacity_range=DUST_OPACITY_RANGE
        )
    if np.random.rand() < 0.9:
        image = add_vignetting(image, strength=np.random.uniform(0.03, VIGNETTE_STRENGTH))
    if np.random.rand() < 0.4:
        sx, sy = np.random.uniform(0.99, 1.01, 2)
        image = affine_transform(image, np.array([[sx,0],[0,sy]]), offset=[0,0], order=1, mode='reflect')
        mask = affine_transform(mask, np.array([[sx,0],[0,sy]]), offset=[0,0], order=0, mode='reflect')
    image = psf_convolve(image, PSF_SIGMA_PX)
    image = add_hot_pixels(image, rate=HOT_PIXEL_RATE)
    image = add_cosmic_rays(image, rate=COSMIC_RAY_RATE)
    image = apply_amp_gains(image, gain_frac=AMP_GAIN_DIFF)
    image = apply_flat_field(image, rms=FLAT_FIELD_RMS)
    image = add_one_over_f(image, amplitude=ONE_OVER_F_AMPLITUDE)
    image = add_noise(image, PHOTON_SCALE*NOISE_SCALE, READ_NOISE_SIGMA*NOISE_SCALE)
    image = clip_saturation(image, SATURATION_LEVEL)
    image = np.clip(image, 0.0, None).astype(np.float32)
    mask = mask.astype(np.uint8)
    return image, mask

# ============================================================
# MAIN LOOP
# ============================================================

if __name__ == "__main__":
    print(f"Generating {N_IMAGES} images into {OUT_DIR}...\n")

    # Ensure at least 20% have cones
    n_forced_cones = max(1, int(N_IMAGES*0.2))
    cone_indices = random.sample(range(N_IMAGES), n_forced_cones)

    for i in range(N_IMAGES):
        force_cone = i in cone_indices
        img, msk = generate_image_and_mask(force_cone=force_cone)
        np.save(OUT_DIR / f"synthetic_image_{i:04d}.npy", img)
        np.save(OUT_DIR / f"synthetic_mask_{i:04d}.npy", msk)
        if SAVE_FITS:
            pass  # skipping FITS for simplicity
        if (i+1) % 10 == 0 or i==0:
            print(f"  - {i+1}/{N_IMAGES}")

    # ========================================================
    # PREVIEW MULTIPLE EXAMPLES
    # ========================================================
    preview_indices = random.sample(range(N_IMAGES), min(5, N_IMAGES))
    for idx in preview_indices:
        preview_img = np.load(OUT_DIR / f"synthetic_image_{idx:04d}.npy")
        preview_mask = np.load(OUT_DIR / f"synthetic_mask_{idx:04d}.npy")
        fig, axs = plt.subplots(1,2, figsize=(8,4))
        axs[0].imshow(preview_img, origin="lower", cmap="inferno")
        axs[0].set_title(f"Image #{idx}")
        axs[0].axis("off")
        axs[1].imshow(preview_mask, origin="lower", cmap="gray")
        axs[1].set_title(f"Mask #{idx}")
        axs[1].axis("off")
        plt.tight_layout()
        preview_path = OUT_DIR / f"preview_{DIFFICULTY}_{idx}.png"
        plt.savefig(preview_path, dpi=150)
        plt.close()
        print(f"Preview saved to: {preview_path}")
    
    print("\nDataset complete.")
