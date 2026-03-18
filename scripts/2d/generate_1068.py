#!/usr/bin/env python3
"""
generate_realistic_2d_cones.py

Produces realistic JWST/NIRCam-like 2D emission-line images
with hollow ionization cones and matching masks.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import affine_transform, gaussian_filter
import random
from astropy.io import fits

# ================= CONFIG =================

N_IMAGES = 50
IMAGE_SIZE = 128
SAVE_FITS = False
CLEAN_OLD = True
RANDOM_SEED = 123

# Instrumental / noise knobs
PSF_SIGMA_PX = 1.1
PHOTON_SCALE = 1e4
READ_NOISE_SIGMA = 0.005
FLAT_FIELD_RMS = 0.01
VIGNETTE_STRENGTH = 0.08
HOT_PIXEL_RATE = 0.0005
COSMIC_RAY_RATE = 0.002
ONE_OVER_F_AMPLITUDE = 0.004
SATURATION_LEVEL = 1.0

# Foreground stars
N_STARS_RANGE = (0, 5)
STAR_FLUX_RANGE = (0.02, 3.0)
STAR_FWHM_RANGE = (0.8, 3.5)

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Output
OUT_DIR = Path("data/2d/processed/realistic_cones")
OUT_DIR.mkdir(parents=True, exist_ok=True)
if CLEAN_OLD:
    for f in OUT_DIR.iterdir():
        try: f.unlink()
        except: pass

# ================== HELPERS ==================

def add_hollow_cone(image_size, apex, angle_deg, opening_deg, sigma_px, amp, asym=1.0):
    """
    Generates a hollow, fan-shaped ionization cone
    """
    y, x = np.indices((image_size, image_size))
    dx = x - apex[1]
    dy = y - apex[0]
    theta = np.degrees(np.arctan2(dy, dx)) % 360
    r = np.sqrt(dx**2 + dy**2)

    # Compute angular mask
    half_open = opening_deg / 2
    ang_min = (angle_deg - half_open) % 360
    ang_max = (angle_deg + half_open) % 360

    # Handle wrap-around
    if ang_min < ang_max:
        cone_mask = (theta >= ang_min) & (theta <= ang_max)
    else:
        cone_mask = (theta >= ang_min) | (theta <= ang_max)

    # Hollow profile along radius
    cone_profile = np.exp(-0.5*((r - r.max()/2)/sigma_px)**2)
    cone_profile *= np.random.uniform(0.7, 1.0)  # slight randomness

    # Asymmetry
    cone_profile *= 1 + (np.random.rand(*cone_profile.shape)-0.5)*(1-asym)

    cone = amp * cone_mask * cone_profile
    cone_mask_uint = (cone>0).astype(np.uint8)

    return cone, cone_mask_uint

def add_noise(image, photon_scale=1e4, read_noise=0.005):
    """
    Poisson + Gaussian noise + 1/f noise
    """
    image = np.clip(image, 0.0, None)
    noisy = np.random.poisson(image*photon_scale)/photon_scale
    noisy += np.random.normal(0, read_noise, size=image.shape)
    # 1/f spatial noise
    f_noise = gaussian_filter(np.random.normal(size=image.shape), sigma=4)
    noisy += ONE_OVER_F_AMPLITUDE * f_noise
    return noisy

def add_hot_pixels(image, rate=0.001):
    mask = np.random.rand(*image.shape) < rate
    image[mask] += np.random.uniform(0.5, 1.0)
    return image

def add_cosmic_rays(image, rate=0.002):
    mask = np.random.rand(*image.shape) < rate
    image[mask] += np.random.uniform(0.1, 0.5)
    return image

def add_vignetting(image, strength=0.05):
    y, x = np.indices(image.shape)
    cy, cx = np.array(image.shape)/2
    r = np.sqrt((x-cx)**2 + (y-cy)**2)
    r /= r.max()
    image *= 1 - strength*r**2
    return image

def add_foreground_stars(image, n_stars=3, flux_range=(0.02,3.0), fwhm_range=(1.0,3.5)):
    for _ in range(n_stars):
        cy = np.random.randint(0,image.shape[0])
        cx = np.random.randint(0,image.shape[1])
        flux = np.random.uniform(*flux_range)
        fwhm = np.random.uniform(*fwhm_range)
        y, x = np.indices(image.shape)
        star = flux * np.exp(-((x-cx)**2 + (y-cy)**2)/(2*(fwhm**2)))
        image += star
    return image

def clip_saturation(image, level=1.0):
    return np.clip(image, 0.0, level)

# ================== IMAGE GENERATOR ==================

def generate_image_and_mask(size=IMAGE_SIZE):
    image = np.zeros((size,size), dtype=np.float32)
    mask = np.zeros_like(image, dtype=np.uint8)

    # Cone parameters
    apex = (np.random.randint(size//4, 3*size//4),
            np.random.randint(size//4, 3*size//4))
    angle = np.random.uniform(0,360)
    opening = np.random.uniform(30, 60)
    sigma_pix = np.random.uniform(6,12)
    amp = np.random.uniform(0.5,1.0)
    asym = np.random.uniform(0.6,1.0)

    cone, cone_mask = add_hollow_cone(size, apex, angle, opening, sigma_pix, amp, asym)
    image += cone
    mask = np.maximum(mask, cone_mask)

    # Foreground stars
    image = add_foreground_stars(
        image,
        n_stars=np.random.randint(*N_STARS_RANGE),
        flux_range=STAR_FLUX_RANGE,
        fwhm_range=STAR_FWHM_RANGE
    )

    # Vignetting
    image = add_vignetting(image, strength=np.random.uniform(0.03, VIGNETTE_STRENGTH))

    # Affine jitter
    sx, sy = np.random.uniform(0.98, 1.02, 2)
    image = affine_transform(image, [[sx,0],[0,sy]], order=1, mode='reflect')
    mask = affine_transform(mask, [[sx,0],[0,sy]], order=0, mode='reflect')

    # Noise + hot pixels + cosmic rays
    image = add_noise(image, PHOTON_SCALE, READ_NOISE_SIGMA)
    image = add_hot_pixels(image, HOT_PIXEL_RATE)
    image = add_cosmic_rays(image, COSMIC_RAY_RATE)
    image = clip_saturation(image, SATURATION_LEVEL)

    return image.astype(np.float32), mask.astype(np.uint8)

# ================== MAIN LOOP ==================

if __name__ == "__main__":
    print(f"Generating {N_IMAGES} images into {OUT_DIR}...\n")
    preview_img = None
    preview_mask = None

    for i in range(N_IMAGES):
        img, msk = generate_image_and_mask()
        if i==0:
            preview_img = img
            preview_mask = msk

        np.save(OUT_DIR/f"synthetic_image_{i:04d}.npy", img)
        np.save(OUT_DIR/f"synthetic_mask_{i:04d}.npy", msk)
        if SAVE_FITS:
            fits.writeto(OUT_DIR/f"synthetic_image_{i:04d}.fits", img, overwrite=True)

        if (i+1)%10==0 or i==0:
            print(f"  - {i+1}/{N_IMAGES}")

    # Preview
    fig, axs = plt.subplots(1,2,figsize=(8,4))
    axs[0].imshow(preview_img, origin='lower', cmap='inferno')
    axs[0].set_title("Synthetic Image")
    axs[0].axis('off')
    axs[1].imshow(preview_mask, origin='lower', cmap='gray')
    axs[1].set_title("Mask")
    axs[1].axis('off')
    plt.tight_layout()
    preview_path = OUT_DIR/"preview.png"
    plt.savefig(preview_path, dpi=150)
    plt.close()
    print(f"\nPreview saved to {preview_path}")
