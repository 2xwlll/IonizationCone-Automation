#!/usr/bin/env python3
"""
hybrid_cone_extraction.py

GOAL:
Extract ionized gas ([O III]) emission for cone detection + ML training.

METHOD:
1. Use continuum image (F547M) → spectral separation
2. Compute robust scale in safe region
3. Subtract continuum
4. Remove smooth residual structure
5. Output clean emission map

This fixes:
- over-subtraction
- spiral arm contamination
- wrong cone direction
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from pathlib import Path

# ─────────────────────────────────────────────
# PATHS (YOUR REAL FILES)
# ─────────────────────────────────────────────

O3_FILE = "data/2d/MAST_2026-04-14T0111/HST/hst_5754_01_wfpc2_pc_f502n_u2m301/hst_5754_01_wfpc2_pc_f502n_u2m301_drz.fits"

CONT_FILE = "data/2d/MAST_continuum/mastDownload/HST/hst_5754_01_wfpc2_pc_f547m_u2m301/hst_5754_01_wfpc2_pc_f547m_u2m301_drz.fits"

OUT_DIR = Path("data/2d/hybrid_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────

print("Loading data...")

with fits.open(O3_FILE) as h:
    o3 = np.nan_to_num(h[1].data.astype(np.float32))

with fits.open(CONT_FILE) as h:
    cont = np.nan_to_num(h[1].data.astype(np.float32))

valid = (o3 > 0) & (cont > 0)

print("Shape:", o3.shape)

# ─────────────────────────────────────────────
# FIND NUCLEUS (robust)
# ─────────────────────────────────────────────

cont_smooth = gaussian_filter(cont * valid, sigma=5)

edge = 50
cont_smooth[:edge,:] = 0
cont_smooth[-edge:,:] = 0
cont_smooth[:,:edge] = 0
cont_smooth[:,-edge:] = 0

ny, nx = np.unravel_index(np.argmax(cont_smooth), cont_smooth.shape)

print(f"Nucleus: {ny}, {nx}")

# ─────────────────────────────────────────────
# CROP (CRITICAL for memory + correctness)
# ─────────────────────────────────────────────

CROP = 600
half = CROP // 2

r0 = max(0, ny - half)
r1 = min(o3.shape[0], ny + half)
c0 = max(0, nx - half)
c1 = min(o3.shape[1], nx + half)

o3 = o3[r0:r1, c0:c1]
cont = cont[r0:r1, c0:c1]
valid = valid[r0:r1, c0:c1]

ny -= r0
nx -= c0

H, W = o3.shape
print("Crop:", o3.shape)

# ─────────────────────────────────────────────
# COORDINATES (memory safe)
# ─────────────────────────────────────────────

y = np.arange(H, dtype=np.float32)
x = np.arange(W, dtype=np.float32)

dy = y[:, None] - ny
dx = x[None, :] - nx

r = np.sqrt(dx**2 + dy**2)

# ─────────────────────────────────────────────
# 🔥 ROBUST SCALE (THIS FIXES EVERYTHING)
# ─────────────────────────────────────────────

print("Computing scale...")

safe_mask = (
    valid &
    (r > 50) & (r < 200)
)

ratio = o3[safe_mask] / (cont[safe_mask] + 1e-8)

# remove garbage
lo, hi = np.percentile(ratio, [10, 90])
ratio = ratio[(ratio > lo) & (ratio < hi)]

scale = float(np.median(ratio))

# sanity clamp (prevents destruction)
scale = np.clip(scale, 0.3, 1.2)

print(f"Scale: {scale:.4f}")

# ─────────────────────────────────────────────
# CONTINUUM SUBTRACTION
# ─────────────────────────────────────────────

emission = o3 - scale * cont

# remove sky offset
sky_mask = (r > 200) & (r < 250)
sky = np.median(emission[sky_mask]) if np.any(sky_mask) else 0.0
emission -= sky

# ─────────────────────────────────────────────
# REMOVE SMOOTH RESIDUAL STRUCTURE
# (kills spiral arms, keeps cone)
# ─────────────────────────────────────────────

smooth = gaussian_filter(emission, sigma=12)
emission -= smooth

# keep only physical emission
emission = np.clip(emission, 0, None)

# ─────────────────────────────────────────────
# MASK NUCLEUS (PSF dominated)
# ─────────────────────────────────────────────

NUC_R = 20
emission[r < NUC_R] = 0

# ─────────────────────────────────────────────
# NORMALIZATION (ML-ready)
# ─────────────────────────────────────────────

def norm(img):
    hi = np.percentile(img, 99.5)
    return np.clip(img / (hi + 1e-8), 0, 1)

em_ml = norm(emission)

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────

np.save(OUT_DIR / "emission_ml.npy", em_ml)
fits.writeto(OUT_DIR / "emission_ml.fits", em_ml, overwrite=True)

# ─────────────────────────────────────────────
# PLOTS (SAFE — DOWNSAMPLED)
# ─────────────────────────────────────────────

print("Saving outputs...")

step = 2
em_small = em_ml[::step, ::step]

plt.figure(figsize=(6,6))
plt.imshow(em_small, origin="lower", cmap="magma", vmin=0, vmax=1)
plt.plot(nx/step, ny/step, "+", color="cyan")
plt.title("Ionized Gas (Hybrid Method)")
plt.axis("off")
plt.savefig(OUT_DIR / "emission.png", dpi=200)
plt.close()

# comparison plot
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(o3[::step,::step], origin="lower", cmap="gray")
plt.title("F502N")

plt.subplot(1,3,2)
plt.imshow(cont[::step,::step], origin="lower", cmap="gray")
plt.title("F547M")

plt.subplot(1,3,3)
plt.imshow(em_small, origin="lower", cmap="magma")
plt.title("Extracted Emission")

for ax in plt.gcf().axes:
    ax.axis("off")

plt.savefig(OUT_DIR / "comparison.png", dpi=200)
plt.close()

print("\nDone.")
print("Output:", OUT_DIR)
