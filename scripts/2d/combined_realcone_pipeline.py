#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from pathlib import Path

# ─────────────────────────────────────────────
# FILES
# ─────────────────────────────────────────────

O3_FILE = "data/2d/MAST_2026-04-14T0111/HST/hst_5754_01_wfpc2_pc_f502n_u2m301/hst_5754_01_wfpc2_pc_f502n_u2m301_drz.fits"

CONT_FILE = "data/2d/MAST_continuum/mastDownload/HST/hst_5754_01_wfpc2_pc_f547m_u2m301/hst_5754_01_wfpc2_pc_f547m_u2m301_drz.fits"

OUT_DIR = Path("data/2d/cone_diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

with fits.open(O3_FILE) as h:
    o3 = np.nan_to_num(h[1].data.astype(np.float32))

with fits.open(CONT_FILE) as h:
    cont = np.nan_to_num(h[1].data.astype(np.float32))

ny, nx = o3.shape

# ─────────────────────────────────────────────
# NUCLEUS (from continuum only)
# ─────────────────────────────────────────────

cont_smooth = gaussian_filter(cont, sigma=5)

ny0, nx0 = np.unravel_index(np.argmax(cont_smooth), cont_smooth.shape)

print(f"Nucleus: {ny0}, {nx0}")

# ─────────────────────────────────────────────
# COORDINATES (lightweight)
# ─────────────────────────────────────────────

y = np.arange(ny)[:, None]
x = np.arange(nx)[None, :]

dy = y - ny0
dx = x - nx0

r = np.sqrt(dx**2 + dy**2)

theta = np.arctan2(dy, dx)

# ─────────────────────────────────────────────
# MINIMAL PREPROCESSING ONLY
# ─────────────────────────────────────────────

# IMPORTANT: no subtraction, only ratio normalization
eps = 1e-8
ratio = o3 / (cont + eps)

# mild smoothing only (preserve morphology)
ratio = gaussian_filter(ratio, sigma=1)

valid = (r > 20) & (r < 300)

# ─────────────────────────────────────────────
# ANGULAR STRUCTURE TEST
# ─────────────────────────────────────────────

theta_vals = theta[valid]
flux_vals = ratio[valid]

bins = np.linspace(-np.pi, np.pi, 180)
centers = 0.5 * (bins[:-1] + bins[1:])

profile = np.zeros(len(centers))

for i in range(len(centers)):
    m = (theta_vals >= bins[i]) & (theta_vals < bins[i+1])
    if np.any(m):
        profile[i] = np.mean(flux_vals[m])

profile = gaussian_filter(profile, sigma=2)

# ─────────────────────────────────────────────
# CONE TEST METRICS
# ─────────────────────────────────────────────

peak = centers[np.argmax(profile)]

# test for bipolar symmetry
opposite = peak + np.pi
if opposite > np.pi:
    opposite -= 2*np.pi

symmetry_score = np.abs(
    profile[np.argmax(profile)] - profile[np.argmin(profile)]
)

print(f"Primary angle: {peak:.3f}")
print(f"Opposite angle: {opposite:.3f}")
print(f"Symmetry score (lower = better cone): {symmetry_score:.4f}")

# ─────────────────────────────────────────────
# VISUALS
# ─────────────────────────────────────────────

plt.figure()
plt.imshow(np.log10(ratio + 1e-3), origin="lower", cmap="magma")
plt.plot(nx0, ny0, "+", color="cyan")
plt.title("Log(OIII / continuum)")
plt.colorbar()
plt.savefig(OUT_DIR / "ratio_map.png", dpi=200)
plt.close()

plt.figure()
plt.plot(centers, profile)
plt.axvline(peak, linestyle="--")
plt.title("Angular structure (ratio-based)")
plt.savefig(OUT_DIR / "angular_profile.png", dpi=200)
plt.close()

print("Saved to:", OUT_DIR)
