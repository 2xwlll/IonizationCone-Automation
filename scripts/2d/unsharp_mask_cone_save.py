#!/usr/bin/env python3
"""
cone_detection_psf_fixed.py

Improved ionization cone detection:
- continuum stacking
- local scaling
- PSF (radial symmetry) removal
- unbiased angular analysis
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from pathlib import Path

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

O3_FILE = "data/2d/MAST_2026-04-14T0111/HST/hst_5754_01_wfpc2_pc_f502n_u2m301/hst_5754_01_wfpc2_pc_f502n_u2m301_drz.fits"

CONT_FILES = [
"data/2d/MAST_continuum/mastDownload/HST/hst_5754_01_wfpc2_pc_f547m_u2m301/hst_5754_01_wfpc2_pc_f547m_u2m301_drz.fits",
"data/2d/MAST_continuum/mastDownload/HST/hst_5754_01_wfpc2_pc_f547m_u2m30101/hst_5754_01_wfpc2_pc_f547m_u2m30101_drz.fits",
"data/2d/MAST_continuum/mastDownload/HST/hst_5754_01_wfpc2_pc_f547m_u2m30102/hst_5754_01_wfpc2_pc_f547m_u2m30102_drz.fits",
]

OUT_DIR = Path("data/2d/cone_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

print("Loading emission image...")
with fits.open(O3_FILE) as h:
    o3 = np.nan_to_num(h[1].data.astype(np.float32))

print("Stacking continuum images...")
stack = []
for f in CONT_FILES:
    with fits.open(f) as h:
        img = np.nan_to_num(h[1].data.astype(np.float32))
        stack.append(img)

cont = np.nanmedian(stack, axis=0)

print("Shapes:", o3.shape, cont.shape)

# ─────────────────────────────────────────────
# FIND NUCLEUS
# ─────────────────────────────────────────────

cont_smooth = gaussian_filter(cont, sigma=5)

edge = 50
cont_smooth[:edge, :] = 0
cont_smooth[-edge:, :] = 0
cont_smooth[:, :edge] = 0
cont_smooth[:, -edge:] = 0

ny, nx = np.unravel_index(np.argmax(cont_smooth), cont_smooth.shape)
print(f"Nucleus at: ({ny}, {nx})")

# ─────────────────────────────────────────────
# DISTANCE + ANGLE MAP
# ─────────────────────────────────────────────

yy, xx = np.indices(o3.shape)
dx = xx - nx
dy = yy - ny

r = np.sqrt(dx**2 + dy**2)
theta = np.arctan2(dy, dx)

# ─────────────────────────────────────────────
# MASK CORE + VALID PIXELS
# ─────────────────────────────────────────────

core_mask = r < 40
valid = (o3 > 0) & (cont > 0) & (~core_mask)

# ─────────────────────────────────────────────
# LOCAL SCALE FACTOR
# ─────────────────────────────────────────────

annulus = (r > 80) & (r < 200) & valid
ratio = o3[annulus] / (cont[annulus] + 1e-8)
scale = np.median(ratio)

print(f"Scale factor: {scale:.4f}")

# ─────────────────────────────────────────────
# CONTINUUM SUBTRACTION
# ─────────────────────────────────────────────

emission = np.zeros_like(o3)
emission[valid] = o3[valid] - scale * cont[valid]

# sky subtraction
sky_region = (r > 250) & valid
sky = np.median(emission[sky_region])
emission -= sky

# soften nucleus
emission[r < 40] *= 0.1

# ─────────────────────────────────────────────
# REMOVE RADIAL (PSF-LIKE) COMPONENT
# ─────────────────────────────────────────────

print("Removing radial PSF component...")

max_r = int(np.max(r))
radial_profile = np.zeros(max_r)

for i in range(max_r - 1):
    mask = (r >= i) & (r < i + 1)
    if np.any(mask):
        radial_profile[i] = np.median(emission[mask])

psf_model = radial_profile[r.astype(int)]
emission -= psf_model

# ─────────────────────────────────────────────
# ANGULAR ANALYSIS (UNBIASED)
# ─────────────────────────────────────────────

analysis_mask = (r > 120) & (r < 300)

theta_vals = theta[analysis_mask]
flux_vals = emission[analysis_mask]

bins = np.linspace(-np.pi, np.pi, 180)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

angular_profile = np.zeros(len(bin_centers))

for i in range(len(bin_centers)):
    in_bin = (theta_vals >= bins[i]) & (theta_vals < bins[i+1])
    if np.any(in_bin):
        angular_profile[i] = np.median(flux_vals[in_bin])

# smooth profile
angular_profile = gaussian_filter1d(angular_profile, sigma=2)

# ─────────────────────────────────────────────
# FIND PEAKS
# ─────────────────────────────────────────────

peak_idx = np.argmax(angular_profile)
theta_peak = bin_centers[peak_idx]

theta_secondary = theta_peak + np.pi
if theta_secondary > np.pi:
    theta_secondary -= 2*np.pi

print(f"Primary cone angle: {theta_peak:.3f} rad")
print(f"Secondary cone angle: {theta_secondary:.3f} rad")

# contrast metric
peak_strength = np.max(angular_profile)
background = np.median(angular_profile)
contrast = peak_strength / (background + 1e-8)

print(f"Angular contrast: {contrast:.2f}")

# ─────────────────────────────────────────────
# PLOT ANGULAR PROFILE
# ─────────────────────────────────────────────

plt.figure(figsize=(8,5))
plt.plot(bin_centers, angular_profile)
plt.axvline(theta_peak, linestyle="--", label="Primary")
plt.axvline(theta_secondary, linestyle="--", label="Secondary")

plt.xlabel("Angle (radians)")
plt.ylabel("Median emission")
plt.title("Angular Emission Profile (PSF-subtracted)")
plt.legend()

plt.tight_layout()
plt.savefig(OUT_DIR / "angular_profile_psf.png", dpi=200)
plt.close()

# ─────────────────────────────────────────────
# EMISSION MAP
# ─────────────────────────────────────────────

plt.figure(figsize=(6,6))
vmax = np.percentile(emission, 99)

plt.imshow(np.clip(emission, 0, vmax), origin="lower", cmap="magma")
plt.plot(nx, ny, "+", color="cyan")

plt.title("PSF-subtracted Emission Map")
plt.axis("off")

plt.savefig(OUT_DIR / "emission_map_psf.png", dpi=200)
plt.close()

print("\nDone.")
print("Check angular_profile_psf.png and emission_map_psf.png")
