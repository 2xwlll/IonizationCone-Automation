#!/usr/bin/env python3
"""
two_component_fit.py (CLEAN + PHYSICALLY CONSISTENT VERSION)

Fixes:
- Removes undefined A matrix dependency
- Removes unstable polynomial background model
- Keeps continuum subtraction as primary background handling
- Makes cone fit fully emission-driven and reproducible
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares
from pathlib import Path
import json

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

O3_FILE   = "data/2d/MAST_2026-04-14T0111/HST/hst_5754_01_wfpc2_pc_f502n_u2m301/hst_5754_01_wfpc2_pc_f502n_u2m301_drz.fits"
CONT_FILE = "data/2d/MAST_continuum/mastDownload/HST/hst_5754_01_wfpc2_pc_f547m_u2m301/hst_5754_01_wfpc2_pc_f547m_u2m301_drz.fits"

OUT_DIR = Path("data/2d/ngc1068_two_component")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

PHOTFLAM_502   = 2.943716e-16
PHOTFLAM_547   = 7.595041e-18
PHOTFLAM_RATIO = PHOTFLAM_547 / PHOTFLAM_502

R_SCALE_FIXED = 35.0  # pixels (~1.6 arcsec)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

print("Loading FITS files...")

with fits.open(O3_FILE) as h:
    o3 = h[1].data.astype(np.float32)

with fits.open(CONT_FILE) as h:
    cont = h[1].data.astype(np.float32)

o3   = np.nan_to_num(o3)
cont = np.nan_to_num(cont)

valid = (o3 > 0) & (cont > 0)

# ─────────────────────────────────────────────
# FIND NUCLEUS
# ─────────────────────────────────────────────

cont_smooth = gaussian_filter(cont * valid, sigma=5)

edge = 50
cont_smooth[:edge,:] = cont_smooth[-edge:,:] = 0
cont_smooth[:,:edge] = cont_smooth[:,-edge:] = 0

ny, nx = np.unravel_index(np.argmax(cont_smooth), cont_smooth.shape)
print(f"Nucleus: {ny}, {nx}")

# ─────────────────────────────────────────────
# CROP
# ─────────────────────────────────────────────

CROP = 400
half = CROP // 2

r0, r1 = max(0, ny-half), min(o3.shape[0], ny+half)
c0, c1 = max(0, nx-half), min(o3.shape[1], nx+half)

o3_crop   = o3[r0:r1, c0:c1]
cont_crop = cont[r0:r1, c0:c1]
valid_crop = valid[r0:r1, c0:c1]

H, W = o3_crop.shape

nuc_cy = ny - r0
nuc_cx = nx - c0

yy, xx = np.indices((H, W))
r_grid = np.sqrt((xx-nuc_cx)**2 + (yy-nuc_cy)**2) + 1e-6
theta_grid = np.arctan2(yy-nuc_cy, xx-nuc_cx)

# ─────────────────────────────────────────────
# CONTINUUM SUBTRACTION
# ─────────────────────────────────────────────

PSF_MATCH_SIGMA = 0.5
cont_matched = gaussian_filter(cont_crop, sigma=PSF_MATCH_SIGMA)

sky_mask = (r_grid > 130) & (r_grid < 180) & valid_crop

scale_factor = PHOTFLAM_RATIO * (
    np.median(o3_crop[sky_mask]) /
    (np.median(cont_matched[sky_mask]) * PHOTFLAM_RATIO + 1e-12)
)

cont_scaled = cont_matched * scale_factor

emission_direct = o3_crop - cont_scaled

noise_sigma = np.std(emission_direct[sky_mask])
snr_map = emission_direct / (noise_sigma + 1e-12)

# ─────────────────────────────────────────────
# BICONE MODEL
# ─────────────────────────────────────────────

def bicone_model(params, r, theta):
    amp, theta0, dtheta, asym = params

    dphi1 = np.angle(np.exp(1j*(theta-theta0)))
    dphi2 = np.angle(np.exp(1j*(theta-(theta0+np.pi))))

    angular1 = np.exp(-(dphi1**2)/(2*dtheta**2))
    angular2 = np.exp(-(dphi2**2)/(2*dtheta**2))

    radial = np.where(
        r < R_SCALE_FIXED,
        (r/R_SCALE_FIXED)**2,
        (R_SCALE_FIXED/r)**1.5
    )

    radial *= np.exp(-(r/(6*R_SCALE_FIXED))**2)
    radial[r < 3] = 0

    return amp * radial * (angular1 + asym * angular2)

# ─────────────────────────────────────────────
# FIT MASK (CLEAN VERSION)
# ─────────────────────────────────────────────

NUC_EXCL_FIT = 20
SNR_CUT = 3.0

fit_mask = (
    valid_crop &
    (r_grid > NUC_EXCL_FIT) &
    (snr_map >= SNR_CUT)
)

print("Fit pixels:", fit_mask.sum())

data_flat = emission_direct.ravel()
mask_flat = fit_mask.ravel()

# ─────────────────────────────────────────────
# INITIAL GUESS (NO A MATRIX ANYMORE)
# ─────────────────────────────────────────────

theta0_init = np.radians(120)

params0 = np.array([
    np.percentile(emission_direct[fit_mask], 98),
    theta0_init,
    np.radians(35),
    0.4
])

lb = [0.0, np.radians(80), np.radians(25), 0.0]
ub = [np.inf, np.radians(160), np.radians(55), 1.0]

# ─────────────────────────────────────────────
# RESIDUALS
# ─────────────────────────────────────────────

def residuals(p):
    model = bicone_model(p, r_grid, theta_grid).ravel()
    return (data_flat - model)[mask_flat]

# ─────────────────────────────────────────────
# FIT
# ─────────────────────────────────────────────

print("Fitting cone...")

result = least_squares(
    residuals,
    params0,
    bounds=(lb, ub),
    method="trf",
    max_nfev=2000,
    verbose=1
)

best = result.x

print("\nRESULTS")
print("amp:", best[0])
print("PA:", np.degrees(best[1]))
print("opening:", np.degrees(best[2]))
print("asym:", best[3])

# ─────────────────────────────────────────────
# SAVE MINIMAL OUTPUT
# ─────────────────────────────────────────────

np.save(OUT_DIR/"emission.npy", emission_direct)

print("DONE → clean physically consistent version")
