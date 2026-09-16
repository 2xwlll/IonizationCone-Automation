#!/usr/bin/env python3
"""
continuum_subtraction.py

Implements the double-component fitting approach recommended by mentor:

  F502N image = smooth_continuum(x,y) + ionization_cone_emission(x,y)

Key insight:
  - Continuum IS smooth → model as 2D polynomial fitted to the data
  - Ionization cone is NOT smooth → it's what remains after removing smooth component
  - We fit the smooth model to regions that should NOT have cone emission
  - Iteratively exclude bright residuals (cone emission) from the fit

This is physically motivated — we're not assuming the F547M perfectly
traces the continuum under F502N. We let the data tell us what's smooth.

Method:
  1. Load F502N
  2. Fit 2D polynomial to "background" regions (masking nucleus + candidate cone)
  3. Subtract smooth model → residual = pure emission
  4. Iterate: refit excluding pixels with significant positive residuals
  5. Final emission map = F502N - best_fit_smooth
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares
from pathlib import Path

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

O3_FILE   = "data/2d/MAST_2026-04-14T0111/HST/hst_5754_01_wfpc2_pc_f502n_u2m301/hst_5754_01_wfpc2_pc_f502n_u2m301_drz.fits"
CONT_FILE = "data/2d/MAST_continuum/mastDownload/HST/hst_5754_01_wfpc2_pc_f547m_u2m301/hst_5754_01_wfpc2_pc_f547m_u2m301_drz.fits"
OUT_DIR   = Path("data/2d/ngc1068_emission")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────

print("Loading FITS files...")
with fits.open(O3_FILE) as h:
    o3     = h[1].data.astype(np.float32)
    hdr_o3 = h[1].header
with fits.open(CONT_FILE) as h:
    cont     = h[1].data.astype(np.float32)
    hdr_cont = h[1].header

o3   = np.nan_to_num(o3,   nan=0.0, posinf=0.0, neginf=0.0)
cont = np.nan_to_num(cont, nan=0.0, posinf=0.0, neginf=0.0)
valid = (o3 > 0) & (cont > 0)
print(f"  F502N: {o3.shape}")

# ─────────────────────────────────────────────
# FIND NUCLEUS (already verified correct)
# ─────────────────────────────────────────────

cont_smooth = gaussian_filter(cont * valid, sigma=5)
edge = 50
cont_smooth[:edge,:] = cont_smooth[-edge:,:] = 0
cont_smooth[:,:edge] = cont_smooth[:,-edge:] = 0
ny, nx = np.unravel_index(np.argmax(cont_smooth), cont_smooth.shape)
print(f"Nucleus: row={ny}  col={nx}")

# ─────────────────────────────────────────────
# CROP TO WORKING REGION
# NGC 1068 NLR extends ~15" = ~326px at WFPC2/PC 0.046"/px
# Use generous 400px crop to have enough background for fitting
# ─────────────────────────────────────────────

CROP = 400
half = CROP // 2
r0 = max(0, ny - half);  r1 = min(o3.shape[0], ny + half)
c0 = max(0, nx - half);  c1 = min(o3.shape[1], nx + half)

o3_crop   = o3[r0:r1, c0:c1]
cont_crop = cont[r0:r1, c0:c1]
valid_crop = valid[r0:r1, c0:c1]

nuc_cy = ny - r0
nuc_cx = nx - c0
H, W   = o3_crop.shape
print(f"Crop: {o3_crop.shape}  nucleus at ({nuc_cx}, {nuc_cy})")

# ─────────────────────────────────────────────
# 2D POLYNOMIAL BASIS FUNCTIONS
#
# We model the smooth continuum as a 2D polynomial of degree N.
# Higher degree captures more of the galaxy's radial structure.
# Degree 4 is usually sufficient for a bulge+disk profile.
# ─────────────────────────────────────────────

POLY_DEGREE = 4

def poly2d_basis(shape, degree):
    """
    Build a matrix of 2D polynomial basis functions.
    Returns (H*W, n_terms) design matrix.
    """
    H, W   = shape
    yy, xx = np.indices((H, W))

    # Normalise coordinates to [-1, 1] for numerical stability
    x_n = (xx - W/2) / (W/2)
    y_n = (yy - H/2) / (H/2)

    cols = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            cols.append((x_n**i * y_n**j).ravel())

    return np.column_stack(cols)   # (H*W, n_terms)

print(f"\nBuilding degree-{POLY_DEGREE} 2D polynomial basis...")
A = poly2d_basis((H, W), POLY_DEGREE)
n_terms = A.shape[1]
print(f"  Basis has {n_terms} terms")

# ─────────────────────────────────────────────
# ITERATIVE SMOOTH CONTINUUM FIT
#
# Algorithm (mentor's recommendation):
#   1. Start with initial mask: exclude nucleus + chip gaps
#   2. Fit 2D polynomial to unmasked pixels (least squares)
#   3. Compute residuals = data - model
#   4. Exclude pixels with large positive residuals (cone emission!)
#   5. Refit. Repeat until convergence.
#
# This is "sigma-clipping" adapted for one-sided exclusion:
# we only clip positive residuals (emission), not negative ones
# (those are just noise/dust lanes, not cone signal).
# ─────────────────────────────────────────────

print("\nIterative smooth continuum fitting...")

NUC_EXCL_R   = 20    # exclude nucleus region (px) — unresolved PSF
N_ITER       = 6     # number of sigma-clip iterations
CLIP_SIGMA   = 2.5   # clip positive residuals above this many sigma

# Coordinate arrays for distance from nucleus
yy_c, xx_c = np.indices((H, W))
r_crop     = np.sqrt((xx_c - nuc_cx)**2 + (yy_c - nuc_cy)**2)

# Initial fit mask: valid pixels only, nucleus excluded
fit_mask = valid_crop.copy() & (r_crop > NUC_EXCL_R)

data_flat = o3_crop.ravel()

for iteration in range(N_ITER):
    idx = np.where(fit_mask.ravel())[0]

    if len(idx) < n_terms * 2:
        print(f"  iter {iteration+1}: too few pixels ({len(idx)}) — stopping")
        break

    # Weighted least squares: solve A[idx] @ coeffs ≈ data[idx]
    A_sub   = A[idx]
    d_sub   = data_flat[idx]

    # Use numpy lstsq (robust, no need for scipy here)
    coeffs, residuals_ls, rank, sv = np.linalg.lstsq(A_sub, d_sub, rcond=None)

    # Evaluate model on full image
    model_flat = A @ coeffs
    resid_flat = data_flat - model_flat

    # One-sided sigma clip: only exclude positive residuals (emission regions)
    resid_in_mask = resid_flat[idx]
    resid_std     = resid_in_mask.std()
    resid_med     = np.median(resid_in_mask)

    # New mask: keep pixels where residual is not a large positive outlier
    new_mask_flat = fit_mask.ravel().copy()
    new_mask_flat[idx] = resid_in_mask < (resid_med + CLIP_SIGMA * resid_std)
    fit_mask = new_mask_flat.reshape(H, W)

    n_masked = fit_mask.sum()
    n_clipped = (~new_mask_flat.reshape(H,W) & valid_crop & (r_crop > NUC_EXCL_R)).sum()
    print(f"  iter {iteration+1}: {len(idx)} → {n_masked} pixels  "
          f"({n_clipped} clipped as emission)  "
          f"resid std={resid_std:.5f}")

# Final model
smooth_model = model_flat.reshape(H, W).astype(np.float32)
smooth_model = np.clip(smooth_model, 0, None)   # no negative continuum

print(f"\nFit complete. Model range: {smooth_model.min():.4f} – {smooth_model.max():.4f}")

# ─────────────────────────────────────────────
# EMISSION MAP = DATA - SMOOTH MODEL
# ─────────────────────────────────────────────

emission = np.zeros_like(o3_crop)
emission[valid_crop] = o3_crop[valid_crop] - smooth_model[valid_crop]

# Residual sky offset from annular region
sky_ann = valid_crop & (r_crop > 150) & (r_crop < 200)
sky     = np.median(emission[sky_ann]) if sky_ann.sum() > 50 else 0.0
emission -= sky
print(f"Residual sky removed: {sky:.6f}")

# ─────────────────────────────────────────────
# NUCLEUS MASK FOR DISPLAY
# (standard in NLR studies — PSF wings dominate inner ~15px)
# ─────────────────────────────────────────────

NUC_MASK_R  = 15
nuc_mask    = r_crop < NUC_MASK_R
em_masked   = emission.copy()
em_masked[nuc_mask] = 0.0

# ─────────────────────────────────────────────
# SNR CHECK
# ─────────────────────────────────────────────

nlr_band  = (r_crop > NUC_MASK_R) & (r_crop < 100) & valid_crop
bg_band   = (r_crop > 160) & (r_crop < 200) & valid_crop
snr       = (np.percentile(emission[nlr_band], 95)
             / (emission[bg_band].std() + 1e-8))
print(f"NLR SNR: {snr:.1f}  {'✓ detected' if snr > 3 else '⚠ check subtraction'}")

# ─────────────────────────────────────────────
# NORMALISE
# ─────────────────────────────────────────────

def norm_pct(img, lo=1.0, hi=99.5):
    p_lo = np.percentile(img, lo)
    p_hi = np.percentile(img, hi)
    return np.clip((img - p_lo) / (p_hi - p_lo + 1e-8), 0, 1).astype(np.float32)

def norm_asinh(img, pct=98.0):
    img = np.clip(img, 0, None)
    sc  = np.percentile(img[img > 0], pct) if (img > 0).any() else 1.0
    return (np.arcsinh(img / (sc + 1e-8)) / np.arcsinh(1.0)).astype(np.float32)

o3_disp      = norm_pct(o3_crop)
cont_disp    = norm_pct(cont_crop)
model_disp   = norm_pct(smooth_model)
em_lin       = norm_pct(emission)
em_asinh     = norm_asinh(em_masked, pct=97.0)
em_asinh_s   = norm_asinh(em_masked, pct=99.0)
em_ml        = norm_pct(np.clip(em_masked, 0, None), lo=0.0, hi=99.5)

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────

np.save(OUT_DIR / "emission_ml.npy",     em_ml)
np.save(OUT_DIR / "emission_masked.npy", em_masked)
np.save(OUT_DIR / "emission_raw.npy",    emission)
np.save(OUT_DIR / "smooth_model.npy",    smooth_model)
fits.writeto(str(OUT_DIR / "emission_ml.fits"),     em_ml,       overwrite=True)
fits.writeto(str(OUT_DIR / "emission_masked.fits"), em_masked,   overwrite=True)
fits.writeto(str(OUT_DIR / "smooth_model.fits"),    smooth_model,overwrite=True)
print(f"Saved to {OUT_DIR}/")

# ─────────────────────────────────────────────
# FIGURE — 3×3 showing full pipeline
# ─────────────────────────────────────────────

fig, axes = plt.subplots(3, 3, figsize=(16, 16))
fig.patch.set_facecolor("#0a0a0a")
fig.suptitle(
    f"NGC 1068  [OIII] Double-Component Continuum Fit\n"
    f"2D poly degree={POLY_DEGREE}, {N_ITER} iterations, "
    f"clip={CLIP_SIGMA}σ  |  NLR SNR≈{snr:.0f}",
    color="white", fontsize=12
)

# Row 0: inputs
for ax, img, title, cmap in [
    (axes[0,0], o3_disp,    "F502N  (emission + continuum)", "gray"),
    (axes[0,1], cont_disp,  "F547M  (reference continuum)",  "gray"),
    (axes[0,2], model_disp, f"Fitted smooth model (poly {POLY_DEGREE})", "gray"),
]:
    ax.imshow(img, origin="lower", cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title, color="white", fontsize=9)
    ax.axis("off")
    ax.plot(nuc_cx, nuc_cy, "+", color="cyan", markersize=12, markeredgewidth=1.5)

# Show what pixels were used in the final fit
fit_display = np.zeros((*o3_crop.shape, 4))
fit_display[fit_mask, 1] = 0.3   # green channel for fit pixels
fit_display[~fit_mask & valid_crop & (r_crop > NUC_EXCL_R), 0] = 0.4  # red = clipped (emission)
axes[0,2].imshow(fit_display, origin="lower")
axes[0,2].set_title(f"Fit mask (green=used, red=emission clipped)",
                    color="white", fontsize=9)

# Row 1: subtraction results
for ax, img, title, cmap in [
    (axes[1,0], em_lin,    "Emission  [linear, nucleus visible]",  "inferno"),
    (axes[1,1], em_asinh,  "Emission  [asinh p97, NUC MASKED]",    "magma"),
    (axes[1,2], em_asinh_s,"Emission  [asinh p99, NUC MASKED]",    "magma"),
]:
    ax.imshow(img, origin="lower", cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title, color="white", fontsize=9)
    ax.axis("off")
    if "MASKED" in title:
        theta_r = np.linspace(0, 2*np.pi, 100)
        ax.plot(nuc_cx + NUC_MASK_R*np.cos(theta_r),
                nuc_cy + NUC_MASK_R*np.sin(theta_r),
                color="cyan", lw=0.8, alpha=0.7)
    else:
        ax.plot(nuc_cx, nuc_cy, "+", color="cyan", markersize=12, markeredgewidth=1.5)

# Row 2: ML-ready + residual diagnostics
axes[2,0].imshow(em_ml, origin="lower", cmap="magma", vmin=0, vmax=1)
axes[2,0].set_title("Emission  [ML-ready, clipped + normalised]", color="white", fontsize=9)
axes[2,0].axis("off")
theta_r = np.linspace(0, 2*np.pi, 100)
axes[2,0].plot(nuc_cx + NUC_MASK_R*np.cos(theta_r),
               nuc_cy + NUC_MASK_R*np.sin(theta_r),
               color="cyan", lw=0.8, alpha=0.7)

# Residual histogram — should be roughly Gaussian centered near 0
bg_resid = emission[bg_band].ravel()
axes[2,1].hist(bg_resid, bins=80, color="#44ccff", alpha=0.8, density=True)
axes[2,1].axvline(0, color="white", lw=1, ls="--")
axes[2,1].axvline(np.median(bg_resid), color="yellow", lw=1,
                  label=f"median={np.median(bg_resid):.4f}")
axes[2,1].set_title("Background residual distribution\n(should be ~Gaussian centred at 0)",
                    color="white", fontsize=9)
axes[2,1].set_facecolor("#111111")
axes[2,1].tick_params(colors="white")
axes[2,1].spines[:].set_color("#444444")
axes[2,1].legend(fontsize=8, labelcolor="white", facecolor="#222222")

# Model vs data scatter (sample)
sample_idx = np.where(valid_crop.ravel())[0][::200]
axes[2,2].scatter(
    smooth_model.ravel()[sample_idx],
    o3_crop.ravel()[sample_idx],
    s=0.5, alpha=0.3, color="#44ccff"
)
lim = max(smooth_model.max(), o3_crop[valid_crop].max()) * 1.05
axes[2,2].plot([0, lim], [0, lim], "w--", lw=0.8, label="1:1 line")
axes[2,2].set_xlabel("Smooth model", color="white", fontsize=8)
axes[2,2].set_ylabel("F502N data", color="white", fontsize=8)
axes[2,2].set_title("Model vs data\n(points above 1:1 = excess emission)",
                    color="white", fontsize=9)
axes[2,2].set_facecolor("#111111")
axes[2,2].tick_params(colors="white")
axes[2,2].spines[:].set_color("#444444")
axes[2,2].legend(fontsize=8, labelcolor="white", facecolor="#222222")

plt.tight_layout()
plt.savefig(OUT_DIR / "continuum_result.png", dpi=200,
            bbox_inches="tight", facecolor="#0a0a0a")
plt.close()
print(f"\nResult → {OUT_DIR}/continuum_result.png")
print("\n✓ DONE")
print("  Key panels to check:")
print("  [0,2] Fit mask — red pixels are where cone emission was detected and excluded from fit")
print("  [1,1] Emission asinh p97 — most aggressive stretch, faintest cone structure")
print("  [2,1] Residual histogram — should be ~Gaussian centred near 0 if subtraction is good")
print("  [2,2] Model vs data — points above 1:1 line are emission regions (the cone)")
