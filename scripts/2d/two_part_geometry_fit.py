#!/usr/bin/env python3
"""
two_component_fit.py

Key change in this version:
    r_scale removed from fitted parameters — fixed to R_SCALE_FIXED=35px.
    Fitting r_scale caused the optimizer to either collapse to the nucleus
    (exponential model) or spread across the whole image (shallow power law).
    Fixing it at the known NGC 1068 NLR peak radius forces the model to fit
    angle, opening, amplitude and asymmetry correctly.

    Radial profile: broken power law r^2 inside, r^-3 outside,
    with exponential cutoff at 3*r_scale to kill emission beyond the NLR.
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
OUT_DIR   = Path("data/2d/ngc1068_two_component")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# PHYSICAL CONSTANTS
# ─────────────────────────────────────────────

PHOTFLAM_502   = 2.943716e-16
PHOTFLAM_547   = 7.595041e-18
PHOTFLAM_RATIO = PHOTFLAM_547 / PHOTFLAM_502

# NGC 1068 NLR peak radius fixed from literature
# ~1.6 arcsec at WFPC2/PC 0.046"/px = 35px
# Removing this from the fit prevents radial collapse
R_SCALE_FIXED  = 35.0   # pixels

# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────

print("Loading FITS files...")
with fits.open(O3_FILE) as h:
    o3   = h[1].data.astype(np.float32)
with fits.open(CONT_FILE) as h:
    cont = h[1].data.astype(np.float32)

o3    = np.nan_to_num(o3,   nan=0.0, posinf=0.0, neginf=0.0)
cont  = np.nan_to_num(cont, nan=0.0, posinf=0.0, neginf=0.0)
valid = (o3 > 0) & (cont > 0)

# ─────────────────────────────────────────────
# FIND NUCLEUS
# ─────────────────────────────────────────────

cont_smooth = gaussian_filter(cont * valid, sigma=5)
edge = 50
cont_smooth[:edge,:]  = cont_smooth[-edge:,:] = 0
cont_smooth[:,:edge]  = cont_smooth[:,-edge:] = 0
ny, nx = np.unravel_index(np.argmax(cont_smooth), cont_smooth.shape)
print(f"Nucleus: row={ny}  col={nx}")

# ─────────────────────────────────────────────
# CROP
# ─────────────────────────────────────────────

CROP = 400
half = CROP // 2
r0 = max(0, ny - half);  r1 = min(o3.shape[0], ny + half)
c0 = max(0, nx - half);  c1 = min(o3.shape[1], nx + half)

o3_crop    = o3[r0:r1,   c0:c1]
cont_crop  = cont[r0:r1, c0:c1]
valid_crop = valid[r0:r1, c0:c1]

H, W   = o3_crop.shape
nuc_cy = ny - r0
nuc_cx = nx - c0

yy, xx     = np.indices((H, W))
r_grid     = np.sqrt((xx - nuc_cx)**2 + (yy - nuc_cy)**2) + 1e-6
theta_grid = np.arctan2(yy - nuc_cy, xx - nuc_cx)

print(f"Crop: {o3_crop.shape}  nucleus at ({nuc_cx}, {nuc_cy})")

# ─────────────────────────────────────────────
# PSF MATCH
# ─────────────────────────────────────────────

PSF_MATCH_SIGMA = 0.5
cont_matched    = gaussian_filter(cont_crop, sigma=PSF_MATCH_SIGMA)
print(f"PSF matching: sigma={PSF_MATCH_SIGMA}px")

# ─────────────────────────────────────────────
# SCALE FACTOR
# ─────────────────────────────────────────────

sky_mask = (r_grid > 130) & (r_grid < 180) & valid_crop
print(f"Sky annulus pixels: {sky_mask.sum()}")

cont_phot_sky       = np.median(cont_matched[sky_mask]) * PHOTFLAM_RATIO
o3_sky              = np.median(o3_crop[sky_mask])
residual_correction = o3_sky / (cont_phot_sky + 1e-12)
scale_factor        = PHOTFLAM_RATIO * residual_correction

print(f"PHOTFLAM ratio:       {PHOTFLAM_RATIO:.6f}")
print(f"Residual correction:  {residual_correction:.4f}")
print(f"Final scale factor:   {scale_factor:.6f}")

if residual_correction < 0.5 or residual_correction > 2.0:
    print("WARNING: residual correction large — check sky annulus")

cont_scaled = cont_matched * scale_factor

# ─────────────────────────────────────────────
# EMISSION MAP + NOISE
# ─────────────────────────────────────────────

emission_direct = o3_crop - cont_scaled
noise_sigma     = np.std(emission_direct[sky_mask])
snr_map         = emission_direct / (noise_sigma + 1e-12)

print(f"Sky noise:  {noise_sigma:.6f}")
print(f"Peak SNR:   {snr_map.max():.1f}")

# ─────────────────────────────────────────────
# NORMALISATION HELPERS
# ─────────────────────────────────────────────

def norm_asinh(img, pct=97.0):
    img = np.clip(img, 0, None)
    sc  = np.percentile(img[img > 0], pct) if (img > 0).any() else 1.0
    return (np.arcsinh(img / (sc + 1e-8)) / np.arcsinh(1.0)).astype(np.float32)

def norm_pct(img, lo=1.0, hi=99.5):
    p_lo = np.percentile(img, lo)
    p_hi = np.percentile(img, hi)
    return np.clip(
        (img - p_lo) / (p_hi - p_lo + 1e-8), 0, 1
    ).astype(np.float32)

# ─────────────────────────────────────────────
# DIAGNOSTIC FIGURE
# ─────────────────────────────────────────────

fig_diag, axes_diag = plt.subplots(1, 3, figsize=(15, 5))
fig_diag.patch.set_facecolor("#0a0a0a")

axes_diag[0].imshow(norm_asinh(o3_crop),     origin="lower", cmap="gray")
axes_diag[0].set_title("F502N raw",          color="white", fontsize=9)
axes_diag[0].plot(nuc_cx, nuc_cy, "+",       color="cyan",  markersize=12)

axes_diag[1].imshow(norm_asinh(cont_scaled), origin="lower", cmap="gray")
axes_diag[1].set_title(
    f"F547M scaled ×{scale_factor:.5f}",     color="white", fontsize=9
)
axes_diag[1].plot(nuc_cx, nuc_cy, "+",       color="cyan",  markersize=12)

axes_diag[2].imshow(
    norm_asinh(np.clip(emission_direct, 0, None)),
    origin="lower", cmap="magma"
)
axes_diag[2].set_title(
    "Emission = F502N − F547M_scaled",       color="white", fontsize=9
)
axes_diag[2].plot(nuc_cx, nuc_cy, "+",       color="cyan",  markersize=12)

for ax in axes_diag:
    ax.axis("off")

plt.tight_layout()
plt.savefig(
    OUT_DIR / "diagnostic_emission.png", dpi=150,
    bbox_inches="tight", facecolor="#0a0a0a"
)
plt.close()
print("Diagnostic saved → diagnostic_emission.png")

# ─────────────────────────────────────────────
# FIT BICONE GEOMETRY
# ─────────────────────────────────────────────

def bicone_model(params, r, theta, min_r=3.0):
    """
    4 parameters: amp, theta0, dtheta, asym
    r_scale fixed at R_SCALE_FIXED (35px = ~1.6 arcsec for NGC 1068)

    Adjusted radial profile:
        - r < r_scale: rises as r^2 (suppressed near nucleus)
        - r >= r_scale: falls as r^-1.5 (shallower to capture the fan/ENLR)
        - cutoff at 6*r_scale: extended to prevent killing outer emission
    """
    amp, theta0, dtheta, asym = params

    dphi1    = np.angle(np.exp(1j * (theta - theta0)))
    dphi2    = np.angle(np.exp(1j * (theta - (theta0 + np.pi))))
    angular1 = np.exp(-(dphi1**2) / (2 * dtheta**2))
    angular2 = np.exp(-(dphi2**2) / (2 * dtheta**2))

    radial = np.where(
        r < R_SCALE_FIXED,
        (r / R_SCALE_FIXED)**2,
        (R_SCALE_FIXED / r)**1.5 # CHANGED: 1.5 instead of 3.0 for shallower falloff
    )
    
    # CHANGED: Extended cutoff to 6.0 * R_SCALE_FIXED
    radial *= np.exp(-(r / (6.0 * R_SCALE_FIXED))**2) 
    radial  = np.where(r < min_r, 0.0, radial)

    return amp * radial * (angular1 + asym * angular2)

def full_model(params):
    poly_params = params[:n_terms]
    cone_params = params[n_terms:]
    smooth      = (A @ poly_params).reshape(H, W)
    emission    = bicone_model(cone_params, r_grid, theta_grid)
    return smooth + emission

# ─────────────────────────────────────────────
# DEFINE FIT MASK
# ─────────────────────────────────────────────

NUC_EXCL_FIT = 20
SNR_CUT      = 3.0

fit_mask = (
    valid_crop
    & (r_grid > NUC_EXCL_FIT)
    & (snr_map >= SNR_CUT)
)

print(f"\nFit pixels (S/N >= {SNR_CUT}): {fit_mask.sum()}")

if fit_mask.sum() < 100:
    print("WARNING: very few fit pixels — try lowering SNR_CUT to 2.0")

data_flat = emission_direct.ravel()
mask_flat  = fit_mask.ravel()
idx_init   = np.where(mask_flat)[0]

# ─────────────────────────────────────────────
# INITIAL LINEAR SOLVE (smooth background component)
# ─────────────────────────────────────────────

c_init, _, _, _ = np.linalg.lstsq(
    A[idx_init],
    data_flat[idx_init],
    rcond=None
)

# ─────────────────────────────────────────────
# CONE INITIALIZATION
# ─────────────────────────────────────────────

theta0_init = np.radians(120.0)

cone_init = [
    float(np.percentile(emission_direct[fit_mask], 98)),  # amplitude guess
    theta0_init,                                           # PA guess
    np.radians(35.0),                                      # opening guess
    0.4                                                    # asymmetry guess
]

params0 = np.concatenate([c_init, cone_init])

# ─────────────────────────────────────────────
# BOUNDS (IMPORTANT: controls geometry bias)
# ─────────────────────────────────────────────

lb = (
    [-np.inf] * n_terms +
    [
        0.0,                    # amplitude ≥ 0
        np.radians(80.0),      # PA lower bound
        np.radians(25.0),      # opening lower bound (prevents spine collapse)
        0.0                    # asymmetry relaxed (FIXED from 0.1)
    ]
)

ub = (
    [+np.inf] * n_terms +
    [
        np.inf,
        np.radians(160.0),
        np.radians(55.0),
        1.0
    ]
)

# ─────────────────────────────────────────────
# RESIDUAL FUNCTION
# ─────────────────────────────────────────────

def residuals(params):
    model = full_model(params).ravel()
    return (data_flat - model)[mask_flat]

# ─────────────────────────────────────────────
# FIT
# ─────────────────────────────────────────────

print(f"\nFitting ({mask_flat.sum()} pixels)...")
print(f"  r_scale fixed at {R_SCALE_FIXED:.0f}px = {R_SCALE_FIXED*0.046:.2f} arcsec")

result = least_squares(
    residuals,
    params0,
    bounds=(lb, ub),
    method="trf",
    max_nfev=2000,
    ftol=1e-6,
    xtol=1e-6,
    verbose=1
)

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────

print(f"\nOptimiser: {result.message}")
print(f"  cost     = {result.cost:.4e}")
print(f"  nfev     = {result.nfev}")
print(f"  success  = {result.success}")

best = result.x
cone_best = best[n_terms:]

print("\nFitted cone parameters:")
print(f"  amplitude   = {cone_best[0]:.5f}")
print(f"  axis angle  = {np.degrees(cone_best[1]):.1f}°  (expected ~120°)")
print(f"  opening σ   = {np.degrees(cone_best[2]):.1f}°")
print(f"  r_scale     = {R_SCALE_FIXED:.0f}px (fixed)")
print(f"  asymmetry   = {cone_best[3]:.3f}  (0 = symmetric, 1 = max asym)")# ─────────────────────────────────────────────
# EXTRACT COMPONENTS
# ─────────────────────────────────────────────

smooth_fit   = (A @ best[:n_terms]).reshape(H, W).astype(np.float32)
bicone_fit   = bicone_model(cone_best, r_grid, theta_grid).astype(np.float32)
residual_map = (emission_direct - bicone_fit - smooth_fit).astype(np.float32)

nuc_mask     = r_grid < 15
emission_map = emission_direct.copy()
emission_map[nuc_mask]    = 0.0
emission_map[~valid_crop] = 0.0

# ─────────────────────────────────────────────
# TRAINING LABELS
# ─────────────────────────────────────────────

theta0_fit  = cone_best[1]
opening_fit = cone_best[2]

dphi1    = np.abs(np.angle(np.exp(1j * (theta_grid - theta0_fit))))
dphi2    = np.abs(np.angle(np.exp(1j * (theta_grid - (theta0_fit + np.pi)))))
dphi_min = np.minimum(dphi1, dphi2)

# CHANGED: NLR_R_MAX increased to 250 to cover the extended ionization cones
NLR_R_MAX      = 250
in_cone_angle  = dphi_min < (2.0 * opening_fit)
in_cone_radial = (r_grid > 15) & (r_grid < NLR_R_MAX)
cone_region    = in_cone_angle & in_cone_radial & valid_crop

snr_emission = emission_map / (noise_sigma + 1e-12)
# CHANGED: SNR cut lowered to 2.5 to include the diffuse outer 'fan'
MASK_SNR_CUT = 2.5

cone_mask = (
    cone_region
    & (snr_emission >= MASK_SNR_CUT)
).astype(np.float32)

soft_mask = np.where(
    cone_region,
    np.clip(snr_emission / 8.0, 0, 1),
    0.0
).astype(np.float32)

print(f"\nCone region pixels:  {cone_region.sum()}")
print(f"Cone mask pixels:    {int(cone_mask.sum())}  (S/N >= {MASK_SNR_CUT})")
# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────

np.save(OUT_DIR / "emission_map.npy",   emission_map)
np.save(OUT_DIR / "smooth_model.npy",   smooth_fit)
np.save(OUT_DIR / "bicone_model.npy",   bicone_fit)
np.save(OUT_DIR / "residual_map.npy",   residual_map)
np.save(OUT_DIR / "cone_mask.npy",      cone_mask)
np.save(OUT_DIR / "soft_mask.npy",      soft_mask)
np.save(OUT_DIR / "snr_map.npy",        snr_emission)
np.save(OUT_DIR / "cont_scaled.npy",    cont_scaled)

fits.writeto(str(OUT_DIR / "emission_map.fits"),  emission_map,  overwrite=True)
fits.writeto(str(OUT_DIR / "bicone_model.fits"),  bicone_fit,    overwrite=True)
fits.writeto(str(OUT_DIR / "smooth_model.fits"),  smooth_fit,    overwrite=True)
fits.writeto(str(OUT_DIR / "cone_mask.fits"),     cone_mask,     overwrite=True)
fits.writeto(str(OUT_DIR / "soft_mask.fits"),     soft_mask,     overwrite=True)
fits.writeto(str(OUT_DIR / "cont_scaled.fits"),   cont_scaled,   overwrite=True)

cone_params_out = {
    "amplitude":            float(cone_best[0]),
    "axis_angle_deg":       float(np.degrees(cone_best[1])),
    "opening_sigma_deg":    float(np.degrees(cone_best[2])),
    "r_scale_px":           R_SCALE_FIXED,
    "r_scale_arcsec":       R_SCALE_FIXED * 0.046,
    "r_scale_fixed":        True,
    "asymmetry":            float(cone_best[3]),
    "radial_model":         "broken_power_law_r2_r3_with_exp_cutoff",
    "photflam_ratio":       float(PHOTFLAM_RATIO),
    "residual_correction":  float(residual_correction),
    "scale_factor":         float(scale_factor),
    "psf_match_sigma":      float(PSF_MATCH_SIGMA),
    "noise_sigma":          float(noise_sigma),
    "snr_cut_fit":          float(SNR_CUT),
    "snr_cut_mask":         float(MASK_SNR_CUT),
    "nlr_r_max_px":         NLR_R_MAX,
    "cone_mask_pixels":     int(cone_mask.sum()),
    "optimiser_success":    bool(result.success),
    "nfev":                 int(result.nfev),
    "cost":                 float(result.cost),
}
with open(OUT_DIR / "cone_params.json", "w") as f:
    json.dump(cone_params_out, f, indent=2)

print(f"\nSaved to {OUT_DIR}/")

# ─────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────

emission_disp = emission_map.copy()
bicone_disp   = bicone_fit.copy()
bicone_disp[nuc_mask] = 0.0

fig, axes = plt.subplots(2, 3, figsize=(16, 11))
fig.patch.set_facecolor("#0a0a0a")

ax_deg = np.degrees(cone_best[1])
fig.suptitle(
    f"NGC 1068  |  cone={ax_deg:.1f}°  "
    f"opening={np.degrees(cone_best[2]):.1f}°σ  "
    f"r_scale={R_SCALE_FIXED:.0f}px FIXED  "
    f"asym={cone_best[3]:.2f}  "
    f"scale={scale_factor:.5f}",
    color="white", fontsize=10
)

def show(ax, img, title, cmap, mark_nuc=True, mark_circle=False):
    ax.imshow(img, origin="lower", cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title, color="white", fontsize=9)
    ax.axis("off")
    if mark_nuc:
        ax.plot(nuc_cx, nuc_cy, "+", color="cyan",
                markersize=12, markeredgewidth=1.5)
    if mark_circle:
        t = np.linspace(0, 2*np.pi, 100)
        ax.plot(nuc_cx + 15*np.cos(t), nuc_cy + 15*np.sin(t),
                color="cyan", lw=0.8, alpha=0.6)

show(axes[0,0], norm_pct(o3_crop),
     "F502N (data)",                              "gray")
show(axes[0,1], norm_pct(cont_scaled),
     f"F547M scaled ×{scale_factor:.5f}",         "gray")
show(axes[0,2], norm_pct(cont_crop),
     "F547M (raw)",                               "gray")
show(axes[1,0], norm_asinh(emission_disp),
     "Emission = F502N − F547M_scaled  [asinh]",  "magma", False, True)
show(axes[1,1], norm_pct(np.clip(bicone_disp, 0, None)),
     f"Fitted bicone  (r_scale={R_SCALE_FIXED:.0f}px fixed)",
     "magma", False, True)
show(axes[1,2], soft_mask,
     "Soft cone mask (training label)",            "inferno", True)

for ax in [axes[1,0], axes[1,1]]:
    for sign in [1, -1]:
        length = 80
        ax.annotate("",
            xy=(nuc_cx + sign * length * np.cos(cone_best[1]),
                nuc_cy + sign * length * np.sin(cone_best[1])),
            xytext=(nuc_cx, nuc_cy),
            arrowprops=dict(arrowstyle="->", color="yellow", lw=1.2))

plt.tight_layout()
out_png = OUT_DIR / "two_component_result.png"
plt.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="#0a0a0a")
plt.close()
print(f"Result → {out_png}")

print("\nDONE")
print("  Check:")
print("  - bicone panel: compact two-lobe structure, NOT a full-frame fan")
print("  - soft mask: two asymmetric lobes near nucleus")
print(f"  - fitted angle near 120° for NGC 1068")
print(f"  - if bicone still fans out, reduce R_SCALE_FIXED or")
print(f"    increase the exp cutoff coefficient (currently 3.0)")
