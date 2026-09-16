#!/usr/bin/env python3
"""
two_component_fit.py

Model:
    F502N(x,y) = smooth_continuum(x,y) + bicone_emission(x,y)

- Continuum = low-order 2D polynomial (smooth by construction)
- Bicone    = two opposite directional Gaussian lobes with radial falloff
              (physically correct for NGC 1068 Type 2 AGN geometry)

F547M used only for validation, not subtraction.
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
OUT_DIR   = Path("data/2d/ngc1068_two_component")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────

print("Loading FITS files...")
with fits.open(O3_FILE) as h:
    o3   = h[1].data.astype(np.float32)
with fits.open(CONT_FILE) as h:
    cont = h[1].data.astype(np.float32)

o3   = np.nan_to_num(o3,   nan=0.0, posinf=0.0, neginf=0.0)
cont = np.nan_to_num(cont, nan=0.0, posinf=0.0, neginf=0.0)
valid = (o3 > 0) & (cont > 0)

# ─────────────────────────────────────────────
# FIND NUCLEUS
# ─────────────────────────────────────────────

cont_smooth = gaussian_filter(cont * valid, sigma=5)
edge = 50
cont_smooth[:edge,:] = cont_smooth[-edge:,:] = 0
cont_smooth[:,:edge] = cont_smooth[:,-edge:] = 0
ny, nx = np.unravel_index(np.argmax(cont_smooth), cont_smooth.shape)
print(f"Nucleus: row={ny}  col={nx}")

# ─────────────────────────────────────────────
# CROP
# ─────────────────────────────────────────────

CROP = 400
half = CROP // 2
r0 = max(0, ny - half);  r1 = min(o3.shape[0], ny + half)
c0 = max(0, nx - half);  c1 = min(o3.shape[1], nx + half)

o3_crop   = o3[r0:r1, c0:c1]
cont_crop = cont[r0:r1, c0:c1]
valid_crop = valid[r0:r1, c0:c1]

H, W    = o3_crop.shape
nuc_cy  = ny - r0
nuc_cx  = nx - c0

yy, xx  = np.indices((H, W))
r_grid  = np.sqrt((xx - nuc_cx)**2 + (yy - nuc_cy)**2) + 1e-6
theta_grid = np.arctan2(yy - nuc_cy, xx - nuc_cx)

print(f"Crop: {o3_crop.shape}  nucleus at ({nuc_cx}, {nuc_cy})")

# ─────────────────────────────────────────────
# POLYNOMIAL BASIS
# Degree 3 is enough — we want smooth, not flexible
# ─────────────────────────────────────────────

def poly2d_basis(shape, degree=3):
    H, W   = shape
    yy_, xx_ = np.indices((H, W))
    x = (xx_ - W/2) / (W/2)
    y = (yy_ - H/2) / (H/2)
    cols = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            cols.append((x**i * y**j).ravel())
    return np.column_stack(cols)

A       = poly2d_basis((H, W), degree=3)
n_terms = A.shape[1]
print(f"Polynomial basis: {n_terms} terms")

# ─────────────────────────────────────────────
# BICONE MODEL
#
# Two opposite lobes centred on nucleus:
#   lobe 1: axis angle theta0
#   lobe 2: axis angle theta0 + pi  (opposite direction)
#
# Each lobe:
#   angular profile: Gaussian in angle from cone axis
#   radial profile:  exponential falloff from nucleus
#   asymmetry: second lobe can be dimmer (torus obscuration)
#
# Parameters:
#   amp     — peak brightness of primary lobe
#   theta0  — cone axis angle (radians)
#   dtheta  — half-opening angle (radians, Gaussian sigma)
#   r_scale — exponential scale length (pixels)
#   asym    — secondary lobe amplitude fraction (0-1)
# ─────────────────────────────────────────────

def bicone_model(params, r, theta, min_r=3.0):
    amp, theta0, dtheta, r_scale, asym = params

    # Angular distance from each cone axis
    dphi1 = np.angle(np.exp(1j * (theta - theta0)))
    dphi2 = np.angle(np.exp(1j * (theta - (theta0 + np.pi))))

    angular1 = np.exp(-(dphi1**2) / (2 * dtheta**2))
    angular2 = np.exp(-(dphi2**2) / (2 * dtheta**2))

    radial   = np.exp(-r / r_scale)

    # Zero out unresolved nucleus (< min_r px)
    radial   = np.where(r < min_r, 0.0, radial)

    return amp * radial * (angular1 + asym * angular2)

# ─────────────────────────────────────────────
# FULL MODEL: polynomial + bicone
# ─────────────────────────────────────────────

def full_model(params):
    poly_params = params[:n_terms]
    cone_params = params[n_terms:]
    smooth   = (A @ poly_params).reshape(H, W)
    emission = bicone_model(cone_params, r_grid, theta_grid)
    return smooth + emission

# ─────────────────────────────────────────────
# INITIAL GUESS
#
# For NGC 1068 the ionization cone axis is well known:
#   PA ≈ 30° (NE-SW), which is ~30° from E = 60° from x-axis
#   in standard image coordinates (origin lower-left)
# ─────────────────────────────────────────────

NUC_EXCL_R = 20
fit_mask   = valid_crop & (r_grid > NUC_EXCL_R)
data_flat  = o3_crop.ravel()
mask_flat  = fit_mask.ravel()

# Polynomial init: least squares fit to data with no cone term
idx_init   = np.where(mask_flat)[0]
c_init, _, _, _ = np.linalg.lstsq(A[idx_init], data_flat[idx_init], rcond=None)

# NGC 1068 cone axis PA ~30° → theta in image coords ~60° from +x
theta0_init = np.radians(60.0)

cone_init = [
    float(np.percentile(o3_crop[fit_mask], 98)),  # amp
    theta0_init,                                    # theta0
    np.radians(20.0),                               # dtheta (opening ~20° sigma)
    60.0,                                           # r_scale (px)
    0.5,                                            # asym
]

params0 = np.concatenate([c_init, cone_init])

# ─────────────────────────────────────────────
# BOUNDS
# Polynomial: unconstrained
# Cone: physical bounds only
# ─────────────────────────────────────────────

n_cone  = len(cone_init)
lb      = [-np.inf] * n_terms + [0.0,    -np.pi, np.radians(5),  5.0,  0.0]
ub      = [+np.inf] * n_terms + [np.inf, +np.pi, np.radians(60), 300.0, 1.0]

# ─────────────────────────────────────────────
# RESIDUAL FUNCTION
# ─────────────────────────────────────────────

def residuals(params):
    model = full_model(params).ravel()
    return (data_flat - model)[mask_flat]

# ─────────────────────────────────────────────
# FIT
# ─────────────────────────────────────────────

print(f"\nFitting two-component model ({mask_flat.sum()} pixels)...")
print("  This may take 30-120 seconds...")

result = least_squares(
    residuals,
    params0,
    bounds=(lb, ub),
    method="trf",          # Trust Region Reflective — handles bounds well
    max_nfev=2000,         # enough for convergence (was 60 — far too low)
    ftol=1e-6,
    xtol=1e-6,
    verbose=1
)

print(f"\nOptimiser: {result.message}")
print(f"  cost={result.cost:.4e}  nfev={result.nfev}  success={result.success}")

best       = result.x
cone_best  = best[n_terms:]
print(f"\nFitted cone parameters:")
print(f"  amplitude  = {cone_best[0]:.4f}")
print(f"  axis angle = {np.degrees(cone_best[1]):.1f}°")
print(f"  opening σ  = {np.degrees(cone_best[2]):.1f}°  (half-angle, Gaussian σ)")
print(f"  r_scale    = {cone_best[3]:.1f} px  "
      f"({cone_best[3]*0.046:.2f} arcsec at WFPC2/PC 0.046\"/px)")
print(f"  asymmetry  = {cone_best[4]:.3f}  (counter-cone fraction)")

# ─────────────────────────────────────────────
# EXTRACT COMPONENTS
# ─────────────────────────────────────────────

smooth_fit   = (A @ best[:n_terms]).reshape(H, W).astype(np.float32)
bicone_fit   = bicone_model(cone_best, r_grid, theta_grid).astype(np.float32)
residual_map = (o3_crop - smooth_fit - bicone_fit).astype(np.float32)
emission_map = (o3_crop - smooth_fit).astype(np.float32)   # data minus continuum

# Mask nucleus for display
nuc_disp_mask = r_grid < 15
emission_disp = emission_map.copy();  emission_disp[nuc_disp_mask] = 0.0
bicone_disp   = bicone_fit.copy();    bicone_disp[nuc_disp_mask]   = 0.0

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────

np.save(OUT_DIR / "emission_map.npy",    emission_disp)
np.save(OUT_DIR / "smooth_model.npy",    smooth_fit)
np.save(OUT_DIR / "bicone_model.npy",    bicone_fit)
np.save(OUT_DIR / "residual_map.npy",    residual_map)
fits.writeto(str(OUT_DIR / "emission_map.fits"),  emission_disp, overwrite=True)
fits.writeto(str(OUT_DIR / "bicone_model.fits"),  bicone_fit,    overwrite=True)
fits.writeto(str(OUT_DIR / "smooth_model.fits"),  smooth_fit,    overwrite=True)

# Save cone fit parameters
cone_params_out = {
    "amplitude":       float(cone_best[0]),
    "axis_angle_deg":  float(np.degrees(cone_best[1])),
    "opening_sigma_deg": float(np.degrees(cone_best[2])),
    "r_scale_px":      float(cone_best[3]),
    "r_scale_arcsec":  float(cone_best[3] * 0.046),
    "asymmetry":       float(cone_best[4]),
    "optimiser_success": bool(result.success),
    "nfev":            int(result.nfev),
    "cost":            float(result.cost),
}
import json
with open(OUT_DIR / "cone_params.json", "w") as f:
    json.dump(cone_params_out, f, indent=2)
print(f"\nSaved to {OUT_DIR}/")

# ─────────────────────────────────────────────
# NORMALISE
# ─────────────────────────────────────────────

def norm_pct(img, lo=1.0, hi=99.5):
    p_lo = np.percentile(img, lo)
    p_hi = np.percentile(img, hi)
    return np.clip((img - p_lo) / (p_hi - p_lo + 1e-8), 0, 1).astype(np.float32)

def norm_asinh(img, pct=97.0):
    img = np.clip(img, 0, None)
    sc  = np.percentile(img[img > 0], pct) if (img > 0).any() else 1.0
    return (np.arcsinh(img / (sc + 1e-8)) / np.arcsinh(1.0)).astype(np.float32)

# ─────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(16, 11))
fig.patch.set_facecolor("#0a0a0a")

ax_deg   = np.degrees(cone_best[1])
fig.suptitle(
    f"NGC 1068 Two-Component Fit  |  "
    f"cone axis={ax_deg:.1f}°  opening={np.degrees(cone_best[2]):.1f}°σ  "
    f"r_scale={cone_best[3]:.0f}px  asym={cone_best[4]:.2f}",
    color="white", fontsize=11
)

def show(ax, img, title, cmap, mark_nuc=True, mark_circle=False):
    ax.imshow(img, origin="lower", cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title, color="white", fontsize=9)
    ax.axis("off")
    if mark_nuc:
        ax.plot(nuc_cx, nuc_cy, "+", color="cyan", markersize=12, markeredgewidth=1.5)
    if mark_circle:
        t = np.linspace(0, 2*np.pi, 100)
        ax.plot(nuc_cx + 15*np.cos(t), nuc_cy + 15*np.sin(t),
                color="cyan", lw=0.8, alpha=0.6)

show(axes[0,0], norm_pct(o3_crop),    "F502N  (data)",             "gray")
show(axes[0,1], norm_pct(smooth_fit), "Fitted smooth continuum",   "gray")
show(axes[0,2], norm_pct(cont_crop),  "F547M  (validation only)",  "gray")

show(axes[1,0], norm_asinh(emission_disp),
     "Emission = data − continuum  [asinh]",      "magma", False, True)
show(axes[1,1], norm_pct(np.clip(bicone_fit, 0, None)),
     "Fitted bicone model",                        "magma", False, True)
show(axes[1,2], norm_pct(np.abs(residual_map)),
     "Residual  |data − model|  (should be noise)","inferno", True)

# Overlay cone axis on emission panel
for ax in [axes[1,0], axes[1,1]]:
    for sign in [1, -1]:
        angle = cone_best[1] + (0 if sign == 1 else np.pi)
        length = 80
        ax.annotate("", xy=(nuc_cx + sign*length*np.cos(cone_best[1]),
                             nuc_cy + sign*length*np.sin(cone_best[1])),
                    xytext=(nuc_cx, nuc_cy),
                    arrowprops=dict(arrowstyle="->", color="yellow", lw=1.2))

plt.tight_layout()
plt.savefig(OUT_DIR / "two_component_result.png", dpi=200,
            bbox_inches="tight", facecolor="#0a0a0a")
plt.close()
print(f"Result → {OUT_DIR}/two_component_result.png")
print("\n✓ DONE")
print("  Key things to check:")
print("  - Cone axis angle should be ~30-35° (NE direction for NGC 1068)")
print("  - Residual panel should look like flat noise, not structured emission")
print("  - Bicone model should match the bright lobes in the emission panel")
