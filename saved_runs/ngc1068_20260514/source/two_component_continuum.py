#!/usr/bin/env python3
"""
two_component_fit.py  —  polar coordinate cone extractor

Philosophy change from previous versions:
    Old: fit Gaussian brightness model → infer geometry from brightness
    New: measure data → extract geometry from emission support in polar space

Pipeline:
    1. Continuum subtraction (F547M scaled by PHOTFLAM ratio)
    2. Build S/N map
    3. Convert to polar coordinates around nucleus
    4. Build 2D polar emission grid (r, theta)
    5. Collapse to 1D angular profile
    6. Find cone edges where emission turns on/off
    7. Measure opening angle and axis from support, not brightness peak
    8. Generate soft mask from angular support region
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import gaussian_filter, uniform_filter1d
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
# PHOTOMETRIC CONSTANTS
# ─────────────────────────────────────────────

PHOTFLAM_502   = 2.943716e-16
PHOTFLAM_547   = 7.595041e-18
PHOTFLAM_RATIO = PHOTFLAM_547 / PHOTFLAM_502

# ─────────────────────────────────────────────
# POLAR GRID PARAMETERS
# Tune these if cone edges look wrong
# ─────────────────────────────────────────────

R_MIN          = 15      # px — inner exclusion (nucleus PSF)
R_MAX          = 110     # px — outer limit (~5 arcsec, full NGC 1068 NLR)
N_R_BINS       = 40      # radial bins
N_THETA_BINS   = 360     # angular bins — 1 degree per bin
ANGULAR_SMOOTH = 7       # bins — smoothing width for angular profile
EDGE_THRESHOLD = 0.20    # fraction of peak — cone edge definition
MIN_CONE_WIDTH = 20      # degrees — ignore features narrower than this
MASK_SNR_CUT   = 2.5     # S/N threshold for final pixel mask

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
theta_grid = np.arctan2(yy - nuc_cy, xx - nuc_cx)   # -pi to pi

print(f"Crop: {o3_crop.shape}  nucleus at ({nuc_cx}, {nuc_cy})")

# ─────────────────────────────────────────────
# PSF MATCH + SCALE
# ─────────────────────────────────────────────

PSF_MATCH_SIGMA = 0.5
cont_matched    = gaussian_filter(cont_crop, sigma=PSF_MATCH_SIGMA)

sky_mask = (r_grid > 130) & (r_grid < 180) & valid_crop
print(f"Sky annulus pixels: {sky_mask.sum()}")

cont_phot_sky       = np.median(cont_matched[sky_mask]) * PHOTFLAM_RATIO
o3_sky              = np.median(o3_crop[sky_mask])
residual_correction = o3_sky / (cont_phot_sky + 1e-12)
scale_factor        = PHOTFLAM_RATIO * residual_correction

print(f"PHOTFLAM ratio:       {PHOTFLAM_RATIO:.6f}")
print(f"Residual correction:  {residual_correction:.4f}")
print(f"Final scale factor:   {scale_factor:.6f}")

cont_scaled = cont_matched * scale_factor

# ─────────────────────────────────────────────
# EMISSION MAP + S/N
# ─────────────────────────────────────────────

emission_direct = o3_crop - cont_scaled
noise_sigma     = np.std(emission_direct[sky_mask])
snr_map_raw     = emission_direct / (noise_sigma + 1e-12)

# smooth emission for polar analysis
# this recovers faint extended structure that individual noisy
# pixels hide — critical for measuring true opening angle
emission_smooth = gaussian_filter(emission_direct, sigma=2.0)
noise_smooth    = noise_sigma / np.sqrt(np.pi * 2.0**2)
snr_emission    = emission_smooth / (noise_smooth + 1e-12)

# nucleus masked
nuc_mask     = r_grid < R_MIN
emission_map = emission_direct.copy()
emission_map[nuc_mask]    = 0.0
emission_map[~valid_crop] = 0.0

print(f"Sky noise (raw):      {noise_sigma:.6f}")
print(f"Sky noise (smoothed): {noise_smooth:.6f}")
print(f"Peak S/N (smoothed):  {snr_emission.max():.1f}")

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
# DIAGNOSTIC — continuum subtraction
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
# STAGE 2 — POLAR COORDINATE GRID
#
# Bin every pixel into (r, theta) space.
# Use median S/N per bin — robust to outliers.
# Pixels below S/N=1 or outside valid region excluded.
# ─────────────────────────────────────────────

print("\nBuilding polar emission grid...")

r_bins     = np.linspace(R_MIN, R_MAX, N_R_BINS + 1)
theta_bins = np.linspace(-np.pi, np.pi, N_THETA_BINS + 1)
r_centers  = 0.5 * (r_bins[:-1]     + r_bins[1:])
t_centers  = 0.5 * (theta_bins[:-1] + theta_bins[1:])

polar_snr = np.full((N_R_BINS, N_THETA_BINS), np.nan, dtype=np.float32)

for i in range(N_R_BINS):
    for j in range(N_THETA_BINS):
        in_bin = (
            (r_grid     >= r_bins[i])     & (r_grid     < r_bins[i+1])
            & (theta_grid >= theta_bins[j]) & (theta_grid < theta_bins[j+1])
            & valid_crop
            & ~nuc_mask
        )
        if in_bin.sum() >= 2:
            polar_snr[i, j] = np.median(snr_emission[in_bin])

print(f"Polar grid built: {N_R_BINS} r-bins × {N_THETA_BINS} theta-bins")
print(f"Filled bins: {(~np.isnan(polar_snr)).sum()} / {N_R_BINS*N_THETA_BINS}")

# ─────────────────────────────────────────────
# STAGE 3 — ANGULAR PROFILE
#
# Collapse polar grid to 1D angular profile.
# Use nanmedian across r — gives emission as function of angle only.
# This is the key step: every angle gets one S/N value,
# regardless of radial brightness distribution.
# ─────────────────────────────────────────────

angular_profile = np.nanmedian(polar_snr, axis=0)
angular_profile = np.nan_to_num(angular_profile, nan=0.0)

# smooth to suppress noise spikes
angular_smooth  = uniform_filter1d(angular_profile, size=ANGULAR_SMOOTH,
                                   mode="wrap")

print(f"\nAngular profile peak S/N: {angular_smooth.max():.2f}")
print(f"Angular profile min  S/N: {angular_smooth.min():.2f}")

# ─────────────────────────────────────────────
# STAGE 4 — FIND CONE EDGES
#
# Threshold at EDGE_THRESHOLD * peak.
# Find contiguous angular regions above threshold.
# The two largest regions = two cone lobes.
# Opening angle = angular span of each region.
# Axis = midpoint of primary lobe.
#
# This measures geometry from data support —
# not from brightness peak or Gaussian width.
# ─────────────────────────────────────────────

threshold   = EDGE_THRESHOLD * angular_smooth.max()
above       = angular_smooth > threshold

print(f"\nEdge threshold: {threshold:.3f} S/N "
      f"({EDGE_THRESHOLD*100:.0f}% of peak {angular_smooth.max():.2f})")
print(f"Bins above threshold: {above.sum()} / {N_THETA_BINS} "
      f"({above.sum()/N_THETA_BINS*360:.1f}°)")

# find contiguous regions (wrap-around aware)
def find_contiguous_regions(mask):
    """
    Find contiguous True regions in a circular boolean array.
    Returns list of (start_idx, end_idx, length) tuples.
    """
    regions = []
    n       = len(mask)
    # double the array to handle wrap-around
    doubled = np.concatenate([mask, mask])
    i       = 0
    while i < n:
        if doubled[i]:
            j = i
            while j < i + n and doubled[j]:
                j += 1
            length = j - i
            if length < n:  # avoid full-circle region
                regions.append((i % n, (j-1) % n, length))
            i = j
        else:
            i += 1
    # deduplicate
    seen    = set()
    unique  = []
    for r in regions:
        key = r[0]
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return sorted(unique, key=lambda x: -x[2])

regions = find_contiguous_regions(above)
print(f"Contiguous regions above threshold: {len(regions)}")
for k, (s, e, l) in enumerate(regions[:4]):
    span_deg = l * 360.0 / N_THETA_BINS
    mid_deg  = np.degrees(t_centers[(s + l//2) % N_THETA_BINS])
    print(f"  region {k}: span={span_deg:.1f}°  midpoint={mid_deg:.1f}°  "
          f"bins={l}")

# filter out narrow spurs
min_bins = int(MIN_CONE_WIDTH / 360.0 * N_THETA_BINS)
regions  = [r for r in regions if r[2] >= min_bins]
print(f"Regions after {MIN_CONE_WIDTH}° minimum width filter: {len(regions)}")

if len(regions) == 0:
    print("WARNING: no cone regions found — lower EDGE_THRESHOLD or MASK_SNR_CUT")
    # fall back to full annulus mask
    cone_axis_deg    = 120.0
    opening_half_deg = 40.0
    counter_axis_deg = cone_axis_deg + 180.0
else:
    # primary lobe
    s1, e1, l1      = regions[0]
    mid1            = (s1 + l1//2) % N_THETA_BINS
    cone_axis_rad   = t_centers[mid1]
    cone_axis_deg   = np.degrees(cone_axis_rad)
    opening_half_deg = 0.5 * l1 * 360.0 / N_THETA_BINS

    print(f"\nPrimary lobe:")
    print(f"  axis    = {cone_axis_deg:.1f}°")
    print(f"  opening = {2*opening_half_deg:.1f}° full  "
          f"({opening_half_deg:.1f}° half-angle)")

    # counter-lobe — look for a region ~180° away
    counter_axis_deg = cone_axis_deg + 180.0
    if counter_axis_deg > 180:
        counter_axis_deg -= 360.0

    if len(regions) >= 2:
        s2, e2, l2       = regions[1]
        mid2             = (s2 + l2//2) % N_THETA_BINS
        counter_axis_rad = t_centers[mid2]
        counter_axis_deg = np.degrees(counter_axis_rad)
        counter_half_deg = 0.5 * l2 * 360.0 / N_THETA_BINS
        print(f"\nCounter-lobe:")
        print(f"  axis    = {counter_axis_deg:.1f}°")
        print(f"  opening = {2*counter_half_deg:.1f}° full")
        asym = l2 / l1   # angular extent ratio
        print(f"  asymmetry (extent ratio) = {asym:.2f}")
    else:
        print("\nNo clear counter-lobe detected")
        counter_half_deg = opening_half_deg * 0.3
        asym             = 0.3

# ─────────────────────────────────────────────
# STAGE 5 — BUILD CONE MASK FROM SUPPORT
#
# Project angular support back to image space.
# This mask follows the data, not a model.
# ─────────────────────────────────────────────

# build angular support mask per pixel
# a pixel is in the cone if its angle is in an above-threshold bin
theta_in_cone = np.zeros(N_THETA_BINS, dtype=bool)
for reg in regions:
    s, e, l = reg
    for k in range(l):
        theta_in_cone[(s + k) % N_THETA_BINS] = True

# map each pixel to its theta bin
theta_bin_idx = np.floor(
    (theta_grid + np.pi) / (2*np.pi) * N_THETA_BINS
).astype(int) % N_THETA_BINS

# cone region: pixel angle is in support AND radial range
in_cone_angle  = theta_in_cone[theta_bin_idx]
in_cone_radial = (r_grid > R_MIN) & (r_grid < R_MAX)
cone_region    = in_cone_angle & in_cone_radial & valid_crop & ~nuc_mask

# soft mask: S/N weighted within cone region
snr_for_mask = emission_map / (noise_sigma + 1e-12)

soft_mask = np.where(
    cone_region,
    np.clip(snr_for_mask / 8.0, 0, 1),
    0.0
).astype(np.float32)

cone_mask = (
    cone_region
    & (snr_for_mask >= MASK_SNR_CUT)
).astype(np.float32)

print(f"\nCone region pixels:  {cone_region.sum()}")
print(f"Cone mask pixels:    {int(cone_mask.sum())}  (S/N >= {MASK_SNR_CUT})")
print(f"Soft mask max:       {soft_mask.max():.3f}")
print(f"Soft mask mean:      {soft_mask[soft_mask>0].mean():.3f}")

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────

np.save(OUT_DIR / "emission_map.npy",     emission_map)
np.save(OUT_DIR / "snr_map.npy",          snr_for_mask)
np.save(OUT_DIR / "cone_mask.npy",        cone_mask)
np.save(OUT_DIR / "soft_mask.npy",        soft_mask)
np.save(OUT_DIR / "polar_snr.npy",        polar_snr)
np.save(OUT_DIR / "angular_profile.npy",  angular_smooth)
np.save(OUT_DIR / "cont_scaled.npy",      cont_scaled)

emission_asinh = norm_asinh(emission_map)
np.save(OUT_DIR / "emission_map_asinh.npy", emission_asinh)

fits.writeto(str(OUT_DIR / "emission_map.fits"), emission_map, overwrite=True)
fits.writeto(str(OUT_DIR / "cone_mask.fits"),    cone_mask,    overwrite=True)
fits.writeto(str(OUT_DIR / "soft_mask.fits"),    soft_mask,    overwrite=True)

cone_params_out = {
    "cone_axis_deg":        float(cone_axis_deg),
    "opening_half_deg":     float(opening_half_deg),
    "opening_full_deg":     float(2 * opening_half_deg),
    "counter_axis_deg":     float(counter_axis_deg),
    "asymmetry":            float(asym),
    "method":               "polar_angular_support",
    "edge_threshold":       float(EDGE_THRESHOLD),
    "r_min_px":             R_MIN,
    "r_max_px":             R_MAX,
    "mask_snr_cut":         float(MASK_SNR_CUT),
    "angular_smooth_bins":  ANGULAR_SMOOTH,
    "scale_factor":         float(scale_factor),
    "photflam_ratio":       float(PHOTFLAM_RATIO),
    "residual_correction":  float(residual_correction),
    "noise_sigma":          float(noise_sigma),
    "cone_mask_pixels":     int(cone_mask.sum()),
    # inside cone_params_out dict:
    "nucleus_cx": int(nuc_cx),
    "nucleus_cy": int(nuc_cy),
    }

with open(OUT_DIR / "cone_params.json", "w") as f:
    json.dump(cone_params_out, f, indent=2)

print(f"\nSaved to {OUT_DIR}/")

# ─────────────────────────────────────────────
# FIGURE — main result
# ─────────────────────────────────────────────

emission_disp = emission_map.copy()

fig, axes = plt.subplots(2, 3, figsize=(16, 11))
fig.patch.set_facecolor("#0a0a0a")

fig.suptitle(
    f"NGC 1068  |  cone axis={cone_axis_deg:.1f}°  "
    f"opening={2*opening_half_deg:.1f}° full  "
    f"asym={asym:.2f}  "
    f"method=polar_support",
    color="white", fontsize=10
)

def show(ax, img, title, cmap, mark_nuc=True):
    ax.imshow(img, origin="lower", cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title, color="white", fontsize=9)
    ax.axis("off")
    if mark_nuc:
        ax.plot(nuc_cx, nuc_cy, "+", color="cyan",
                markersize=12, markeredgewidth=1.5)

show(axes[0,0], norm_pct(o3_crop),
     "F502N (data)",                              "gray")
show(axes[0,1], norm_pct(cont_scaled),
     f"F547M scaled ×{scale_factor:.5f}",         "gray")
show(axes[0,2], norm_pct(cont_crop),
     "F547M (raw)",                               "gray")
show(axes[1,0], norm_asinh(emission_disp),
     "Emission = F502N − F547M_scaled  [asinh]",  "magma", False)
axes[1,0].plot(nuc_cx, nuc_cy, "+", color="cyan", markersize=12)

# angular profile plot
ax_prof = axes[1,1]
ax_prof.set_facecolor("#0a0a0a")
t_deg = np.degrees(t_centers)
ax_prof.plot(t_deg, angular_smooth, color="white",  lw=1.5, label="S/N profile")
ax_prof.axhline(threshold, color="yellow", lw=1, ls="--", label=f"threshold={threshold:.2f}")
ax_prof.axvline(cone_axis_deg,    color="cyan",  lw=1.2, label=f"axis={cone_axis_deg:.1f}°")
ax_prof.axvline(counter_axis_deg, color="orange",lw=1.2, ls="--",
                label=f"counter={counter_axis_deg:.1f}°")
ax_prof.set_xlabel("angle (degrees)", color="white", fontsize=8)
ax_prof.set_ylabel("median S/N",      color="white", fontsize=8)
ax_prof.set_title("Angular emission profile",  color="white", fontsize=9)
ax_prof.tick_params(colors="white", labelsize=7)
ax_prof.spines[:].set_color("#444")
ax_prof.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="white",
               loc="upper right")
ax_prof.set_xlim(-180, 180)

# cone axis arrow on emission panel
cone_axis_rad_plot = np.radians(cone_axis_deg)
for sign in [1, -1]:
    length = 90
    axes[1,0].annotate("",
        xy=(nuc_cx + sign * length * np.cos(cone_axis_rad_plot),
            nuc_cy + sign * length * np.sin(cone_axis_rad_plot)),
        xytext=(nuc_cx, nuc_cy),
        arrowprops=dict(arrowstyle="->", color="yellow", lw=1.2))

show(axes[1,2], soft_mask,
     "Soft cone mask (training label)",  "inferno", True)

plt.tight_layout()
out_png = OUT_DIR / "two_component_result.png"
plt.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="#0a0a0a")
plt.close()
print(f"Result → {out_png}")

# ─────────────────────────────────────────────
# FIGURE — polar diagnostic
# ─────────────────────────────────────────────

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.patch.set_facecolor("#0a0a0a")

im = axes2[0].imshow(
    polar_snr, origin="lower", aspect="auto", cmap="magma",
    extent=[-180, 180, R_MIN, R_MAX], vmin=0, vmax=np.nanpercentile(polar_snr, 98)
)
axes2[0].set_xlabel("angle (degrees)", color="white", fontsize=9)
axes2[0].set_ylabel("radius (pixels)", color="white", fontsize=9)
axes2[0].set_title("Polar S/N grid (r, θ)", color="white", fontsize=9)
axes2[0].tick_params(colors="white")
axes2[0].spines[:].set_color("#444")
axes2[0].axvline(cone_axis_deg,    color="cyan",  lw=1.2, ls="--")
axes2[0].axvline(counter_axis_deg, color="orange",lw=1.2, ls="--")
plt.colorbar(im, ax=axes2[0], label="median S/N").ax.yaxis.set_tick_params(color="white")

axes2[1].set_facecolor("#0a0a0a")
axes2[1].plot(t_deg, angular_smooth, color="white", lw=2)
axes2[1].fill_between(t_deg, 0, angular_smooth,
                      where=angular_smooth > threshold,
                      alpha=0.4, color="cyan", label="cone support")
axes2[1].axhline(threshold, color="yellow", lw=1, ls="--",
                 label=f"{EDGE_THRESHOLD*100:.0f}% threshold")
axes2[1].axvline(cone_axis_deg,    color="cyan",  lw=1.5,
                 label=f"axis {cone_axis_deg:.1f}°")
axes2[1].axvline(counter_axis_deg, color="orange",lw=1.5, ls="--",
                 label=f"counter {counter_axis_deg:.1f}°")
axes2[1].set_xlabel("angle (degrees)", color="white", fontsize=9)
axes2[1].set_ylabel("median S/N",      color="white", fontsize=9)
axes2[1].set_title(
    f"Angular profile  →  opening={2*opening_half_deg:.1f}°",
    color="white", fontsize=9
)
axes2[1].tick_params(colors="white", labelsize=8)
axes2[1].spines[:].set_color("#444")
axes2[1].legend(fontsize=8, facecolor="#1a1a1a", labelcolor="white")
axes2[1].set_xlim(-180, 180)

plt.tight_layout()
plt.savefig(
    OUT_DIR / "polar_diagnostic.png", dpi=150,
    bbox_inches="tight", facecolor="#0a0a0a"
)
plt.close()
print(f"Polar diagnostic → {OUT_DIR}/polar_diagnostic.png")

print("\nDONE")
print("  Key outputs:")
print(f"  cone_axis_deg    = {cone_axis_deg:.1f}°  (expect ~120° for NGC 1068)")
print(f"  opening_full_deg = {2*opening_half_deg:.1f}°  (literature ~80°)")
print(f"  cone_mask pixels = {int(cone_mask.sum())}")
print("\n  If opening angle is wrong:")
print(f"  - lower  EDGE_THRESHOLD (currently {EDGE_THRESHOLD}) to include more faint emission")
print(f"  - raise  R_MAX (currently {R_MAX}px) if emission extends further")
print(f"  - lower  ANGULAR_SMOOTH (currently {ANGULAR_SMOOTH}) for sharper edges")
print("\n  Check polar_diagnostic.png — the angular profile plot is the key diagnostic")
