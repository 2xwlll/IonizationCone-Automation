#!/usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from pathlib import Path

# ─────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────

O3_FILE = "data/2d/MAST_2026-04-14T0111/HST/hst_5754_01_wfpc2_pc_f502n_u2m301/hst_5754_01_wfpc2_pc_f502n_u2m301_drz.fits"

OUT_DIR = Path("cone_v3_stable_output")
OUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────

with fits.open(O3_FILE) as h:
    o3 = np.nan_to_num(h[1].data.astype(np.float32))

o3 = gaussian_filter(o3, sigma=1)

ny, nx = o3.shape
print("Shape:", o3.shape)

# ─────────────────────────────────────────────
# NUCLEUS
# ─────────────────────────────────────────────

ny0, nx0 = np.unravel_index(np.argmax(o3), o3.shape)
print(f"Nucleus: {ny0}, {nx0}")

# ─────────────────────────────────────────────
# COORDINATES (FIXED: single allocation, reused flat arrays)
# ─────────────────────────────────────────────

y = np.arange(ny, dtype=np.float32)[:, None]
x = np.arange(nx, dtype=np.float32)[None, :]

dy = y - ny0
dx = x - nx0

r = np.sqrt(dx * dx + dy * dy).astype(np.float32)
theta = np.arctan2(dy, dx).astype(np.float32)

# flatten ONCE (important memory fix)
r_flat = r.ravel()
theta_flat = theta.ravel()

# ─────────────────────────────────────────────
# U-NET EMISSION MAP (STABLE + VISIBLE)
# ─────────────────────────────────────────────

small = gaussian_filter(o3, sigma=1)
large = gaussian_filter(o3, sigma=10)

emission_map = small - large
emission_map = np.clip(emission_map, 0, None)

# log compression (keeps faint structure)
emission_map = np.log1p(emission_map)

# normalize safely (no over-flattening)
p99 = np.percentile(emission_map, 99)
emission_map = np.clip(emission_map / (p99 + 1e-8), 0, 1)

# ─────────────────────────────────────────────
# CONE ANALYSIS (MEMORY SAFE HISTOGRAM METHOD)
# ─────────────────────────────────────────────

n_r = 15
n_theta = 180

r_edges = np.linspace(10, 300, n_r + 1)
theta_edges = np.linspace(-np.pi, np.pi, n_theta + 1)
theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])

peak_track = []

for i in range(n_r):

    rmin, rmax = r_edges[i], r_edges[i + 1]

    rm = (r_flat >= rmin) & (r_flat < rmax)

    if np.sum(rm) < 200:
        continue

    th = theta_flat[rm]

    inds = np.digitize(th, theta_edges) - 1
    inds = inds[(inds >= 0) & (inds < n_theta)]

    if inds.size == 0:
        continue

    hist = np.bincount(inds, minlength=n_theta)

    peak_track.append(np.argmax(hist))

peak_track = np.array(peak_track)

# ─────────────────────────────────────────────
# CONE METRICS
# ─────────────────────────────────────────────

if peak_track.size > 0:
    mode_bin = np.bincount(peak_track).argmax()
    stability = np.std(peak_track)
else:
    mode_bin = -1
    stability = np.inf

cone_score = 1.0 / (1.0 + stability)

peak_angle = theta_centers[mode_bin] if mode_bin >= 0 else 0.0
opp_angle = peak_angle + np.pi
if opp_angle > np.pi:
    opp_angle -= 2 * np.pi

print("\n--- CONE RESULTS ---")
print(f"Primary axis: {peak_angle:.3f} rad")
print(f"Opposite axis: {opp_angle:.3f} rad")
print(f"Cone stability score: {cone_score:.4f}")

# ─────────────────────────────────────────────
# SAFE PLOTTING (NO MEMORY SPIKES)
# ─────────────────────────────────────────────

step = 3  # aggressive downsample for safety

plt.figure(figsize=(6, 6))
plt.imshow(emission_map[::step, ::step], origin="lower", cmap="magma")
plt.plot(nx / step, ny / step, "c+")
plt.title("U-Net Emission Map (stable)")
plt.savefig(OUT_DIR / "emission_map.png", dpi=200)
plt.close()

plt.figure()
plt.plot(peak_track, marker="o")
plt.title("Angular Stability Across Radius")
plt.xlabel("Radius bin")
plt.ylabel("Peak angle bin")
plt.savefig(OUT_DIR / "stability.png", dpi=200)
plt.close()

plt.figure()
plt.hist(peak_track, bins=n_theta)
plt.title("Angular Distribution")
plt.savefig(OUT_DIR / "angular_hist.png", dpi=200)
plt.close()

print("\nSaved outputs to:", OUT_DIR)
