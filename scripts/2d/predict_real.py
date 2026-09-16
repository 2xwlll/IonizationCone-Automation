#!/usr/bin/env python3
"""
predict_real.py — run trained UNet on real NGC 1068 emission map

Input:  data/2d/ngc1068_two_component/emission_map_asinh.npy
        data/2d/ngc1068_two_component/cone_params.json
Output: results/visualizations/ngc1068_prediction.png
        results/visualizations/ngc1068_prediction_params.json
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, label
from src.machine_learning.models.model_2d import UNet

# --------------------------
# CONFIG
# --------------------------
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
BASE_RESULTS    = "results/2d/unet"
EMISSION_PATH   = "data/2d/ngc1068_two_component/emission_map_asinh.npy"
SOFT_MASK_PATH  = "data/2d/ngc1068_two_component/soft_mask.npy"
PARAMS_PATH     = "data/2d/ngc1068_two_component/cone_params.json"
SAVE_DIR        = "results/visualizations"
GRID            = 128
THRESHOLD       = 0.5
NUCLEUS_CROP    = 110

os.makedirs(SAVE_DIR, exist_ok=True)

# --------------------------
# FIND LATEST RUN
# --------------------------
def get_latest_run(base_dir):
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"No runs found at: {base_dir}")
    runs = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("run_")
    ])
    if not runs:
        raise ValueError("No run folders found.")
    return os.path.join(base_dir, runs[-1])

# --------------------------
# LOAD NUCLEUS COORDS
# --------------------------
def load_nucleus_coords(params_path):
    if not os.path.exists(params_path):
        print("  WARNING: cone_params.json not found — using brightest pixel")
        return None, None

    with open(params_path) as f:
        params = json.load(f)

    cx = params.get("nucleus_cx", None)
    cy = params.get("nucleus_cy", None)

    if cx is None or cy is None:
        print("  WARNING: nucleus_cx/cy missing from cone_params.json")
        print("  Add to cone_params_out in two_component_fit.py:")
        print('      "nucleus_cx": int(nuc_cx),')
        print('      "nucleus_cy": int(nuc_cy),')
        print("  Then rerun two_component_fit.py. Falling back to brightest pixel.")
        return None, None

    print(f"  Nucleus from cone_params.json: ({cx}, {cy})")
    return cx, cy

# --------------------------
# PREPROCESS
# --------------------------
def preprocess(emission_map, cx=None, cy=None, grid=GRID):
    img = emission_map.copy().astype(np.float32)
    img = np.clip(img, 0, 1)

    if cx is None or cy is None:
        cy, cx = np.unravel_index(np.argmax(img), img.shape)
        print(f"  Fallback nucleus (brightest pixel): ({cx}, {cy})")

    half     = NUCLEUS_CROP
    y0       = max(0, cy - half)
    y1       = min(img.shape[0], cy + half)
    x0       = max(0, cx - half)
    x1       = min(img.shape[1], cx + half)
    img_crop = img[y0:y1, x0:x1]

    print(f"  Crop: [{y0}:{y1}, {x0}:{x1}] → {img_crop.shape}")

    if img_crop.shape != (grid, grid):
        zoom_y   = grid / img_crop.shape[0]
        zoom_x   = grid / img_crop.shape[1]
        img_crop = zoom(img_crop, (zoom_y, zoom_x), order=1)

    tensor = torch.from_numpy(img_crop).unsqueeze(0).unsqueeze(0)
    return tensor, img_crop, (y0, y1, x0, x1)

def preprocess_mask(mask, crop_bounds, grid=GRID):
    y0, y1, x0, x1 = crop_bounds
    m = mask[y0:y1, x0:x1].astype(np.float32)
    if m.shape != (grid, grid):
        zoom_y = grid / m.shape[0]
        zoom_x = grid / m.shape[1]
        m      = zoom(m, (zoom_y, zoom_x), order=1)
    return m

# --------------------------
# LOAD MODEL
# --------------------------
def load_model(model_path):
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"  Loaded: {model_path}")
    return model

# --------------------------
# PREDICT
# --------------------------
@torch.no_grad()
def predict(model, tensor):
    tensor = tensor.to(DEVICE)
    logits = model(tensor)
    probs  = torch.sigmoid(logits)
    binary = (probs > THRESHOLD).float()
    return (
        probs.squeeze().cpu().numpy(),
        binary.squeeze().cpu().numpy()
    )

# --------------------------
# MEASURE CONE GEOMETRY
# --------------------------
def measure_cone_geometry(binary, grid=GRID):
    """
    Measure cone geometry from predicted binary mask.

    Method:
        1. Find nucleus (center of grid — crop is centered on nucleus)
        2. Convert each predicted pixel to polar coords (r, theta)
        3. Build angular profile — count predicted pixels per angle bin
        4. Find contiguous above-threshold angular region = cone lobe
        5. Measure axis (midpoint) and opening angle (span)

    Returns dict with cone_axis_deg, opening_full_deg, and per-lobe details.
    Comparable directly to two_component_fit.py output and literature values.

    NGC 1068 literature reference:
        Axis PA ~ 120° (E of N)
        Opening angle ~ 80° full
        (Wilson & Tsvetanov 1994, Das et al. 2006)
    """
    if binary.sum() == 0:
        print("  WARNING: no predicted pixels — cannot measure geometry")
        return {}

    c   = grid // 2
    yy, xx = np.mgrid[0:grid, 0:grid]

    # polar coords centered on nucleus (grid center)
    dx = (xx - c).astype(np.float32)
    dy = (yy - c).astype(np.float32)
    r_grid     = np.sqrt(dx**2 + dy**2) + 1e-6
    theta_grid = np.degrees(np.arctan2(dy, dx))  # -180 to 180

    # only consider predicted pixels outside nucleus PSF
    r_min_px = grid * 0.05
    valid    = (binary > 0.5) & (r_grid > r_min_px)

    if valid.sum() == 0:
        print("  WARNING: all predicted pixels inside nucleus exclusion zone")
        return {}

    # angular histogram — 1 degree bins
    n_bins    = 360
    bin_edges = np.linspace(-180, 180, n_bins + 1)
    counts, _ = np.histogram(theta_grid[valid], bins=bin_edges)

    # smooth slightly to suppress single-pixel spurs
    from scipy.ndimage import uniform_filter1d
    counts_smooth = uniform_filter1d(counts.astype(float), size=5, mode="wrap")

    # find contiguous above-zero regions (cone lobes)
    above  = counts_smooth > 0
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # wrap-aware contiguous region finder
    regions = []
    doubled = np.concatenate([above, above])
    i = 0
    while i < n_bins:
        if doubled[i]:
            j = i
            while j < i + n_bins and doubled[j]:
                j += 1
            length = j - i
            if length < n_bins:
                regions.append((i % n_bins, (j-1) % n_bins, length))
            i = j
        else:
            i += 1

    # deduplicate
    seen, unique = set(), []
    for r in regions:
        if r[0] not in seen:
            seen.add(r[0])
            unique.append(r)
    regions = sorted(unique, key=lambda x: -x[2])

    if not regions:
        return {}

    results = {}

    # primary lobe
    s1, e1, l1       = regions[0]
    mid1_idx         = (s1 + l1 // 2) % n_bins
    axis_deg         = float(bin_centers[mid1_idx])
    opening_full_deg = float(l1)   # 1 bin = 1 degree

    results["cone_axis_deg"]     = round(axis_deg, 1)
    results["opening_full_deg"]  = round(opening_full_deg, 1)
    results["opening_half_deg"]  = round(opening_full_deg / 2, 1)
    results["predicted_pixels"]  = int(valid.sum())
    results["cone_fraction_pct"] = round(100 * valid.sum() / binary.size, 2)

    print(f"\n  ── UNet Cone Geometry ──────────────────────")
    print(f"  Cone axis PA:      {axis_deg:.1f}°")
    print(f"  Opening angle:     {opening_full_deg:.1f}° (full)  "
          f"{opening_full_deg/2:.1f}° (half)")
    print(f"  Predicted pixels:  {valid.sum()} ({results['cone_fraction_pct']:.1f}%)")

    # counter-lobe if present
    if len(regions) >= 2:
        s2, e2, l2       = regions[1]
        mid2_idx         = (s2 + l2 // 2) % n_bins
        counter_axis_deg = float(bin_centers[mid2_idx])
        counter_open_deg = float(l2)
        asym             = round(l2 / l1, 2)

        results["counter_axis_deg"]     = round(counter_axis_deg, 1)
        results["counter_opening_deg"]  = round(counter_open_deg, 1)
        results["lobe_asymmetry"]       = asym

        print(f"  Counter-lobe axis: {counter_axis_deg:.1f}°")
        print(f"  Counter opening:   {counter_open_deg:.1f}°")
        print(f"  Lobe asymmetry:    {asym}")

    # literature comparison
    lit_axis    = 120.0
    lit_opening = 80.0
    axis_offset = abs(axis_deg - lit_axis)
    # handle wrap-around
    axis_offset = min(axis_offset, 360 - axis_offset)
    open_offset = abs(opening_full_deg - lit_opening)

    results["literature_axis_deg"]        = lit_axis
    results["literature_opening_full_deg"] = lit_opening
    results["axis_offset_from_lit_deg"]   = round(axis_offset, 1)
    results["opening_offset_from_lit_deg"] = round(open_offset, 1)

    print(f"\n  ── Literature Comparison ───────────────────")
    print(f"  Literature axis:    {lit_axis:.1f}°  "
          f"(offset: {axis_offset:.1f}°)")
    print(f"  Literature opening: {lit_opening:.1f}°  "
          f"(offset: {open_offset:.1f}°)")
    print(f"  ────────────────────────────────────────────")

    return results

# --------------------------
# VISUALIZE
# --------------------------
def visualize(img, probs, binary, geometry, soft_mask=None):
    n_panels = 5 if soft_mask is not None else 4
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 5))
    fig.patch.set_facecolor("#0a0a0a")

    def show(ax, data, title, cmap, vmin=0, vmax=1):
        ax.imshow(data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, color="white", fontsize=9)
        ax.axis("off")

    show(axes[0], img,    "Emission map [asinh]",     "magma")
    show(axes[1], probs,  "UNet probability",         "viridis")
    show(axes[2], binary, "UNet prediction (binary)", "gray")

    # overlay with cone axis arrow if geometry measured
    axes[3].imshow(img, origin="lower", cmap="magma", vmin=0, vmax=1)
    if binary.sum() > 0:
        axes[3].contour(
            binary, levels=[0.5], colors=["cyan"], linewidths=1.2
        )
    if geometry.get("cone_axis_deg") is not None:
        c          = GRID // 2
        axis_rad   = np.radians(geometry["cone_axis_deg"])
        arrow_len  = GRID * 0.35
        for sign in [1, -1]:
            axes[3].annotate(
                "",
                xy=(c + sign * arrow_len * np.cos(axis_rad),
                    c + sign * arrow_len * np.sin(axis_rad)),
                xytext=(c, c),
                arrowprops=dict(arrowstyle="->", color="yellow", lw=1.2)
            )
    axes[3].set_title("Emission + predicted cone", color="white", fontsize=9)
    axes[3].axis("off")

    if soft_mask is not None:
        show(axes[4], soft_mask,
             "Classical soft mask\n(two_component_fit)", "inferno")

    # build title with geometry
    cone_pct  = 100 * binary.sum() / binary.size
    axis_str  = (f"axis={geometry['cone_axis_deg']:.1f}°  "
                 f"opening={geometry['opening_full_deg']:.1f}°  "
                 if geometry else "")
    plt.suptitle(
        f"NGC 1068  |  UNet prediction  |  threshold={THRESHOLD}  |  "
        f"cone={cone_pct:.1f}%  |  {axis_str}",
        color="white", fontsize=9
    )
    plt.tight_layout()

    out_path = os.path.join(SAVE_DIR, "ngc1068_prediction.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close()
    print(f"\n  Figure → {out_path}")

# --------------------------
# MAIN
# --------------------------
def main():
    print("\n--- NGC 1068 PREDICTION ---\n")

    run_dir    = get_latest_run(BASE_RESULTS)
    model_path = os.path.join(run_dir, "models", "best.pth")
    print(f"Run: {run_dir}\n")

    print("Loading nucleus coordinates...")
    cx, cy = load_nucleus_coords(PARAMS_PATH)

    print(f"\nLoading emission map...")
    if not os.path.exists(EMISSION_PATH):
        raise FileNotFoundError(
            f"Asinh emission map not found at {EMISSION_PATH}\n"
            f"Add to two_component_fit.py after the np.save block:\n\n"
            f"    emission_asinh = norm_asinh(emission_map)\n"
            f"    np.save(OUT_DIR / 'emission_map_asinh.npy', emission_asinh)\n\n"
            f"And add to cone_params_out dict:\n\n"
            f'    "nucleus_cx": int(nuc_cx),\n'
            f'    "nucleus_cy": int(nuc_cy),\n\n'
            f"Then rerun two_component_fit.py."
        )

    emission_map = np.load(EMISSION_PATH)
    print(f"  Shape: {emission_map.shape}")
    print(f"  Min={emission_map.min():.4f}  Max={emission_map.max():.4f}")

    soft_mask = None
    if os.path.exists(SOFT_MASK_PATH):
        soft_mask = np.load(SOFT_MASK_PATH)
        print(f"  Classical mask loaded: {soft_mask.shape}")

    print("\nPreprocessing...")
    tensor, img_processed, crop_bounds = preprocess(emission_map, cx=cx, cy=cy)

    if soft_mask is not None:
        soft_mask = preprocess_mask(soft_mask, crop_bounds)

    print("\nLoading model...")
    model = load_model(model_path)

    print("\nPredicting...")
    probs, binary = predict(model, tensor)
    print(f"  Prob min={probs.min():.3f}  max={probs.max():.3f}  "
          f"mean={probs.mean():.3f}")
    print(f"  Cone pixels: {int(binary.sum())} / {binary.size} "
          f"({100*binary.sum()/binary.size:.1f}%)")

    print("\nMeasuring cone geometry...")
    geometry = measure_cone_geometry(binary)

    # save geometry params
    if geometry:
        params_out = os.path.join(SAVE_DIR, "ngc1068_prediction_params.json")
        with open(params_out, "w") as f:
            json.dump(geometry, f, indent=2)
        print(f"  Params → {params_out}")

    print("\nGenerating figure...")
    visualize(img_processed, probs, binary, geometry, soft_mask)

    print("\nDone.\n")

if __name__ == "__main__":
    main()
