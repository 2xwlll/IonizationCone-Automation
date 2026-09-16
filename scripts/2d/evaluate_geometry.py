#!/usr/bin/env python3
"""
evaluate_geometry.py — compute IoU, PA error, and opening angle error
from saved model weights on the validation set.

No retraining needed — loads best.pth directly.

Outputs:
    - per-sample table to terminal
    - summary: mean IoU, mean |ΔPA|, mean |Δopening|
    - results/visualizations/geometry_eval.json
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.ndimage import uniform_filter1d

from src.machine_learning.datasets.ionization_dataset import IonizationConeDataset2D
from src.machine_learning.models.model_2d import UNet

# --------------------------
# CONFIG
# --------------------------
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
BASE_RESULTS = "results/2d/unet"
DATASET_ROOT = "data/2d/synthetic_oiii_realistic/val"
GRID         = 128
THRESHOLD    = 0.5
SAVE_DIR     = "results/visualizations"

os.makedirs(SAVE_DIR, exist_ok=True)

# --------------------------
# FIND LATEST RUN
# --------------------------
def get_latest_run(base_dir):
    runs = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("run_")
    ])
    return os.path.join(base_dir, runs[-1])

# --------------------------
# IoU
# --------------------------
def iou_score(pred, target, eps=1e-6):
    """
    Intersection over Union.
    0 = no overlap, 1 = perfect overlap.
    """
    pred   = (pred > THRESHOLD).astype(np.float32)
    target = target.astype(np.float32)
    inter  = (pred * target).sum()
    union  = pred.sum() + target.sum() - inter
    return float((inter + eps) / (union + eps))

# --------------------------
# CONE GEOMETRY FROM MASK
# --------------------------
def measure_geometry(binary, grid=GRID):
    """
    Measure PA and opening angle from a binary mask.
    Uses polar angular histogram centered on nucleus (grid center).
    Returns (pa_deg, opening_full_deg) or (None, None).
    """
    if binary.sum() == 0:
        return None, None

    c           = grid // 2
    yy, xx      = np.mgrid[0:grid, 0:grid]
    dx          = (xx - c).astype(np.float32)
    dy          = (yy - c).astype(np.float32)
    r_grid      = np.sqrt(dx**2 + dy**2) + 1e-6
    theta_grid  = np.degrees(np.arctan2(dy, dx))

    valid = (binary > 0.5) & (r_grid > grid * 0.05)
    if valid.sum() == 0:
        return None, None

    n_bins      = 360
    bin_edges   = np.linspace(-180, 180, n_bins + 1)
    counts, _   = np.histogram(theta_grid[valid], bins=bin_edges)
    counts_s    = uniform_filter1d(counts.astype(float), size=5, mode="wrap")
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    above   = counts_s > 0
    doubled = np.concatenate([above, above])
    regions = []
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

    seen, unique = set(), []
    for r in regions:
        if r[0] not in seen:
            seen.add(r[0])
            unique.append(r)
    regions = sorted(unique, key=lambda x: -x[2])

    if not regions:
        return None, None

    s, e, l  = regions[0]
    mid_idx  = (s + l // 2) % n_bins
    pa_deg   = float(bin_centers[mid_idx])
    open_deg = float(l)

    return pa_deg, open_deg

# --------------------------
# MAIN
# --------------------------
def main():
    print("\n--- GEOMETRY EVALUATION ---\n")

    run_dir    = get_latest_run(BASE_RESULTS)
    model_path = os.path.join(run_dir, "models", "best.pth")
    print(f"Run:   {run_dir}")
    print(f"Model: {model_path}\n")

    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    dataset = IonizationConeDataset2D(
        image_dir=os.path.join(DATASET_ROOT, "images"),
        mask_dir=os.path.join(DATASET_ROOT, "masks"),
        normalize=False
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"Val samples: {len(dataset)}\n")

    # accumulators
    ious        = []
    delta_pas   = []   # |pred PA - gt PA| per sample
    delta_opens = []   # |pred opening - gt opening| per sample
    no_pred     = 0    # samples where model predicted nothing
    no_gt       = 0    # samples with empty ground truth (no-AGN cases)

    print(f"{'Sample':>8}  {'IoU':>6}  {'|ΔPA|':>7}  {'|Δopen|':>8}  {'Note':>12}")
    print("-" * 55)

    with torch.no_grad():
        for i, (img, mask) in enumerate(loader):
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            pred      = torch.sigmoid(model(img))
            binary    = (pred > THRESHOLD).float()

            pred_np = binary.squeeze().cpu().numpy()
            mask_np = mask.squeeze().cpu().numpy()

            iou = iou_score(pred_np, mask_np)
            ious.append(iou)

            pred_pa, pred_open = measure_geometry(pred_np)
            gt_pa,   gt_open   = measure_geometry(mask_np)

            # skip no-AGN ground truth samples
            if gt_pa is None:
                no_gt += 1
                print(f"{i:>8}  {iou:>6.3f}  {'--':>7}  {'--':>8}  {'no-AGN GT':>12}")
                continue

            if pred_pa is None:
                no_pred += 1
                print(f"{i:>8}  {iou:>6.3f}  {'--':>7}  {'--':>8}  {'no predict':>12}")
                continue

            # PA error — wrap-aware so 179° and -179° = 2° apart not 358°
            delta_pa   = abs(pred_pa - gt_pa)
            delta_pa   = min(delta_pa, 360 - delta_pa)
            delta_open = abs(pred_open - gt_open)

            delta_pas.append(delta_pa)
            delta_opens.append(delta_open)

            print(f"{i:>8}  {iou:>6.3f}  {delta_pa:>7.1f}  {delta_open:>8.1f}")

    print("-" * 55)

    iou_arr   = np.array(ious)
    dpa_arr   = np.array(delta_pas)
    dopen_arr = np.array(delta_opens)

    print(f"\n── Summary ─────────────────────────────────────────────")
    print(f"  Total samples:           {len(ious)}")
    print(f"  No-AGN ground truth:     {no_gt}  (skipped)")
    print(f"  No prediction made:      {no_pred}")
    print(f"  Valid comparisons:       {len(dpa_arr)}")
    print()
    print(f"  Mean IoU:                {iou_arr.mean():.3f} ± {iou_arr.std():.3f}")
    print(f"  Median IoU:              {np.median(iou_arr):.3f}")
    print(f"  Samples with IoU > 0.5:  {(iou_arr > 0.5).sum()} / {len(iou_arr)}")
    print()
    if len(dpa_arr) > 0:
        print(f"  Mean |ΔPA|:              {dpa_arr.mean():.1f}° ± {dpa_arr.std():.1f}°")
        print(f"  Median |ΔPA|:            {np.median(dpa_arr):.1f}°")
        print(f"  PA within 10°:           {(dpa_arr < 10).sum()} / {len(dpa_arr)}")
        print(f"  PA within 20°:           {(dpa_arr < 20).sum()} / {len(dpa_arr)}")
    print()
    if len(dopen_arr) > 0:
        print(f"  Mean |Δopening angle|:   {dopen_arr.mean():.1f}° ± {dopen_arr.std():.1f}°")
        print(f"  Median |Δopening|:       {np.median(dopen_arr):.1f}°")
        print(f"  Opening within 10°:      {(dopen_arr < 10).sum()} / {len(dopen_arr)}")
        print(f"  Opening within 20°:      {(dopen_arr < 20).sum()} / {len(dopen_arr)}")
    print(f"────────────────────────────────────────────────────────\n")

    results = {
        "n_samples":              len(ious),
        "n_valid_comparisons":    len(dpa_arr),
        "n_no_agn_gt":            no_gt,
        "n_no_prediction":        no_pred,
        "mean_iou":               float(iou_arr.mean()),
        "std_iou":                float(iou_arr.std()),
        "median_iou":             float(np.median(iou_arr)),
        "samples_iou_gt_0.5":     int((iou_arr > 0.5).sum()),
        "mean_delta_pa_deg":      float(dpa_arr.mean())    if len(dpa_arr) > 0   else None,
        "std_delta_pa_deg":       float(dpa_arr.std())     if len(dpa_arr) > 0   else None,
        "median_delta_pa_deg":    float(np.median(dpa_arr)) if len(dpa_arr) > 0  else None,
        "pa_within_10deg":        int((dpa_arr < 10).sum()) if len(dpa_arr) > 0  else None,
        "pa_within_20deg":        int((dpa_arr < 20).sum()) if len(dpa_arr) > 0  else None,
        "mean_delta_open_deg":    float(dopen_arr.mean())   if len(dopen_arr) > 0 else None,
        "std_delta_open_deg":     float(dopen_arr.std())    if len(dopen_arr) > 0 else None,
        "median_delta_open_deg":  float(np.median(dopen_arr)) if len(dopen_arr) > 0 else None,
        "opening_within_10deg":   int((dopen_arr < 10).sum()) if len(dopen_arr) > 0 else None,
        "opening_within_20deg":   int((dopen_arr < 20).sum()) if len(dopen_arr) > 0 else None,
    }

    out_path = os.path.join(SAVE_DIR, "geometry_eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out_path}\n")

if __name__ == "__main__":
    main()
