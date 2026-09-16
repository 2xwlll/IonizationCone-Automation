#!/usr/bin/env python3
"""Export existing synthetic image/mask arrays as observational inspection PNGs."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render existing .npy image/mask pairs without generating data."
    )
    parser.add_argument("dataset", type=Path, help="Dataset containing split directories")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory (default: DATASET/inspection_pngs)",
    )
    parser.add_argument(
        "--splits", nargs="+", default=("train", "val", "test"),
        help="Dataset splits to export",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum pairs per split")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def load_sample_metadata(dataset):
    path = dataset / "samples.json"
    if not path.exists():
        return {}
    with path.open() as handle:
        samples = json.load(handle)
    return {item["sample_id"]: item for item in samples}


def asinh_stretch(image):
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image)
    background = np.percentile(finite, 5)
    high = np.percentile(finite, 99.5)
    scaled = np.clip((image - background) / max(high - background, 1e-8), 0, None)
    return np.arcsinh(8.0 * scaled) / np.arcsinh(8.0)


def render_pair(image_path, mask_path, output_path, metadata, dpi):
    image = np.load(image_path).astype(np.float32)
    mask = np.load(mask_path).astype(np.float32)
    if image.shape != mask.shape:
        raise ValueError(f"Shape mismatch: {image_path} {image.shape} != {mask.shape}")
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D array: {image_path} has shape {image.shape}")

    stretched = asinh_stretch(image)
    binary_mask = mask > 0.5
    overlay = np.zeros((*image.shape, 4), dtype=np.float32)
    overlay[..., 0] = 1.0
    overlay[..., 3] = binary_mask.astype(np.float32) * 0.32

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), constrained_layout=True)
    panels = (
        (image, "linear [0, 1]", "gray", 0, 1),
        (stretched, "asinh (5–99.5%)", "gray", 0, 1),
        (mask, "soft mask", "inferno", 0, 1),
    )
    for axis, (data, title, cmap, vmin, vmax) in zip(axes[:3], panels):
        axis.imshow(data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")
    axes[3].imshow(stretched, origin="lower", cmap="gray", vmin=0, vmax=1)
    axes[3].imshow(overlay, origin="lower")
    axes[3].set_title("mask > 0.5 overlay")
    axes[3].axis("off")

    label = metadata.get("negative_kind") or ("bicone" if metadata.get("bicone") else "cone")
    foreground = float(binary_mask.mean())
    fig.suptitle(
        f"{image_path.stem} | {label} | foreground={foreground:.3f}", fontsize=10
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    dataset = args.dataset.resolve()
    output = (args.output or dataset / "inspection_pngs").resolve()
    metadata = load_sample_metadata(dataset)
    exported = 0

    for split in args.splits:
        image_dir = dataset / split / "images"
        mask_dir = dataset / split / "masks"
        if not image_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(f"Missing image/mask directories for split: {split}")
        image_paths = sorted(image_dir.glob("*.npy"))
        if args.limit is not None:
            image_paths = image_paths[:args.limit]
        for image_path in image_paths:
            mask_path = mask_dir / image_path.name
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing paired mask: {mask_path}")
            render_pair(
                image_path, mask_path, output / split / f"{image_path.stem}.png",
                metadata.get(image_path.stem, {}), args.dpi,
            )
            exported += 1

    print(f"Exported {exported} existing pairs to {output}")


if __name__ == "__main__":
    main()
