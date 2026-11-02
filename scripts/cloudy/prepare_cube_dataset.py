#!/usr/bin/env python3
"""Prepare channel-first Cloudy cubes for segmentation without filter integration.

Keeps the source NPZ archives untouched and saves an invertible shared-scale
asinh training copy plus the original wavelength axis and normalization scale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path,
                        default=Path("data/cloudy/ngc1068_f502n_gif_v1"))
    parser.add_argument("--pattern", default="cube_*.npz")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/cloudy/ngc1068_spectral_training_v1"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--scale-percentile", type=float, default=99.8)
    parser.add_argument("--asinh-softening", type=float, default=0.01,
                        help="Softening relative to the shared cube flux percentile")
    args = parser.parse_args()
    if not (0 < args.val_fraction < 1 and 0 < args.test_fraction < 1
            and args.val_fraction+args.test_fraction < 1):
        parser.error("Validation/test fractions must be positive and sum to < 1")
    if args.asinh_softening <= 0 or not 0 < args.scale_percentile <= 100:
        parser.error("Softening must be positive and percentile must be in (0,100]")
    paths = sorted(args.source_dir.glob(args.pattern))
    if len(paths) < 3:
        parser.error("Need at least three independent cubes for train/val/test")
    if (args.output_dir/"manifest.json").exists():
        parser.error("Dataset already has a manifest; select a new output directory")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty to avoid mixing dataset versions")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(paths))
    nval = max(1, round(len(paths)*args.val_fraction))
    ntest = max(1, round(len(paths)*args.test_fraction))
    if nval+ntest >= len(paths):
        parser.error("Fractions leave no training samples")
    split_for = {int(i): "val" for i in order[:nval]}
    split_for.update({int(i): "test" for i in order[nval:nval+ntest]})
    split_for.update({int(i): "train" for i in order[nval+ntest:]})

    samples, digests, seen_seeds = [], set(), set()
    reference_wave = reference_shape = None
    for i, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as data:
            cube = data["input"].astype(np.float32)
            mask = data["mask"]
            wave = data["wavelength_angstrom"].astype(np.float64)
        if cube.ndim != 3 or cube.shape[0] != len(wave):
            raise ValueError(f"Expected (wavelength,y,x) cube, got {cube.shape}: {path}")
        if mask.shape != cube.shape[1:] or not np.isin(mask, [0, 1]).all():
            raise ValueError(f"Invalid binary mask: {path}")
        if label(mask > 0.5)[1] != 1:
            raise ValueError(f"Cone mask must have one connected component: {path}")
        if not np.isfinite(cube).all() or not np.isfinite(wave).all():
            raise ValueError(f"Nonfinite cube or wavelengths: {path}")
        if not np.all(np.diff(wave) > 0):
            raise ValueError(f"Wavelengths must increase: {path}")
        if reference_wave is None:
            reference_wave, reference_shape = wave, cube.shape
        if cube.shape != reference_shape or not np.array_equal(wave, reference_wave):
            raise ValueError(f"All samples must share shape and wavelength grid: {path}")
        digest = hashlib.sha256(cube.tobytes()).hexdigest()
        if digest in digests:
            raise ValueError(f"Duplicate cube would leak across splits: {path}")
        digests.add(digest)
        source_metadata = json.loads(path.with_suffix(".json").read_text())
        source_seed = source_metadata.get("seed")
        if source_seed is None or source_seed in seen_seeds:
            raise ValueError(f"Each source cube must have a unique generation seed: {path}")
        seen_seeds.add(source_seed)

        shared_scale = float(np.percentile(np.abs(cube), args.scale_percentile))
        if shared_scale <= 0:
            raise ValueError(f"Cube has no usable flux: {path}")
        softening = shared_scale*args.asinh_softening
        transformed = np.arcsinh(cube.astype(np.float64)/softening)
        stored = transformed.astype(args.dtype)
        inverse = (np.sinh(stored.astype(np.float64))*softening).astype(np.float32)
        max_flux_error_over_scale = float(np.max(np.abs(inverse-cube))/shared_scale)
        split = split_for[i]
        base = args.output_dir/split
        for dirname in ("images", "masks", "metadata"):
            (base/dirname).mkdir(parents=True, exist_ok=True)
        np.save(base/"images"/f"{path.stem}.npy", stored)
        np.save(base/"masks"/f"{path.stem}.npy", mask.astype(np.uint8))
        record = {
            "id": path.stem, "split": split, "seed": source_seed,
            "source_cube": str(path.resolve()), "source_sha256": digest,
            "shape": list(cube.shape), "dtype": args.dtype,
            "normalization": {
                "transform": "asinh(input/softening)",
                "inverse": "sinh(training_tensor)*softening",
                "scope": "one scale across all wavelengths and spatial pixels",
                "shared_flux_scale": shared_scale, "softening": softening,
                "scale_percentile": args.scale_percentile,
                "max_inverse_error_over_scale": max_flux_error_over_scale,
            },
            "source_metadata": source_metadata,
        }
        (base/"metadata"/f"{path.stem}.json").write_text(json.dumps(record, indent=2))
        samples.append(record)
        print(f"{split}: {path.stem} -> {stored.shape} {stored.dtype}", flush=True)

    np.save(args.output_dir/"wavelength_angstrom.npy", reference_wave)
    manifest = {
        "axis_order": ["wavelength", "y", "x"],
        "shape": list(reference_shape), "dtype": args.dtype,
        "wavelength_file": "wavelength_angstrom.npy", "split_seed": args.seed,
        "split_counts": {s: sum(r["split"] == s for r in samples)
                         for s in ("train", "val", "test")},
        "normalization": "shared-scale asinh; no wavelength integration",
        "source_units": "normalized generator flux; not absolutely calibrated physical flux",
        "validation_scope": "independent seeds drawn from the same Cloudy grid",
        "samples": samples,
    }
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved spectral dataset: {args.output_dir}; {manifest['split_counts']}")


if __name__ == "__main__":
    main()
