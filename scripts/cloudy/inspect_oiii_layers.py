#!/usr/bin/env python3
"""Export and animate each wavelength layer in an [O III] spectral cube."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

C_KMS = 299792.458


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("cube", type=Path, help="NPZ produced by generate_oiii_cube.py")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/cloudy/oiii_layers"))
    parser.add_argument("--duration-ms", type=int, default=180)
    parser.add_argument("--percentile", type=float, default=99.7,
                        help="Fixed display ceiling across every wavelength layer")
    parser.add_argument("--linear", action="store_true",
                        help="Use linear display instead of the default asinh stretch")
    parser.add_argument("--per-layer-scale", action="store_true",
                        help="Scale each layer separately to reveal faint continuum")
    return parser.parse_args()


def nearest_line_label(wavelength: float, names: np.ndarray,
                       rest_wavelengths: np.ndarray, redshift: float) -> str:
    observed = rest_wavelengths * (1.0 + redshift)
    velocities = C_KMS * (wavelength / observed - 1.0)
    nearest = int(np.argmin(np.abs(velocities)))
    if abs(velocities[nearest]) > 1800:
        return "continuum / between-line channel"
    readable = str(names[nearest]).replace("OIII_", "[O III] ")
    return f"{readable}: {velocities[nearest]:+.0f} km s⁻¹"


def main() -> None:
    args = parse_args()
    with np.load(args.cube) as data:
        cube = data["input"].astype(np.float64)
        mask = data["mask"]
        wavelength = data["wavelength_angstrom"]
        names = data["rest_line_names"]
        rest_wavelengths = data["rest_line_wavelength_angstrom"]

    metadata_path = args.cube.with_suffix(".json")
    redshift = 0.0
    if metadata_path.exists():
        import json
        redshift = float(json.loads(metadata_path.read_text()).get("redshift", 0.0))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output_dir / "frames"
    frame_dir.mkdir(exist_ok=True)
    spectrum = cube.sum(axis=(1, 2))
    if args.per_layer_scale:
        ceiling = np.percentile(cube, args.percentile, axis=(1, 2))[:, None, None]
        floor = np.percentile(cube, 1.0, axis=(1, 2))[:, None, None]
        ceiling = np.maximum(ceiling, floor + 1e-12)
        scale_scope = "per-layer"
    else:
        ceiling = float(np.percentile(cube, args.percentile))
        floor = float(np.percentile(cube, 1.0))
        ceiling = max(ceiling, floor + 1e-12)
        scale_scope = "fixed"

    if args.linear:
        display_cube = np.clip((cube - floor) / (ceiling - floor), 0, 1)
        stretch_name = f"{scale_scope} linear"
    else:
        scaled = np.clip((cube - floor) / (ceiling - floor), 0, None)
        display_cube = np.arcsinh(8.0 * scaled) / np.arcsinh(8.0)
        display_cube = np.clip(display_cube, 0, 1)
        stretch_name = f"{scale_scope} asinh"

    frame_paths = []
    for index, current_wavelength in enumerate(wavelength):
        fig, (image_ax, spectrum_ax) = plt.subplots(
            1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [1.0, 1.25]}
        )
        image_ax.imshow(display_cube[index], origin="lower", cmap="magma", vmin=0, vmax=1)
        image_ax.contour(mask, levels=[0.5], colors="cyan", linewidths=0.55, alpha=0.65)
        image_ax.set_title(
            f"Layer {index + 1:03d}/{len(wavelength):03d}\n"
            f"{current_wavelength:.2f} Å — "
            f"{nearest_line_label(current_wavelength, names, rest_wavelengths, redshift)}"
        )
        image_ax.set_axis_off()

        spectrum_ax.plot(wavelength, spectrum, color="tab:green", linewidth=1.6)
        spectrum_ax.axvline(current_wavelength, color="tab:red", linewidth=1.4)
        for name, rest in zip(names, rest_wavelengths):
            observed = rest * (1.0 + redshift)
            spectrum_ax.axvline(observed, color="0.35", linestyle="--", alpha=0.55)
            spectrum_ax.text(observed, spectrum.max() * 0.96,
                             str(name).replace("OIII_", "[O III] "),
                             rotation=90, va="top", ha="right", fontsize=8)
        spectrum_ax.set(xlabel="Observed wavelength (Å)",
                        ylabel="Spatially summed relative flux")
        spectrum_ax.set_xlim(wavelength[0], wavelength[-1])
        spectrum_ax.set_ylim(0, spectrum.max() * 1.08)
        spectrum_ax.grid(alpha=0.2)
        fig.suptitle(f"NGC 1068 synthetic [O III] cube — {stretch_name} scale")
        fig.tight_layout()
        frame_path = frame_dir / f"layer_{index:03d}_{current_wavelength:08.2f}A.png"
        fig.savefig(frame_path, dpi=110)
        plt.close(fig)
        frame_paths.append(frame_path)

    frames = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE)
              for path in frame_paths]
    gif_path = args.output_dir / "ngc1068_oiii_wavelength_layers.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=args.duration_ms, loop=0, optimize=False, disposal=2)
    for frame in frames:
        frame.close()

    index_path = args.output_dir / "layers.tsv"
    with index_path.open("w") as handle:
        handle.write("channel\twavelength_angstrom\tdescription\tframe\n")
        for index, (current_wavelength, frame_path) in enumerate(zip(wavelength, frame_paths)):
            handle.write(
                f"{index}\t{current_wavelength:.8f}\t"
                f"{nearest_line_label(current_wavelength, names, rest_wavelengths, redshift)}\t"
                f"{frame_path.name}\n"
            )
    print(f"Exported {len(frame_paths)} layers to {frame_dir}")
    print(f"Animation: {gif_path}")
    print(f"Layer index: {index_path}")


if __name__ == "__main__":
    main()
