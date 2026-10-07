#!/usr/bin/env python3
"""Create a compact wavelength-cube and mask diagnostic figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cube", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("results/cloudy/ngc1068_oiii_cube_preview.png"))
    args = parser.parse_args()
    with np.load(args.cube) as data:
        cube = data["input"]
        mask = data["mask"]
        wavelength = data["wavelength_angstrom"]

    spectrum = cube.sum(axis=(1, 2))
    blue = cube[wavelength < 5025.83].sum(axis=0)
    red = cube[wavelength >= 5025.83].sum(axis=0)
    integrated = cube.sum(axis=0)
    images = [integrated, blue, red, mask]
    titles = ["All wavelengths", "Blue channels", "Red channels", "Geometric mask"]

    fig, axes = plt.subplots(2, 4, figsize=(14, 6), constrained_layout=True)
    for axis, image, title in zip(axes[0], images, titles):
        axis.imshow(image, origin="lower", cmap="magma" if title != "Geometric mask" else "gray")
        axis.set_title(title)
        axis.set_axis_off()
    axes[1, 0].plot(wavelength, spectrum, color="tab:green")
    axes[1, 0].set(xlabel="Observed wavelength (Å)", ylabel="Summed relative flux")
    axes[1, 0].grid(alpha=0.2)
    selected = np.linspace(0, len(wavelength) - 1, 3, dtype=int)
    for axis, channel in zip(axes[1, 1:], selected):
        axis.imshow(cube[channel], origin="lower", cmap="magma")
        axis.set_title(f"{wavelength[channel]:.1f} Å")
        axis.set_axis_off()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    print(args.output)


if __name__ == "__main__":
    main()
