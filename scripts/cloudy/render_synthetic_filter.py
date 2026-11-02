"""Turn an existing narrow wavelength cube into one custom-filter image."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.simple_utils.synthetic_filters import (
    filter_cube, gaussian_passband, top_hat_passband,
)


def load_cube(path):
    metadata = {}
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        metadata = json.loads(sidecar.read_text())
    with np.load(path, allow_pickle=False) as data:
        key = "input" if "input" in data else "cube"
        if key not in data or "wavelength_angstrom" not in data:
            raise ValueError("Input needs input/cube and wavelength_angstrom arrays")
        cube = data[key]
        wave = data["wavelength_angstrom"]
        mask = data["mask"] if "mask" in data else None
        if "metadata" in data:
            metadata.update(json.loads(str(data["metadata"].item())))
    if metadata.get("wavelength_frame", "observed") != "observed":
        raise ValueError("Input must already use observed-frame wavelengths")
    if metadata.get("normalization", {}).get("transform", "linear") != "linear":
        raise ValueError("Use the linear source cube, not log/asinh training data")
    if cube.ndim != 3 or (mask is not None and mask.shape != cube.shape[1:]):
        raise ValueError("Expected (wavelength,y,x) cube and matching optional (y,x) mask")
    return cube, wave, mask, metadata


def preview(path, cube, wave, image, fw, response, description):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import AsinhNorm

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].plot(wave, cube.sum(axis=(1, 2)), color="tab:blue", label="Cube spectrum")
    axes[0].set(xlabel="Observed wavelength (Å)", ylabel="Summed relative Fλ")
    twin = axes[0].twinx()
    plot_wave = np.r_[fw[0], fw, fw[-1]]
    plot_response = np.r_[0, response, 0]
    twin.fill_between(plot_wave, plot_response, color="tab:orange", alpha=0.3)
    twin.set(ylabel="Synthetic throughput", ylim=(0, 1.05))
    axes[0].set_xlim(wave[0], wave[-1])
    axes[0].set_title(description)
    axes[0].grid(alpha=0.2)
    high = max(float(np.percentile(image, 99.8)), float(image.max())*1e-6, 1e-12)
    axes[1].imshow(image, origin="lower", cmap="magma",
                   norm=AsinhNorm(linear_width=high*0.03, vmin=0, vmax=high))
    axes[1].set_title("Synthetic narrowband image\n(asinh display; saved image is linear)")
    axes[1].set_axis_off()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cube", type=Path)
    p.add_argument("--output", type=Path, required=True, help="New output .npz path")
    filters = p.add_mutually_exclusive_group()
    filters.add_argument("--tophat", nargs=2, type=float, metavar=("LOWER_A", "UPPER_A"))
    filters.add_argument("--gaussian", nargs=2, type=float, metavar=("CENTER_A", "FWHM_A"))
    filters.add_argument("--throughput", type=Path,
                         help="Two-column CSV: observed Angstrom, throughput in [0,1]")
    p.add_argument("--preview", type=Path, help="Optional PNG with spectrum, passband and image")
    args = p.parse_args()
    destinations = [args.output, args.output.with_suffix(".json")]
    if args.preview:
        if args.preview.suffix.lower() != ".png":
            p.error("Preview must be a .png")
        destinations.append(args.preview)
    if args.output.suffix.lower() != ".npz":
        p.error("Output must be a .npz")
    if any(path.exists() for path in destinations):
        p.error("Output already exists; choose a new output/preview name")
    cube, wave, mask, source_metadata = load_cube(args.cube)
    if args.throughput:
        curve = np.loadtxt(args.throughput, delimiter=",")
        if curve.ndim != 2 or curve.shape[1] != 2:
            p.error("Throughput CSV must have exactly two columns")
        fw, response = curve.T
        definition = {"shape": "tabulated", "source": str(args.throughput)}
        description = f"Custom table: {args.throughput.name}"
    elif args.gaussian:
        center, fwhm = args.gaussian
        fw, response = gaussian_passband(center, fwhm)
        definition = {"shape": "gaussian", "center_angstrom": center,
                      "fwhm_angstrom": fwhm, "truncate_sigma": 4.0}
        description = f"Gaussian: {center:g} Å, FWHM {fwhm:g} Å"
    else:
        lower, upper = args.tophat if args.tophat else (wave[0], wave[-1])
        fw, response = top_hat_passband(lower, upper)
        definition = {"shape": "tophat", "lower_angstrom": float(lower),
                      "upper_angstrom": float(upper)}
        description = f"Top-hat: {lower:g}–{upper:g} Å"
    image, weights, coverage = filter_cube(cube, wave, fw, response)
    metadata = {
        "source_cube": str(args.cube), "source_metadata": source_metadata,
        "filter": definition, "coverage": coverage, "wavelength_frame": "observed",
        "wavelength_units": "Angstrom", "axis_order": ["y", "x"],
        "normalization": {"transform": "linear", "per_image_rescaling": False},
        "response_units": "relative photon response for normalized source cubes",
        "spatial_psf": "inherited from source cube; no additional spatial convolution",
        "continuum": "included if present in source; no continuum subtraction",
    }
    arrays = {"image": image, "weights": weights, "wavelength_angstrom": wave,
              "filter_wavelength_angstrom": fw, "throughput": response,
              "metadata": json.dumps(metadata)}
    if mask is not None:
        arrays["mask"] = mask
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.preview:
        preview(args.preview, cube, wave, image, fw, response, description)
    np.savez_compressed(args.output, **arrays)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(f"Saved {image.shape} synthetic-filter image: {args.output}")


if __name__ == "__main__":
    main()
