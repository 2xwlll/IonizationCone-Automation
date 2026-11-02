#!/usr/bin/env python3
"""Check and plot wavelength weighting in a rendered 2D F502N pair."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/agn-f502n-matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label

from render_hst_f502n import load_curve
from src.simple_utils.synthetic_photometry import HC_ERG_ANGSTROM, photon_bandpass_weights


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pair", type=Path)
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/cloudy/ngc1068_f502n_v2"))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads(args.pair.with_suffix(".json").read_text())
    with np.load(args.pair) as d:
        observed, mask = d["input"], d["mask"]
        saved_raw = d["filter_integrated_gas_raw"]
        weights = d["photon_integration_weights"]
    with np.load(meta["source_cube"]) as d:
        cube, wave = d["input"], d["wavelength_angstrom"]
    fw, response = load_curve(Path(meta.get("throughput_file",
        "configs/instruments/hst_wfpc2_pc_f502n.csv")))
    independent_weights = photon_bandpass_weights(wave, fw, response)
    np.testing.assert_allclose(weights, independent_weights, rtol=1e-13)
    rebuilt = np.tensordot(independent_weights, cube.astype(np.float64), axes=(0, 0))
    np.testing.assert_allclose(rebuilt, saved_raw, rtol=1e-12, atol=1e-9)
    if observed.shape != mask.shape or observed.ndim != 2:
        raise ValueError("Final model input and target must be matching 2D maps")
    if not np.isfinite(observed).all() or not np.isin(mask, [0, 1]).all():
        raise ValueError("Invalid image or target values")
    if label(mask > 0.5)[1] != 1:
        raise ValueError("Target is not one connected component")
    source = meta["source_metadata"]
    z, c = source["redshift"], 299792.458
    sigma_v = float(source.get("velocity_dispersion_kms", 85.0))
    velocity = np.linspace(-900, 900, 181)
    efficiencies = {}
    line_report = {}
    for name, rest in (("[O III] 4959", 4958.91), ("[O III] 5007", 5006.84)):
        center = rest*(1+z)*(1+velocity/c)
        sigma = rest*(1+z)*sigma_v/c
        profiles = np.exp(-0.5*((wave[:, None]-center[None, :])/sigma)**2)
        profiles /= sigma*np.sqrt(2*np.pi)
        efficiency = (weights@profiles)*HC_ERG_ANGSTROM/center
        efficiencies[name] = efficiency
        middle = len(velocity)//2
        line_report[name] = {
            "observed_center_angstrom": float(center[middle]),
            "point_throughput_at_center": float(np.interp(center[middle], fw, response)),
            "profile_averaged_photon_efficiency": float(efficiency[middle]),
            "efficiency_at_minus_900_kms": float(efficiency[0]),
            "efficiency_at_plus_900_kms": float(efficiency[-1]),
        }
    report = {
        "input_shape": list(observed.shape), "dtype": str(observed.dtype),
        "target_shape": list(mask.shape), "mask_components": 1,
        "tabulated_photon_response_coverage": meta["spectral_integration"]["photon_response_coverage"],
        "raw_response_max_absolute_reconstruction_error": float(np.max(abs(rebuilt-saved_raw))),
        "line_profile_sigma_kms": sigma_v, "line_response": line_report,
        "note": "Relative photometry; Gaussian PSF remains approximate",
    }
    (args.output_dir/"validation.json").write_text(json.dumps(report, indent=2))

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].plot(fw, response, color="tab:blue")
    for (name, info), color in zip(line_report.items(), ("tab:blue", "tab:orange")):
        axes[0, 0].axvline(info["observed_center_angstrom"], linestyle="--", label=name, color=color)
    axes[0, 0].set(xlabel="Observed wavelength (Å)", ylabel="Total-system throughput",
                   title="F502N at NGC 1068 redshift")
    axes[0, 0].legend(fontsize=8)
    for name, efficiency in efficiencies.items():
        axes[0, 1].plot(velocity, efficiency, label=name)
    axes[0, 1].set(xlabel="Line-of-sight velocity (km/s)", ylabel="Profile-averaged efficiency",
                   title="The filter weights moving gas differently")
    axes[0, 1].legend(fontsize=8)
    spatial_spectrum = cube.sum(axis=(1, 2), dtype=np.float64)
    detected_spectrum = spatial_spectrum*weights
    axes[0, 2].plot(wave, detected_spectrum/detected_spectrum.max(), color="tab:green")
    axes[0, 2].set(xlabel="Observed wavelength (Å)", ylabel="Relative weighted contribution",
                   title="Contributions to the narrowband image")
    for ax in axes[0]:
        ax.grid(alpha=0.2)

    def display(a):
        ceiling = max(float(np.percentile(a, 99.7)), 1e-12)
        return np.clip(np.arcsinh(12*np.clip(a, 0, None)/ceiling)/np.arcsinh(12), 0, 1)
    simple = np.trapezoid(cube, wave, axis=0)
    for ax, image, title in zip(axes[1], (simple, saved_raw, observed),
            ("Unweighted gas sum (comparison)", "Photon-weighted F502N gas", "Final 2D image + target boundary")):
        ax.imshow(display(image), origin="lower", cmap="magma", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10); ax.set_axis_off()
    axes[1, 2].contour(mask, levels=[0.5], colors="cyan", linewidths=0.6)
    fig.suptitle("Validated wavelength integration for a 2D F502N training image")
    fig.supxlabel("Images use individual asinh stretches to compare morphology, not absolute photometry.", fontsize=9)
    fig.savefig(args.output_dir/"wavelength_response.png", dpi=150)
    plt.close(fig)
    print(json.dumps(report, indent=2))
    print(f"Saved wavelength diagnostic to {args.output_dir/'wavelength_response.png'}")


if __name__ == "__main__":
    main()
