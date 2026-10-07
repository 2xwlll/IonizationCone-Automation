#!/usr/bin/env python3
"""Generate paired morphology previews without writing a training dataset."""
import copy
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


def main():
    with patch.object(sys, "argv", ["generator"]):
        module = runpy.run_path("scripts/2d/generate_realistic_emission.py")
    module["validate_config"]()
    config = module["CONFIG"]
    cloud_config = copy.deepcopy(config["clouds"])
    out = Path("results/synthetic_ionization_cone")
    out.mkdir(parents=True, exist_ok=True)
    seeds = [0, 1, 2, 3, 4, 5]
    fig, axes = plt.subplots(3, len(seeds), figsize=(15, 8))
    for col, seed in enumerate(seeds):
        config.pop("clouds", None)
        old, _, old_params = module["generate_sample"](seed)
        config["clouds"] = cloud_config
        new, mask, params = module["generate_sample"](seed)
        assert all(params[k] == v for k, v in old_params.items())
        for row, field in enumerate((old, new, mask)):
            axes[row, col].imshow(field, origin="lower", cmap="inferno", vmin=0, vmax=1)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        axes[0, col].set_title(f"Seed {seed}")
    for row, label in enumerate(("Previous emission", "Compact clouds", "New soft mask")):
        axes[row, 0].set_ylabel(label)
    fig.tight_layout()
    fig.savefig(out / "cloud_comparison.png", dpi=150)
    plt.close(fig)

    # Isolate morphology from nucleus, dust and noise; same cloud random seed.
    np.random.seed(0)
    params = module["sample_params"](128, 0)
    illum = module["warped_cone"](128, 25, 35, 48, 20, params)
    settings = module["sample_cloud_settings"](0)
    controls = [
        ("count", [12, 30, 75]),
        ("sigma_pixels", [[0.8, 1.5], [1.5, 3], [3, 6]]),
        ("cluster_fraction", [0., .5, 1.]),
        ("luminosity_scatter", [0., .7, 1.4]),
        ("irregularity", [0., .5, 1.]),
        ("diffuse_weight", [0., .3, 1.]),
    ]
    fig, axes = plt.subplots(len(controls), 3, figsize=(9, 17))
    for row, (name, values) in enumerate(controls):
        fields = []
        for value in values:
            varied = dict(settings, **{name: value})
            clouds, diffuse = module["cloud_components"](128, illum, varied, 42)
            field = illum * (
                module["normalize_component"](clouds, illum)
                + varied["diffuse_weight"] * module["normalize_component"](diffuse, illum)
            ) / (1 + varied["diffuse_weight"])
            fields.append(gaussian_filter(field, 1.0))
        vmax = max(float(f.max()) for f in fields)
        for col, (field, value) in enumerate(zip(fields, values)):
            axes[row, col].imshow(field, origin="lower", cmap="inferno", vmin=0, vmax=vmax)
            axes[row, col].set_title(f"{name} = {value}", fontsize=10)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    fig.suptitle("One control per row; fixed geometry; PSF sigma 1 px; shared scale within each row")
    fig.tight_layout(rect=(0, 0, 1, .98))
    fig.savefig(out / "cloud_controls.png", dpi=130)
    plt.close(fig)
    print("Saved cloud_comparison.png and cloud_controls.png")


if __name__ == "__main__":
    main()
