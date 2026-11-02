#!/usr/bin/env python3
"""Resample cached Cloudy line-free continuum; preserve existing line intensities.

No Cloudy models are rerun. Both the source grid and raw .con files are needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_oiii_grid import parse_emergent_continuum


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid", type=Path,
                   default=Path("data/cloudy/ngc1068_oiii_full_v1/grid.npz"))
    p.add_argument("--config", type=Path,
                   default=Path("configs/cloudy/ngc1068_f502n_v2.json"))
    p.add_argument("--output", type=Path,
                   default=Path("data/cloudy/ngc1068_f502n_v2/grid.npz"))
    args = p.parse_args()
    if args.output.resolve() == args.grid.resolve() or args.output.exists():
        p.error("Select a new output grid; cached source data will not be overwritten")
    config = json.loads(args.config.read_text())
    with np.load(args.grid) as d:
        arrays = {k: d[k] for k in d.files}
    physics = config["cloudy"]
    for key in ("log_u", "log_nh_cm3", "metallicity_solar"):
        if not np.array_equal(np.unique(arrays[key]), sorted(physics[key])):
            raise ValueError(f"Config changes {key}; rerun Cloudy for new physical conditions")
    for key in ("dust_scale", "pah_scale"):
        requested = sorted({g[key] for g in physics["grain_models"]})
        if not np.array_equal(np.unique(arrays[key]), requested):
            raise ValueError(f"Config changes {key}; rerun Cloudy for new grain conditions")
    cfg = config["cube"]
    observed = np.linspace(cfg["wavelength_min_angstrom"],
                           cfg["wavelength_max_angstrom"], cfg["wavelength_channels"])
    rest = observed/(1+config["redshift"])
    arrays["continuum_outward_flambda"] = np.stack([
        parse_emergent_continuum(args.grid.parent/"runs"/f"{model}.con", rest)
        for model in arrays["model_id"]])
    arrays["continuum_rest_wavelength_angstrom"] = rest
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    args.output.with_suffix(".json").write_text(json.dumps({
        "source_grid": str(args.grid.resolve()), "config": str(args.config),
        "models": len(arrays["model_id"]), "cloudy_rerun": False,
        "continuum_observed_limits_angstrom": [float(observed[0]), float(observed[-1])],
        "continuum_channels": len(observed),
        "note": "Reused existing physics parameters; only continuum sampling changed",
    }, indent=2))
    print(f"Resampled {len(arrays['model_id'])} cached Cloudy continua to {len(rest)} samples: {args.output}")


if __name__ == "__main__":
    main()
