"""Render intrinsic rest-frame F_lambda cubes through all NIRCam imaging filters."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.jwst_utils.nircam import render_filters


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True,
                   help="NPZ with intrinsic cube and rest_wavelength_angstrom")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--redshift", type=float, required=True)
    p.add_argument("--pixel-scale", type=float, required=True,
                   help="Model grid arcsec/pixel; output retains this grid")
    p.add_argument("--filters", nargs="+", default=None, help="Default: all 29")
    p.add_argument("--input-psf-fwhm", type=float, required=True,
                   help="Must be zero: input must be intrinsic, without telescope blur")
    args = p.parse_args()
    if args.input_psf_fwhm != 0:
        p.error("NIRCam rendering needs an intrinsic cube; cannot undo an existing PSF")
    with np.load(args.input, allow_pickle=False) as data:
        if "metadata" in data:
            metadata = json.loads(str(data["metadata"].item()))
            if float(metadata.get("psf_fwhm_arcsec", 0)) != 0:
                p.error("Input metadata records existing PSF blur; use an intrinsic cube")
        if "rest_wavelength_angstrom" not in data:
            p.error("Input must explicitly contain rest_wavelength_angstrom; observed wavelengths are not interchangeable")
        images, metadata = render_filters(
            data["cube"], data["rest_wavelength_angstrom"], redshift=args.redshift,
            pixel_scale_arcsec=args.pixel_scale, filters=args.filters,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **images, metadata=json.dumps(metadata))
    print(f"Rendered {len(images)} filters on the common model grid: {args.output}")


if __name__ == "__main__":
    main()
