#!/usr/bin/env python3
"""
extract_line_slices.py — Extracts line slices from MIRI cubes and flags partial coverage.
"""

import os
import glob
import numpy as np
from astropy.io import fits

# ---------- CONFIG ----------
RAW_DIR = "data/cubes/raw/real"
OUT_DIR = "data/cubes/processed/real/line_slices"
LINES = {
    "Pf_alpha_7.46": 7.46,
    "NeV_14.32": 14.32,
    "NeII_12.81": 12.81,
    "OIV_25.89": 25.89,
}
# -----------------------------

def load_cube(file):
    """Load FITS cube, return data and header."""
    hdu = fits.open(file)[0]
    if hdu.data is None:
        raise ValueError(f"{file} has no data!")
    return hdu.data.astype(float), hdu.header

def find_closest_slice(spec_wave, target_um):
    """Return index of closest spectral slice, or None if outside range."""
    if target_um < spec_wave.min() or target_um > spec_wave.max():
        return None
    return np.argmin(np.abs(spec_wave - target_um))

def main():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.fits")))
    if not files:
        print(f"No cubes found in {RAW_DIR}")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    for lname, lam in LINES.items():
        print(f"\n--- Processing line: {lname} ({lam} μm) ---")
        for f in files:
            try:
                data, header = load_cube(f)
            except ValueError as e:
                print(e)
                continue

            # Build spectral axis
            crval3 = header.get("CRVAL3", 1.0)
            cdelt3 = header.get("CDELT3", 1.0)
            crpix3 = header.get("CRPIX3", 1.0)
            nspec = data.shape[0]  # assume spectral axis first
            spec_wave = (np.arange(nspec) - (crpix3 - 1)) * cdelt3 + crval3

            idx = find_closest_slice(spec_wave, lam)

            if idx is None:
                status = "PARTIAL COVERAGE"
                slice_data = np.full(data.shape[1:], np.nan)
            else:
                slice_data = data[idx]
                if np.isnan(slice_data).all() or np.all(slice_data == 0):
                    status = "EMPTY/ALL ZERO"
                else:
                    status = f"slice idx {idx}"

            print(f"{os.path.basename(f)} -> {status}")

            # Save slice
            out_file = os.path.join(
                OUT_DIR,
                f"{os.path.basename(f).replace('.fits','')}_{lname}.fits"
            )
            fits.PrimaryHDU(slice_data, header=header).writeto(out_file, overwrite=True)

if __name__ == "__main__":
    main()

