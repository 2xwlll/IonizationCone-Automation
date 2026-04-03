#!/usr/bin/env python3
"""
align_and_ratio_cubes_report_partial.py — Aligns MIRI cubes, computes line ratios,
and reports which slices were used, empty, out-of-range, or partially covered.
"""

import os
import glob
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
import matplotlib.pyplot as plt

RAW_DIR = "data/cubes/raw/real"
OUT_DIR = "data/cubes/processed/real/aligned_ratios_partial"

LINES = {
    "Pf_alpha_7.46": 7.46,
    "NeV_14.32": 14.32,
    "NeII_12.81": 12.81,
    "OIV_25.89": 25.89,
}
RATIOS = [("NeV_14.32", "NeII_12.81"), ("OIV_25.89", "NeII_12.81")]

PARTIAL_MARGIN = 2  # number of spectral slices to consider "partial coverage"


def load_cube(file):
    hdul = fits.open(file)
    # Look for SCI extension
    for hdu in hdul:
        if hdu.name == "SCI":
            data = hdu.data.astype(float)
            wcs = WCS(hdu.header)
            header = hdu.header
            hdul.close()
            return data, wcs, header
    raise ValueError(f"No SCI extension found in {file}")

def find_closest_slice(spec_wave, target_um):
    if target_um < spec_wave.min() or target_um > spec_wave.max():
        # Partial coverage if within PARTIAL_MARGIN slices from edge
        if target_um < spec_wave.min() + PARTIAL_MARGIN * (spec_wave[1] - spec_wave[0]) \
           or target_um > spec_wave.max() - PARTIAL_MARGIN * (spec_wave[1] - spec_wave[0]):
            return -1  # flag for partial coverage
        return None
    idx = np.argmin(np.abs(spec_wave - target_um))
    return idx

def align_to_reference(data, wcs, ref_wcs, shape):
    aligned, footprint = reproject_interp((data, wcs), ref_wcs, shape_out=shape)
    return aligned

def main():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.fits")))
    if not files:
        print(f"No cubes found in {RAW_DIR}")
        return

    # Pick first cube as reference
    ref_data, ref_wcs, ref_header = load_cube(files[0])
    ny, nx = ref_data.shape[1], ref_data.shape[2]

    os.makedirs(OUT_DIR, exist_ok=True)

    for lname, lam in LINES.items():
        print(f"\nProcessing line: {lname} ({lam} μm)")
        slices = []
        used_info = []

        for f in files:
            data, wcs, header = load_cube(f)
            nspec = data.shape[0]  # spectral axis first
            crval3 = header.get("CRVAL3", 1.0)
            cdelt3 = header.get("CDELT3", 1.0)
            crpix3 = header.get("CRPIX3", 1.0)
            spec_wave = (np.arange(nspec) - (crpix3 - 1)) * cdelt3 + crval3

            idx = find_closest_slice(spec_wave, lam)
            if idx is None:
                used_info.append((os.path.basename(f), "OUT OF RANGE"))
                continue

            if idx == -1:
                used_info.append((os.path.basename(f), "PARTIAL COVERAGE"))
                continue

            slice_data = data[idx]
            if np.all(slice_data == 0) or np.isnan(slice_data).all():
                used_info.append((os.path.basename(f), "EMPTY/ALL ZERO"))
                continue

            aligned = align_to_reference(slice_data, wcs.celestial, ref_wcs.celestial, (ny, nx))
            slices.append(aligned)
            used_info.append((os.path.basename(f), f"slice idx {idx}"))

        # Report which cubes contributed
        print("Cubes contributing to this line:")
        for fname, status in used_info:
            print(f"  {fname}: {status}")

        if not slices:
            print(f"No valid slices found for {lname}, skipping saving.")
            continue

        stack = np.nanmean(np.array(slices), axis=0)
        fits.PrimaryHDU(stack, header=ref_wcs.to_header()).writeto(
            f"{OUT_DIR}/{lname}_aligned.fits", overwrite=True
        )

    # Compute ratios
    for num, den in RATIOS:
        try:
            num_data = fits.getdata(f"{OUT_DIR}/{num}_aligned.fits")
            den_data = fits.getdata(f"{OUT_DIR}/{den}_aligned.fits")
        except FileNotFoundError:
            print(f"Skipping ratio {num}/{den} — one of the aligned files missing")
            continue

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.true_divide(num_data, den_data)
            ratio[~np.isfinite(ratio)] = np.nan  # turn inf and NaN into NaN

        fits.PrimaryHDU(ratio, header=ref_wcs.to_header()).writeto(
            f"{OUT_DIR}/{num}_over_{den}.fits", overwrite=True
        )

        plt.imshow(ratio, origin="lower", cmap="viridis")
        plt.colorbar(label=f"{num}/{den}")
        plt.title(f"{num}/{den}")
        plt.savefig(f"{OUT_DIR}/{num}_over_{den}.png", dpi=150)
        plt.close()
        print(f"Saved ratio {num}/{den}")

if __name__ == "__main__":
    main()
