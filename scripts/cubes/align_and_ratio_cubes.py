#!/usr/bin/env python3
"""
science_align_and_ratio.py

Science-first pipeline:
- Select only cubes that truly cover each emission line
- Slice at the target wavelength
- Reproject to a common WCS
- Average only valid contributors
- Compute trusted ratio maps
"""

import os
import glob
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------

RAW_DIR = "data/cubes/raw/real"
OUT_DIR = "data/cubes/processed/real/science_ratios"

LINES = {
    "NeV_14.32": 14.32,
    "NeII_12.81": 12.81,
    "OIV_25.89": 25.89,
}

RATIOS = [
    ("NeV_14.32", "NeII_12.81"),
    ("OIV_25.89", "NeII_12.81"),
]

MIN_VALID_PIXELS = 50   # science sanity check

# ----------------------------------------


def load_cube(file):
    with fits.open(file) as hdul:
        hdu = hdul["SCI"]
        data = hdu.data.astype(float)
        header = hdu.header
        wcs = WCS(header)
    return data, header, wcs


def build_wavelength_axis(header, nspec):
    crval = header.get("CRVAL3")
    cdelt = header.get("CDELT3")
    crpix = header.get("CRPIX3")

    if crval is None or cdelt is None or crpix is None:
        return None

    return (np.arange(nspec) - (crpix - 1)) * cdelt + crval


def extract_valid_slice(data, header, target_um):
    nspec = data.shape[0]
    wave = build_wavelength_axis(header, nspec)
    if wave is None:
        return None

    idx = np.argmin(np.abs(wave - target_um))
    slice_2d = data[idx]

    if not np.isfinite(slice_2d).any():
        return None

    if np.count_nonzero(np.isfinite(slice_2d)) < MIN_VALID_PIXELS:
        return None

    return slice_2d


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.fits")))

    if not files:
        raise RuntimeError("No FITS cubes found")

    # Use first cube as WCS reference
    ref_data, ref_header, ref_wcs = load_cube(files[0])
    ny, nx = ref_data.shape[1:]

    aligned_maps = {}

    # -------- LINE MAPS --------
    for lname, lam in LINES.items():
        print(f"\nProcessing {lname} ({lam} µm)")
        slices = []

        for f in files:
            data, header, wcs = load_cube(f)
            slice_2d = extract_valid_slice(data, header, lam)

            if slice_2d is None:
                continue

            aligned, _ = reproject_interp(
                (slice_2d, wcs.celestial),
                ref_wcs.celestial,
                shape_out=(ny, nx),
            )

            slices.append(aligned)

        if len(slices) == 0:
            print("  ❌ No valid contributors — skipping")
            continue

        stack = np.nanmean(slices, axis=0)
        aligned_maps[lname] = stack

        fits.PrimaryHDU(
            stack,
            header=ref_wcs.celestial.to_header()
        ).writeto(f"{OUT_DIR}/{lname}.fits", overwrite=True)

        print(f"  ✅ {len(slices)} cubes contributed")

    # -------- RATIOS --------
    for num, den in RATIOS:
        if num not in aligned_maps or den not in aligned_maps:
            print(f"Skipping ratio {num}/{den} (missing data)")
            continue

        ratio = aligned_maps[num] / aligned_maps[den]
        ratio[~np.isfinite(ratio)] = np.nan

        fits.PrimaryHDU(
            ratio,
            header=ref_wcs.celestial.to_header()
        ).writeto(f"{OUT_DIR}/{num}_over_{den}.fits", overwrite=True)

        plt.imshow(ratio, origin="lower", cmap="inferno")
        plt.colorbar(label=f"{num}/{den}")
        plt.title(f"{num}/{den}")
        plt.savefig(f"{OUT_DIR}/{num}_over_{den}.png", dpi=150)
        plt.close()

        print(f"  📈 Saved ratio {num}/{den}")

    print("\nScience pipeline complete.")


if __name__ == "__main__":
    main()
