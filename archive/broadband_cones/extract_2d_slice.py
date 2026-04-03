#!/usr/bin/env python3
"""
extract_2d_broadband_nirspec.py
Integrates NIRSpec cubes over [O III] 5007 Å emission to produce
scientifically accurate 2D broadband images.

- Automatically detects wavelength units (Å or µm)
- Skips cubes that do not cover the observed [O III]
- Outputs FITS with preserved header
"""

import os
import glob
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

# -----------------------
# CONFIG
# -----------------------
RED_SHIFT = 0.0
INPUT_DIR = "data/cubes/raw/real"
OUTPUT_DIR = "data/2d/processed/science"
OIII_REST = 0.5007  # microns
INTEGRATION_WIDTH = 0.003  # microns (~30 Å)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------
# FUNCTIONS
# -----------------------
def load_cube(fits_path):
    with fits.open(fits_path) as hdul:
        data = hdul[1].data
        hdr = hdul[1].header
        wcs = WCS(hdr)
    return data, hdr, wcs

def get_wavelengths(hdr):
    """Return wavelength array in microns."""
    crval3 = hdr.get("CRVAL3")
    cdelt3 = hdr.get("CDELT3")
    crpix3 = hdr.get("CRPIX3", 1)
    naxis3 = hdr.get("NAXIS3")
    if crval3 is None or cdelt3 is None:
        raise ValueError("Missing wavelength axis info in FITS header")
    wavelengths = crval3 + cdelt3 * (np.arange(naxis3) - (crpix3 - 1))
    # Convert Å → µm if wavelength is >1 (likely in Å)
    if np.median(wavelengths) > 1.0:
        wavelengths /= 1e4
    return wavelengths

def integrate_broadband(cube, wavelengths, oiii_obs, width):
    mask = (wavelengths >= oiii_obs - width / 2) & (wavelengths <= oiii_obs + width / 2)
    if not np.any(mask):
        return None
    return np.sum(cube[mask, :, :], axis=0)

def save_fits(image2d, header, out_path):
    hdu = fits.PrimaryHDU(data=image2d, header=header)
    hdu.writeto(out_path, overwrite=True)

# -----------------------
# MAIN
# -----------------------
fits_files = glob.glob(os.path.join(INPUT_DIR, "**/*_s3d.fits"), recursive=True)
print(f"Found {len(fits_files)} total cubes.")

oiii_obs = OIII_REST * (1 + RED_SHIFT)
processed_count = 0

for fpath in fits_files:
    # Skip non-NIRSpec cubes
    if "nirspec" not in fpath.lower():
        continue

    try:
        cube, hdr, wcs = load_cube(fpath)
        wavelengths = get_wavelengths(hdr)
        img2d = integrate_broadband(cube, wavelengths, oiii_obs, INTEGRATION_WIDTH)

        if img2d is None:
            print(f"[SKIP] {fpath}: [O III] not in wavelength range.")
            continue

        base = os.path.splitext(os.path.basename(fpath))[0]
        out_path = os.path.join(OUTPUT_DIR, f"{base}_broadband.fits")
        save_fits(img2d, hdr, out_path)
        processed_count += 1
        print(f"[OK] {base}: integrated [O III] → saved.")

    except Exception as e:
        print(f"[ERROR] {fpath}: {e}")

print(f"Processed {processed_count} NIRSpec cubes.")

