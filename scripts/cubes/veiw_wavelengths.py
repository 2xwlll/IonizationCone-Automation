from astropy.io import fits
import os
import numpy as np

cube_dir = "data/cubes/raw/real"

fits_files = [f for f in os.listdir(cube_dir) if f.endswith(".fits")]

for fname in fits_files:
    path = os.path.join(cube_dir, fname)
    with fits.open(path) as hdul:
        hdr = hdul[1].header

        n = hdr["NAXIS3"]
        crval3 = hdr["CRVAL3"]
        crpix3 = hdr["CRPIX3"]
        cdelt3 = hdr["CDELT3"]

        wavelengths = crval3 + (np.arange(n) + 1 - crpix3) * cdelt3
        print(f"{fname}: {wavelengths.min():.4f} → {wavelengths.max():.4f} μm")

