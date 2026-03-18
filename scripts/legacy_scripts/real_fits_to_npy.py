
# scripts/convert_fits_to_npy.py

import numpy as np
from astropy.io import fits
from pathlib import Path
from skimage.transform import resize
import os

# --- Paths ---
RAW_DIR = Path("data/raw/manga_cubes")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Settings ---
TARGET_SHAPE = (128, 128)
NORMALIZE = True
TARGET_WAVELENGTH = 5007  # [O III] line in Ångströms

def process_fits_file(fits_path):
    try:
        with fits.open(fits_path) as hdul:
            # Load data
            flux = hdul["FLUX"].data      # (wavelength, y, x)
            loglam = hdul["WAVE"].data    # (wavelength,)
            lam = 10 ** loglam            # Convert from log(Å) to Å

            # Find nearest slice to 5007 Å
            idx = np.abs(lam - TARGET_WAVELENGTH).argmin()
            slice_oiii = flux[idx, :, :]  # Shape: (y, x)

            # Clean NaNs
            slice_oiii = np.nan_to_num(slice_oiii)

            # Resize
            if slice_oiii.shape != TARGET_SHAPE:
                slice_oiii = resize(slice_oiii, TARGET_SHAPE, mode='reflect', anti_aliasing=True)

            # Normalize to [0, 1]
            if NORMALIZE:
                data_min, data_max = slice_oiii.min(), slice_oiii.max()
                if data_max > data_min:
                    slice_oiii = (slice_oiii - data_min) / (data_max - data_min)

            # Save
            out_name = OUT_DIR / (fits_path.stem.replace("-LOGCUBE", "") + "_oiii.npy")
            np.save(out_name, slice_oiii)
            print(f"[✓] Saved: {out_name.name}")

    except Exception as e:
        print(f"[✗] {fits_path.name}: {e}")

if __name__ == "__main__":
    files = list(RAW_DIR.glob("*.fits.gz"))
    print(f"Found {len(files)} FITS cubes.")
    for f in files:
        process_fits_file(f)

