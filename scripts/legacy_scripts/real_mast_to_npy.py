# scripts/convert_fits_to_npy.py

import numpy as np
from astropy.io import fits
from pathlib import Path
from skimage.transform import resize
import os

# --- Configurable Paths ---
RAW_DIR = Path("data/mast_raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Settings ---
TARGET_SHAPE = (128, 128)  # UNet input shape
NORMALIZE = True           # Normalize pixel values

# --- Conversion Script ---

def process_fits_file(fits_path):
    try:
        with fits.open(fits_path) as hdul:
            # Prefer SCI extension, fallback to PRIMARY
            data = None
            for ext in ("SCI", 0):
                try:
                    data = hdul[ext].data
                    if data is not None:
                        break
                except Exception:
                    continue

            if data is None:
                print(f"[SKIP] No usable data in {fits_path.name}")
                return

            # Squeeze and drop any NaNs
            data = np.nan_to_num(np.squeeze(data))

            # Resize to 128x128 if needed
            if data.shape != TARGET_SHAPE:
                data = resize(data, TARGET_SHAPE, mode='reflect', anti_aliasing=True)

            # Normalize to [0, 1] range
            if NORMALIZE:
                data_min, data_max = data.min(), data.max()
                if data_max > data_min:
                    data = (data - data_min) / (data_max - data_min)

            # Save
            out_name = OUT_DIR / (fits_path.stem + ".npy")
            np.save(out_name, data)
            print(f"[OK] Saved {out_name.name}")

    except Exception as e:
        print(f"[ERR] {fits_path.name}: {e}")


if __name__ == "__main__":
    files = list(RAW_DIR.glob("*.fits"))
    print(f"Found {len(files)} FITS files.")
    for f in files:
        process_fits_file(f)

