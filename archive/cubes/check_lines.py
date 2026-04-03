from astropy.io import fits
import numpy as np
import os

# --- CONFIG ---
# Path to FITS cubes, relative to project root
FITS_DIR = os.path.join("data", "cubes", "raw", "real")

# Emission lines to look for (µm)
LINES = {
    "FeII_5.34": 5.34,
    "FeVIII_5.45": 5.45,
    "MgVII_5.50": 5.50,
    "MgV_5.61": 5.61,
    "ArII_6.99": 6.99,
    "NaIII_7.32": 7.32,
    "Pf_alpha_7.46": 7.46,
    "NeVI_7.65": 7.65,
    "FeVII_7.82": 7.82,
    "ArV_7.90": 7.90,
    "ArIII_8.99": 8.99,
    "FeVII_9.53": 9.53,
    "SIV_10.51": 10.51,
    "Hu_alpha_12.37": 12.37,
    "NeII_12.81": 12.81,
    "ArV_13.10": 13.10,
    "NeV_14.32": 14.32,
    "NeIII_15.56": 15.56,
    "SIII_18.71": 18.71,
    "ArIII_21.83": 21.83,
    "NeV_24.32": 24.32,
    "OIV_25.89": 25.89,
    "FeII_25.99": 25.99
}

# --- SCRIPT ---

# Resolve full absolute path
FITS_DIR = os.path.abspath(FITS_DIR)

if not os.path.isdir(FITS_DIR):
    raise FileNotFoundError(f"Directory does not exist: {FITS_DIR}")

fits_files = [
    os.path.join(FITS_DIR, f)
    for f in os.listdir(FITS_DIR)
    if f.lower().endswith(('.fits', '.fits.gz'))
]

if not fits_files:
    raise FileNotFoundError(f"No FITS files found in: {FITS_DIR}")

for fname in fits_files:
    print(f"\nChecking {os.path.basename(fname)}...")

    try:
        with fits.open(fname) as hdul:
            hdr = hdul[1].header

            nwave = hdr["NAXIS3"]
            base = hdr["CRVAL3"]
            delta = hdr["CDELT3"]
            refpix = hdr["CRPIX3"]

            wavelengths = base + (np.arange(nwave) + 1 - refpix) * delta
            wmin, wmax = wavelengths.min(), wavelengths.max()

            print(f"  Spectral range: {wmin:.3f} → {wmax:.3f} µm")

            lines_in_cube = [
                f"{name} ({wl:.2f} µm)"
                for name, wl in LINES.items()
                if wmin <= wl <= wmax
            ]

            if lines_in_cube:
                print("  Emission lines inside range:")
                for L in lines_in_cube:
                    print("   -", L)
            else:
                print("  No emission lines within this cube's spectral range.")

    except Exception as e:
        print(f"  ERROR: {e}")
