#!/usr/bin/env python3
"""
test_blank_ratios_v2.py
Diagnose why line ratio maps are blank by verifying:
- Wavelength axis extraction (WMAP, WCS fallback)
- Flux values in spectral windows
- NaN distribution in output
"""

import os, glob
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt

RAW_DIR = "data/cubes/raw/real"

# Lines to test
TEST_LINES = {
    "NeII_12.81": 12.81,
    "NeV_14.32": 14.32,
}

# ---------------------------------------------------------
# JWST-correct wavelength extraction
# ---------------------------------------------------------
def build_spec_wave(hdul):
    """
    Returns spectral axis in microns.

    Priority:
    1. JWST WMAP (MIRI/NIRSpec)
    2. WCS (fallback)
    """

    # ----------- JWST wavelength map method -----------
    if "WMAP" in hdul:
        wmap = hdul["WMAP"].data  # (ny, nx, nspec) or (nspec, ny, nx)
        print("Using WMAP wavelength solution")

        # Ensure spectral axis is first
        if wmap.shape[0] not in [400, 500, 600]:   # crude spectral-length heuristic
            wmap = np.moveaxis(wmap, -1, 0)        # now (nspec, ny, nx)

        # Take spatial median to get a 1D wavelength axis
        spec_wave = np.nanmedian(wmap, axis=(1, 2))

        return spec_wave

    # ----------- WCS fallback (less reliable) -----------
    print("No WMAP found: falling back to WCS wavelength")

    try:
        w = WCS(hdul[0].header, naxis=3)
        spec_wcs = w.sub(['spectral'])
        nspec = hdul[0].data.shape[0]
        pix = np.arange(nspec)
        wave = spec_wcs.wcs_pix2world(pix, 0)[0]
        if np.nanmedian(wave) > 100:
            wave /= 1e4
        return wave
    except Exception as e:
        print("WCS extraction failed:", e)
        return None


# ---------------------------------------------------------
# Line measurement function
# ---------------------------------------------------------
def measure_line_window(spec_wave, spec_flux, center_um, half_width_um=0.02):
    """Returns (flux, error) using a simple continuum-subtraction window."""
    in_line = (spec_wave >= center_um - half_width_um) & (spec_wave <= center_um + half_width_um)

    if in_line.sum() < 1:
        return np.nan, np.nan

    dl = np.nanmedian(np.diff(spec_wave))
    if not np.isfinite(dl) or dl == 0:
        return np.nan, np.nan

    flux = np.nansum(spec_flux[in_line]) * dl
    ferr = np.nanstd(spec_flux[in_line]) * np.sqrt(in_line.sum()) * dl

    return flux, ferr


# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
cube_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.fits*")))
if not cube_files:
    print("No cubes found.")
    exit()

for cube_file in cube_files:
    print("\n" + "="*70)
    print(f"Testing cube: {cube_file}")
    print("="*70)

    hdul = fits.open(cube_file, ignore_missing_end=True)

    # Find 3D data HDU
    data_hdu = None
    for h in hdul:
        if h.data is not None and h.data.ndim == 3:
            data_hdu = h
            break

    if data_hdu is None:
        print("No 3D data found.")
        hdul.close()
        continue

    # Extract spectral axis
    spec_wave = build_spec_wave(hdul)
    if spec_wave is None:
        print("Failed to extract wavelength axis.")
        hdul.close()
        continue

    cube = data_hdu.data
    nspec, ny, nx = cube.shape

    print(f"\nCube shape: (spec={nspec}, y={ny}, x={nx})")
    print(f"Spectral range: {np.nanmin(spec_wave):.4f} → {np.nanmax(spec_wave):.4f} µm")
    print(f"Median dλ: {np.nanmedian(np.diff(spec_wave)):.6f} µm")

    # Plot central spectrum
    j, i = ny // 2, nx // 2
    spec = cube[:, j, i]

    plt.figure(figsize=(10, 4))
    plt.plot(spec_wave, spec, lw=1)
    for name, lam in TEST_LINES.items():
        plt.axvline(lam, color='red', alpha=0.4)
    plt.title(f"Spectrum at center pixel ({j},{i})")
    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Flux")
    plt.tight_layout()
    plt.show()

    # Evaluate flux maps
    for name, lam in TEST_LINES.items():
        print(f"\nTesting line: {name} ({lam} µm)")

        flux_map = np.full((ny, nx), np.nan)
        for y in range(ny):
            for x in range(nx):
                f, _ = measure_line_window(spec_wave, cube[:, y, x], lam)
                flux_map[y, x] = f

        nan_frac = np.isnan(flux_map).mean() * 100
        print(f"  NaN fraction in flux map: {nan_frac:.1f}%")
        print(f"  Flux map min={np.nanmin(flux_map):.4e}, max={np.nanmax(flux_map):.4e}")

        plt.figure(figsize=(5, 5))
        plt.imshow(flux_map, origin='lower')
        plt.colorbar(label=f"{name} flux")
        plt.title(f"{name} Flux Map")
        plt.tight_layout()
        plt.show()

    hdul.close()

