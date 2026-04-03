#!/usr/bin/env python3
"""
jwst_cube_utils.py

Utilities for robustly reading JWST (and generic) IFU cubes:
- detect spectral axis automatically
- read WMAP wavelength solutions when present
- fall back to WCS / CRVAL/CDELT heuristics
- return cube in (nspec, ny, nx) order and wavelength as either:
    - 1D array (nspec,) representing median wavelength across spaxels, or
    - 3D array (nspec, ny, nx) if requested (full per-spaxel WMAP)
"""

from __future__ import annotations
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from typing import Optional, Tuple

def find_3d_hdu(hdul: fits.HDUList) -> Optional[fits.ImageHDU]:
    """Return the first HDU with 3D data (ImageHDU) or None."""
    for h in hdul:
        if getattr(h, "data", None) is None:
            continue
        if h.data.ndim == 3:
            return h
    return None

def _get_wmap_hdu(hdul: fits.HDUList) -> Optional[fits.ImageHDU]:
    """Return the HDU object for a WMAP-like extension (case-insensitive)."""
    # common names: 'WMAP', 'WAVE', 'WAVEL' etc. Prioritise 'WMAP' if present.
    for h in hdul:
        name = getattr(h, "name", "") or ""
        if name.upper() == "WMAP":
            return h
    # fallback: any ext whose header mentions 'wavelength' or whose shape matches SCI
    for h in hdul:
        name = getattr(h, "name", "") or ""
        if "WAVE" in name.upper() or "WAVEL" in name.upper():
            return h
    # last fallback: an ImageHDU with same shape as SCI but last dimension different - not guaranteed
    return None

def build_spec_wave_from_wmap(hdul: fits.HDUList, return_full_map: bool = False) -> Optional[np.ndarray]:
    """
    Attempt to read WMAP (wavelength map) extension and return:
      - if return_full_map=False -> 1D array (nspec,) median across spaxels
      - if return_full_map=True  -> 3D array (nspec, ny, nx) matching cube order
    Returns None if no usable WMAP found.
    """
    wmap_hdu = _get_wmap_hdu(hdul)
    if wmap_hdu is None:
        return None
    wmap = wmap_hdu.data  # could be (ny, nx, nspec) or (nspec, ny, nx) or similar
    if wmap is None:
        return None
    # Ensure we have numpy array
    wmap = np.asarray(wmap)
    # Identify spectral axis as the longest axis (common for NIRSpec/MIRI)
    spec_axis = int(np.argmax(wmap.shape))
    # Move spectral axis to axis 0 for consistency: (nspec, A, B)
    wmap = np.moveaxis(wmap, spec_axis, 0)
    # Now try to reshape to (nspec, ny, nx) if possible
    if wmap.ndim == 1:
        # degenerate case - can't help
        return None
    # If the remaining axes are (ny, nx) or (nx, ny), we don't strictly care — we just keep them.
    if return_full_map:
        return wmap  # (nspec, ny, nx) maybe with ny/nx swapped; calling code must handle
    # Otherwise return 1D median spectrum across spatial dims
    try:
        spec_wave_1d = np.nanmedian(wmap, axis=(1, 2))
        return spec_wave_1d
    except Exception:
        # maybe only one spatial dimension; fallback to median along axis=1
        spec_wave_1d = np.nanmedian(wmap, axis=1)
        return spec_wave_1d

def build_spec_wave_from_wcs(hdu: fits.ImageHDU) -> Optional[np.ndarray]:
    """Try WCS extraction to form a 1D wavelength array in microns."""
    header = hdu.header
    data = hdu.data
    try:
        w = WCS(header, naxis=3)
        # find spectral axis index in WCS ordering
        # attempt to subselect spectral and convert pixel->world
        naxis_spec = data.shape[0]
        pix = np.arange(naxis_spec)
        spec_wcs = w.sub(['spectral'])
        # wcs_pix2world expects shape (Npix, ) but API differs; using safe route:
        # for simple cases, wcs_pix2world returns (arr,) -> take [0]
        world = spec_wcs.wcs_pix2world(pix, 0)
        # some cases produce (1, N) arrays
        if isinstance(world, tuple) or hasattr(world, '__len__') and np.asarray(world).ndim > 1:
            # try to index 0
            try:
                wave = np.asarray(world)[0]
            except Exception:
                wave = np.asarray(world).ravel()
        else:
            wave = np.asarray(world).ravel()
        # Unit heuristic: if values look like Angstroms (>100), convert to microns
        if np.nanmedian(wave) > 100:
            wave = wave / 1e4
        return wave
    except Exception:
        # fallback to header CRVAL/CDELT heuristics
        crval = header.get('CRVAL3') or header.get('CRVAL1') or header.get('CRVAL')
        cdelt = header.get('CDELT3') or header.get('CD3_3') or header.get('CDELT1') or header.get('CDELT')
        crpix = header.get('CRPIX3') or header.get('CRPIX1') or header.get('CRPIX') or 1
        if crval is None or cdelt is None:
            return None
        naxis_spec = data.shape[0]
        pix = np.arange(naxis_spec)
        wave = (pix - (crpix - 1)) * cdelt + crval
        if np.nanmedian(wave) > 100:
            wave = wave / 1e4
        return wave

def normalize_cube_axes(hdu: fits.ImageHDU, prefer_wmap: bool = True,
                        return_full_wave_map: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray], bool]:
    """
    Normalize a 3D cube HDU into:
      cube: ndarray in shape (nspec, ny, nx)
      spec_wave: None or 1D ndarray (nspec,) or 3D array (nspec, ny, nx) if return_full_wave_map True
      ok: bool indicating whether wavelength solution looks sane

    Steps:
      - try WMAP (preferred)
      - try WCS
      - fallback to longest-axis-as-spectral with header heuristics
    """
    data = hdu.data
    if data is None or data.ndim != 3:
        raise ValueError("HDU must contain 3D data")

    # Try WMAP first
    spec_wave = None
    wmap = None
    try:
        # caller may pass the full hdul; attempt to use it
        parent_hdul = hdu._parent if hasattr(hdu, "_parent") else None
    except Exception:
        parent_hdul = None

    if parent_hdul is None:
        # Can't find parent; try to construct minimal hdulist
        parent_hdul = [hdu]

    if prefer_wmap:
        try:
            # Accept hdulist or list-like
            import astropy.io.fits as _fits
            if isinstance(parent_hdul, _fits.HDUList):
                hdul = parent_hdul
            else:
                hdul = parent_hdul
            wmap_attempt = build_spec_wave_from_wmap(hdul)
            if wmap_attempt is not None:
                # If wmap_attempt is 3D (nspec, ny, nx) or 1D
                if return_full_wave_map and wmap_attempt.ndim == 3:
                    wmap = wmap_attempt
                elif wmap_attempt.ndim == 1:
                    spec_wave = wmap_attempt
                else:
                    # If a 2D-like thing, attempt to reduce
                    if wmap_attempt.ndim == 2:
                        spec_wave = np.nanmedian(wmap_attempt, axis=1)
                    else:
                        spec_wave = np.nanmedian(wmap_attempt, axis=1)
        except Exception:
            spec_wave = None

    # If spec_wave still None, try WCS on this HDU
    if spec_wave is None:
        spec_wave = build_spec_wave_from_wcs(hdu)

    # Now we must ensure cube has spectral axis as axis 0
    # Identify which axis in data is spectral (longest axis heuristics + header hints)
    spec_axis = int(np.argmax(data.shape))
    if spec_axis != 0:
        cube = np.moveaxis(data, spec_axis, 0)
    else:
        cube = data.copy()

    # If we have a wmap full map (nspec, ny, nx), ensure orientation matches cube
    if wmap is not None:
        # try to move wmap spectral axis to 0 if needed
        if wmap.shape[0] != cube.shape[0]:
            # find longest axis in wmap and move to front
            sa = int(np.argmax(wmap.shape))
            wmap = np.moveaxis(wmap, sa, 0)
        # optionally return full map
        if return_full_wave_map:
            spec_wave = wmap  # shape (nspec, ny, nx)
        else:
            # collapse to 1D median
            try:
                spec_wave = np.nanmedian(wmap, axis=(1, 2))
            except Exception:
                spec_wave = np.nanmedian(wmap, axis=1)

    # Final sanity checks
    ok = True
    if spec_wave is None:
        ok = False
    else:
        # if 1D, check spacing not zero
        if hasattr(spec_wave, "ndim") and spec_wave.ndim == 1:
            d = np.nanmedian(np.diff(spec_wave))
            if not np.isfinite(d) or np.isclose(d, 0.0):
                ok = False
        # if 3D, ensure spectral dim length matches cube
        if hasattr(spec_wave, "ndim") and spec_wave.ndim == 3:
            if spec_wave.shape[0] != cube.shape[0]:
                ok = False

    # Convert to microns if appears to be Angstroms (>100 typical)
    if spec_wave is not None and hasattr(spec_wave, "ndim") and spec_wave.ndim >= 1:
        med = np.nanmedian(spec_wave)
        if med is not None and med > 100:
            spec_wave = spec_wave / 1e4

    return cube, spec_wave, ok

# Simple continuum-subtracted integrated flux estimator (per-spaxel)
def measure_line_window(spec_wave, spec_flux, center_um, half_width_um, cont_width_um=None):
    """
    Accepts spec_wave either as 1D (nspec,) or 3D (nspec, ny, nx).
    For the 3D case, spec_flux must be 1D (nspec,) for the same spaxel.
    Returns (flux, ferr) or (np.nan, np.nan) on failure.
    """
    if spec_wave is None or spec_flux is None:
        return np.nan, np.nan

    # If spec_wave is 3D, caller should have passed the per-spaxel 1D slice.
    if getattr(spec_wave, "ndim", None) == 3:
        # invalid usage
        raise ValueError("spec_wave is 3D; pass the 1D wavelength array for the spaxel instead")

    # Determine in_line and continuum sidebands
    # if user did not pass cont_width_um, set a default relative to sampling
    dl = np.nanmedian(np.diff(spec_wave))
    if not np.isfinite(dl) or dl == 0:
        return np.nan, np.nan
    if cont_width_um is None:
        cont_width_um = max(8 * dl, 0.05)

    in_line = (spec_wave >= (center_um - half_width_um)) & (spec_wave <= (center_um + half_width_um))
    left_cont = (spec_wave >= (center_um - half_width_um - cont_width_um)) & (spec_wave < (center_um - half_width_um))
    right_cont = (spec_wave > (center_um + half_width_um)) & (spec_wave <= (center_um + half_width_um + cont_width_um))
    cont_idx = np.where(left_cont | right_cont)[0]

    if cont_idx.size < 3 or in_line.sum() < 1:
        return np.nan, np.nan

    cont_med = np.nanmedian(spec_flux[cont_idx])
    flux = np.nansum((spec_flux[in_line] - cont_med) * dl)
    sigma = np.nanstd(spec_flux[cont_idx])
    ferr = sigma * np.sqrt(in_line.sum()) * dl
    return flux, ferr

