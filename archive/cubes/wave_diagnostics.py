#!/usr/bin/env python3
"""
wave_diagnostics.py
Quick checks for JWST S3D cubes' wavelength solutions (WMAP / WCS / header).
Run from project root: PYTHONPATH=. python3 scripts/cubes/wave_diagnostics.py
"""

import os, glob
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

RAW_DIR = "data/cubes/raw/real"
PATTERN = os.path.join(RAW_DIR, "*.fits*")

def find_3d_hdu(hdul):
    for idx, h in enumerate(hdul):
        if getattr(h, "data", None) is None:
            continue
        if h.data.ndim == 3:
            return idx, h
    return None, None

def find_wmap_hdu(hdul):
    for idx, h in enumerate(hdul):
        name = (getattr(h, "name", "") or "").upper()
        if name in ("WMAP", "WAVELENGTH", "WAVE"):
            return idx, h
    # fallback: any 3D hdu that is not the SCI/ERR/DQ by common name
    for idx, h in enumerate(hdul):
        if getattr(h, "data", None) is None:
            continue
        if h.data.ndim == 3 and (not ((getattr(h, "name","") or "").upper() in ("SCI","ERR","DQ"))):
            return idx, h
    return None, None

def header_wcs_spec_wave(hdu):
    # attempt to build spec axis from WCS or header keywords
    try:
        w = WCS(hdu.header, naxis=3)
        spec_wcs = w.sub(['spectral'])
        nspec = hdu.data.shape[0]
        pix = np.arange(nspec)
        world = spec_wcs.wcs_pix2world(pix, 0)
        # extract numeric array
        try:
            wave = np.asarray(world)[0]
        except Exception:
            wave = np.asarray(world).ravel()
        if np.nanmedian(wave) > 100:
            wave = wave / 1e4
        return wave, "WCS"
    except Exception:
        # header CRVAL/CDELT fallback
        hdr = hdu.header
        crval3 = hdr.get('CRVAL3') or hdr.get('CRVAL1') or hdr.get('CRVAL')
        cdelt3 = hdr.get('CDELT3') or hdr.get('CD3_3') or hdr.get('CDELT1') or hdr.get('CDELT')
        crpix3 = hdr.get('CRPIX3') or hdr.get('CRPIX1') or hdr.get('CRPIX') or 1
        nspec = hdu.data.shape[0]
        if crval3 is None or cdelt3 is None:
            return None, "HEADER_FAIL"
        pix = np.arange(nspec)
        wave = (pix - (crpix3 - 1)) * cdelt3 + crval3
        if np.nanmedian(wave) > 100:
            wave = wave / 1e4
        return wave, "HEADER"

def analyze_wave_array(wave):
    out = {}
    if wave is None:
        out['ok'] = False
        out['reason'] = "no_wave"
        return out
    wave = np.asarray(wave, dtype=float)
    out['nspec'] = wave.size
    out['nan_frac'] = np.isnan(wave).mean()
    out['zero_frac'] = (wave == 0).mean()
    out['min'] = np.nanmin(wave)
    out['max'] = np.nanmax(wave)
    diffs = np.diff(wave)
    out['median_dlambda'] = np.nanmedian(diffs) if diffs.size>0 else np.nan
    out['dl_nonzero'] = np.any(np.abs(diffs) > 1e-8) if diffs.size>0 else False
    out['range'] = out['max'] - out['min']
    out['ok'] = (out['range'] > 1e-6) and np.isfinite(out['median_dlambda']) and (out['median_dlambda'] > 1e-8)
    if not out['ok']:
        if out['range'] <= 1e-6:
            out['reason'] = "constant_axis"
        elif not np.isfinite(out['median_dlambda']) or out['median_dlambda'] <= 1e-8:
            out['reason'] = "tiny_dlambda"
        else:
            out['reason'] = "unknown"
    return out

def inspect_file(path):
    print("\n" + "="*70)
    print("File:", path)
    try:
        hdul = fits.open(path, ignore_missing_end=True)
    except Exception as e:
        print("  ERROR opening file:", e)
        return
    # list HDUs
    for i,h in enumerate(hdul):
        name = getattr(h, "name", "")
        shape = getattr(h, "data", None).shape if getattr(h, "data", None) is not None else None
        print(f"  HDU {i:02d}: name={name!r}, shape={shape}")
    sci_idx, sci_hdu = find_3d_hdu(hdul)
    if sci_hdu is None:
        print("  No 3D SCI HDU found.")
        hdul.close()
        return
    print(f"  Using SCI HDU index {sci_idx}")
    wmap_idx, wmap_hdu = find_wmap_hdu(hdul)
    if wmap_hdu is not None:
        print(f"  Found WMAP-like HDU at index {wmap_idx}, shape={wmap_hdu.data.shape}")
        # try to coerce spectral axis to front
        w = np.asarray(wmap_hdu.data)
        spec_axis = int(np.argmax(w.shape))
        w = np.moveaxis(w, spec_axis, 0)
        print(f"    Moved spectral axis -> shape now {w.shape} (nspec, A, B)")
        # median collapse and analyze
        wave1d = np.nanmedian(w, axis=(1,2)) if w.ndim==3 else np.nanmedian(w, axis=1)
        a = analyze_wave_array(wave1d)
        print(f"    wave1d nspec={a.get('nspec')} min={a.get('min'):.6f}, max={a.get('max'):.6f}, median_dλ={a.get('median_dlambda')}, ok={a.get('ok')}, reason={a.get('reason', '')}")
        # sample few values
        print("    sample wave1d[:8]:", wave1d[:8])
        # check if full map is constant or degenerate
        if w.ndim==3:
            # fraction of spaxels that have identical wave1d
            first = w[0,:,:]
            same_count = np.mean([np.allclose(w[:,iy,ix], wave1d, atol=1e-9) for iy in range(w.shape[1]) for ix in range(w.shape[2])])
            print(f"    fraction of spaxels matching median spectrum exactly (rare): {same_count:.3f}")
    else:
        print("  No WMAP extension found; attempting WCS/header extraction from SCI HDU")
        wave1d, method = header_wcs_spec_wave(sci_hdu)
        if wave1d is None:
            print("    Header/WCS fallback failed (no CRVAL/CDELT).")
        else:
            a = analyze_wave_array(wave1d)
            print(f"    method={method} nspec={a.get('nspec')} min={a.get('min'):.6f}, max={a.get('max'):.6f}, median_dλ={a.get('median_dlambda')}, ok={a.get('ok')}, reason={a.get('reason','')}")
            print("    sample wave1d[:8]:", wave1d[:8])

    # print key header keywords that help
    hdr = sci_hdu.header
    keys = ['CRVAL3','CDELT3','CRPIX3','CRVAL1','CDELT1','CRPIX1','CTYPE3','CTYPE1']
    print("  Useful header keywords:")
    for k in keys:
        print(f"    {k}: {hdr.get(k)}")
    # ASDF presence
    asdf_found = any(((getattr(h, "name", "") or "").upper() == "ASDF" or ("ASDF" in str(type(h.data)))) for h in hdul)
    print("  ASDF ext present?:", asdf_found)

    hdul.close()

def main():
    files = sorted(glob.glob(PATTERN))
    if not files:
        print("No files found:", PATTERN)
        return
    for f in files:
        inspect_file(f)

if __name__ == "__main__":
    main()

