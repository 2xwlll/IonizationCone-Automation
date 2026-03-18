#!/usr/bin/env python3
"""
compute_line_ratios_multi.py

Multi-cube line extraction + ratio pipeline for JWST MIRI IFU cubes.

Usage:
    PYTHONPATH=. python3 scripts/cubes/compute_line_ratios_multi.py
"""

import os
import glob
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.optimize import curve_fit
from scipy.ndimage import binary_opening, binary_closing, label, zoom
import matplotlib.pyplot as plt

# -------------------- User / run-time defaults --------------------
RAW_DIR = "data/cubes/raw/real"
OUT_DIR = "data/cubes/processed/real"
MIN_HALF_WIDTH_UM = 0.02    # minimum half-width for fitting window (µm)
MIN_SNR = 3.0               # S/N threshold for "good" pixels
PERCENTILE_MASK = 90        # fallback quantile mask
REDSHIFT = 0.0

# Lines from your Table 1 (rest wavelengths in microns)
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

DEFAULT_RATIOS = {
    "NeV14p32__NeII12p81": ("NeV_14.32", "NeII_12.81"),
    "OIV25p89__NeII12p81": ("OIV_25.89", "NeII_12.81"),
    "NeIII15p56__NeII12p81": ("NeIII_15.56", "NeII_12.81")
}

# -------------------- utilities --------------------

def find_science_hdu(hdul):
    for h in hdul:
        if getattr(h, "data", None) is not None and h.data.ndim == 3:
            return h
    return None

def find_wmap_hdu(hdul):
    for h in hdul:
        name = (getattr(h, "name", "") or "").upper()
        if name in ("WMAP", "WAVELENGTH", "WAVE") and getattr(h, "data", None) is not None and h.data.ndim == 3:
            return h
    return None

def find_err_hdu(hdul):
    for h in hdul:
        name = (getattr(h, "name", "") or "").upper()
        if name in ("ERR", "ERROR", "UNC") and getattr(h, "data", None) is not None and h.data.ndim == 3:
            return h
    return None

# try jwst datamodel (optional)
try:
    from jwst.datamodels import CubeModel
    JWST_AVAILABLE = True
except Exception:
    JWST_AVAILABLE = False

def build_spectral_axis(flux_hdu, wave_hdu=None, cube_file=None):
    """Return 1D spectral axis in µm for a cube HDU."""
    hdr = flux_hdu.header
    nspec_guess = flux_hdu.data.shape[0]

    # 0) JWST CubeModel (MIRI) — returns µm if available
    if JWST_AVAILABLE and cube_file is not None:
        try:
            cm = CubeModel(cube_file)
            wave = cm.wavelength
            cm.close()
            if np.all(np.isfinite(wave)):
                return np.asarray(wave).astype(float)
        except Exception:
            pass

    # 1) WCS
    try:
        w = WCS(hdr, naxis=3)
        spec_wcs = w.sub(['spectral'])
        pix = np.arange(nspec_guess)
        wave = spec_wcs.wcs_pix2world(pix, 0)[0]
        # convert Angstrom -> µm if necessary
        if np.nanmedian(wave) > 100:
            wave = wave / 1e4
        if np.all(np.isfinite(wave)) and wave.size == nspec_guess:
            return np.asarray(wave).astype(float)
    except Exception:
        pass

    # 2) header CRVAL3 / CDELT3 fallback
    crval3 = hdr.get("CRVAL3") or hdr.get("CRVAL1")
    cdelt3 = hdr.get("CDELT3") or hdr.get("CDELT1")
    crpix3 = hdr.get("CRPIX3") or hdr.get("CRPIX1") or 1.0
    if (crval3 is not None) and (cdelt3 is not None):
        pix = np.arange(nspec_guess)
        wave = (pix - (crpix3 - 1)) * cdelt3 + crval3
        if np.nanmedian(wave) > 100:
            wave = wave / 1e4
        return np.asarray(wave).astype(float)

    # 3) WMAP fallback (3D wavelength map)
    if wave_hdu is not None:
        wmap = np.asarray(wave_hdu.data)
        spec_axis = int(np.argmax(wmap.shape))
        wmap = np.moveaxis(wmap, spec_axis, 0)
        try:
            wave1d = np.nanmedian(wmap, axis=(1,2))
        except Exception:
            wave1d = np.nanmedian(wmap, axis=1)
        if np.nanmax(wave1d) - np.nanmin(wave1d) > 1e-6:
            return wave1d.astype(float)

    return None

def reorder_cube_to_spec_first(cube, spec_wave):
    shape = tuple(cube.shape)
    if shape[0] == spec_wave.size:
        return cube
    if shape[-1] == spec_wave.size:
        return np.moveaxis(cube, -1, 0)
    for ax, s in enumerate(shape):
        if s == spec_wave.size:
            return np.moveaxis(cube, ax, 0)
    return cube

# -------------------- fitting helpers (robust) --------------------

def gaussian(x, amp, cen, sigma):
    return amp * np.exp(-0.5 * ((x - cen) / sigma) ** 2)

def linear_cont(x, a, b):
    return a + b * (x - np.nanmean(x))

def gauss_plus_linear(x, amp, cen, sigma, a, b):
    return gaussian(x, amp, cen, sigma) + linear_cont(x, a, b)

def fit_line_gaussian(spec_wave, spec_flux, center_um,
                      min_nchan_fit=6,
                      half_width_um=None,
                      min_half_width_um=MIN_HALF_WIDTH_UM,
                      cont_width_um=0.05,
                      max_sigma_um=None):
    """
    Fit Gaussian + linear continuum to one 1D spectrum (returns flux in same units).
    Falls back to local integration if fit doesn't converge or insufficient channels.
    """
    if spec_wave is None or spec_flux is None:
        return np.nan, np.nan, False, {"reason": "no_input"}

    # sampling
    dl = np.nanmedian(np.diff(spec_wave))
    if not np.isfinite(dl) or dl <= 0:
        return np.nan, np.nan, False, {"reason": "bad_sampling"}

    half = max(half_width_um or (3 * dl), min_half_width_um)

    low_all = center_um - half - cont_width_um
    high_all = center_um + half + cont_width_um
    sel_all = np.isfinite(spec_flux) & (spec_wave >= low_all) & (spec_wave <= high_all)
    if np.sum(sel_all) < min_nchan_fit:
        return _fallback_integration(spec_wave, spec_flux, center_um, half, cont_width_um)

    x = spec_wave[sel_all]
    y = spec_flux[sel_all]

    # continuum windows
    left_idx = np.isfinite(spec_flux) & (spec_wave >= (center_um - half - cont_width_um)) & (spec_wave < (center_um - half))
    right_idx = np.isfinite(spec_flux) & (spec_wave > (center_um + half)) & (spec_wave <= (center_um + half + cont_width_um))
    cont_idx = left_idx | right_idx

    cont_med = np.nanmedian(spec_flux[cont_idx]) if np.sum(cont_idx) >= 1 else np.nanmedian(y)

    amp0 = np.nanmax(y) - cont_med
    if amp0 <= 0:
        return _fallback_integration(spec_wave, spec_flux, center_um, half, cont_width_um)

    cen0 = center_um
    sigma0 = max(3 * dl, 0.5 * min_half_width_um)
    p0 = [amp0, cen0, sigma0, cont_med, 0.0]

    lower = [0.0, center_um - half, dl * 0.5, -np.inf, -np.inf]
    upper = [np.inf, center_um + half, max_sigma_um if max_sigma_um is not None else half * 3, np.inf, np.inf]

    try:
        popt, pcov = curve_fit(gauss_plus_linear, x, y, p0=p0, bounds=(lower, upper), maxfev=5000)
        amp, cen, sigma, a, b = popt
        if amp <= 0 or sigma <= 0:
            raise RuntimeError("unphysical params")
        flux = amp * sigma * np.sqrt(2.0 * np.pi)
        # error from covariance (amp, sigma)
        if pcov is not None and np.all(np.isfinite(pcov)):
            dF_dA = sigma * np.sqrt(2.0 * np.pi)
            dF_dSigma = amp * np.sqrt(2.0 * np.pi)
            varF = (dF_dA ** 2) * pcov[0, 0] + (dF_dSigma ** 2) * pcov[2, 2] + 2 * dF_dA * dF_dSigma * pcov[0, 2]
            ferr = np.sqrt(varF) if varF > 0 else np.nan
        else:
            ferr = np.nan
        return flux, ferr, True, {"amp": amp, "cen": cen, "sigma": sigma}
    except Exception as e:
        return _fallback_integration(spec_wave, spec_flux, center_um, half, cont_width_um)

def _fallback_integration(spec_wave, spec_flux, center_um, half, cont_width):
    dl = np.nanmedian(np.diff(spec_wave))
    center_mask = np.isfinite(spec_flux) & (spec_wave >= (center_um - half)) & (spec_wave <= (center_um + half))
    if np.sum(center_mask) == 0:
        return np.nan, np.nan, False, {"reason": "integration_no_data"}
    left = np.isfinite(spec_flux) & (spec_wave >= (center_um - half - cont_width)) & (spec_wave < (center_um - half))
    right = np.isfinite(spec_flux) & (spec_wave > (center_um + half)) & (spec_wave <= (center_um + half + cont_width))
    cont_idx = left | right
    if np.sum(cont_idx) >= 1:
        cont_med = np.nanmedian(spec_flux[cont_idx])
        sigma_cont = np.nanstd(spec_flux[cont_idx])
    else:
        cont_med = np.nanmedian(spec_flux[center_mask])
        sigma_cont = np.nanstd(spec_flux[center_mask])
    flux = np.nansum((spec_flux[center_mask] - cont_med) * dl)
    ferr = sigma_cont * np.sqrt(np.sum(center_mask)) * dl if np.isfinite(sigma_cont) else np.nan
    return flux, ferr, False, {"reason": "fallback"}

# -------------------- mask cleaning --------------------
def clean_mask(mask, opening=2, closing=2, min_size=6):
    m = binary_opening(mask, structure=np.ones((opening, opening)))
    m = binary_closing(m, structure=np.ones((closing, closing)))
    lab, n = label(m)
    out = np.zeros_like(m, dtype=bool)
    for lbl in range(1, n + 1):
        comp = lab == lbl
        if comp.sum() >= min_size:
            out[comp] = True
    return out

# -------------------- high-level --------------------

class CubeInfo:
    """Holds minimal metadata and lazy data for a cube file."""
    def __init__(self, path):
        self.path = path
        self.hdul = None
        self.spec_wave = None
        self.data_shape = None
        self.wmin = None
        self.wmax = None
        self.sci_hdu_idx = None
        self.err_hdu_idx = None

    def inspect(self):
        hdul = fits.open(self.path, ignore_missing_end=True)
        self.hdul = hdul
        sci = find_science_hdu(hdul)
        wmap = find_wmap_hdu(hdul)
        err = find_err_hdu(hdul)
        # keep indices for later reopen safe usage
        for i, h in enumerate(hdul):
            if h is sci:
                self.sci_hdu_idx = i
            if h is err:
                self.err_hdu_idx = i
        self.spec_wave = build_spectral_axis(sci, wmap, cube_file=self.path)
        if self.spec_wave is None:
            self.wmin = None
            self.wmax = None
        else:
            self.wmin = np.nanmin(self.spec_wave)
            self.wmax = np.nanmax(self.spec_wave)
            self.data_shape = sci.data.shape
        hdul.close()
        # clear hdul for lazy load
        self.hdul = None

    def load_cube(self):
        hdul = fits.open(self.path, ignore_missing_end=True)
        sci = hdul[self.sci_hdu_idx] if self.sci_hdu_idx is not None else find_science_hdu(hdul)
        err = hdul[self.err_hdu_idx] if self.err_hdu_idx is not None else find_err_hdu(hdul)
        cube = sci.data.astype(float)
        cube = reorder_cube_to_spec_first(cube, self.spec_wave)
        errmap = err.data.astype(float) if err is not None else None
        if errmap is not None:
            errmap = reorder_cube_to_spec_first(errmap, self.spec_wave)
        hdul.close()
        return cube, errmap

# -------------------- main pipeline functions --------------------

def build_cube_catalog(raw_dir):
    files = sorted(glob.glob(os.path.join(raw_dir, "*.fits")) + glob.glob(os.path.join(raw_dir, "*.fits.gz")))
    catalog = []
    for f in files:
        ci = CubeInfo(f)
        try:
            ci.inspect()
            if ci.spec_wave is None:
                print(f"[WARN] Could not build spectral axis for {os.path.basename(f)} — skipping.")
            else:
                print(f"[CAT] {os.path.basename(f)}  {ci.wmin:.4f} → {ci.wmax:.4f} µm  shape={ci.data_shape}")
                catalog.append(ci)
        except Exception as e:
            print(f"[ERR] inspecting {f}: {e}")
    return catalog

def choose_cube_for_line(catalog, line_um):
    """Choose the best cube (from catalog) that contains the observed line wavelength."""
    obs = line_um * (1.0 + REDSHIFT)
    candidates = [c for c in catalog if (c.wmin is not None and c.wmax is not None and (obs >= c.wmin - 1e-9) and (obs <= c.wmax + 1e-9))]
    if candidates:
        # pick cube with smallest spectral range (best local sampling) or simplest: first
        candidates.sort(key=lambda c: (c.wmax - c.wmin))
        return candidates[0]
    # fallback: pick cube with closest center
    if len(catalog) == 0:
        return None
    diffs = [min(abs(obs - c.wmin), abs(obs - c.wmax)) if (c.wmin is not None) else np.inf for c in catalog]
    return catalog[int(np.argmin(diffs))]

def extract_line_map_from_cube(ci: CubeInfo, line_um, half_width_um=None):
    """Return flux_map, err_map, ok_mask for given line using the provided cube info."""
    cube, errmap = ci.load_cube()
    spec_wave = ci.spec_wave * (1.0 + REDSHIFT)
    nspec, ny, nx = cube.shape
    flux_map = np.full((ny, nx), np.nan, dtype=np.float32)
    err_map = np.full((ny, nx), np.nan, dtype=np.float32)
    ok_mask = np.zeros((ny, nx), dtype=bool)

    # per-spaxel fit
    for j in range(ny):
        for i in range(nx):
            spec = cube[:, j, i]
            if not np.isfinite(spec).any():
                continue
            f, fe, ok, _ = fit_line_gaussian(spec_wave, spec, line_um,
                                             half_width_um=half_width_um,
                                             min_half_width_um=MIN_HALF_WIDTH_UM,
                                             cont_width_um=max(0.05, 8*np.median(np.diff(spec_wave))))
            flux_map[j, i] = f
            # prefer errmap if present (propagated from pipeline), else use fit error
            if errmap is not None:
                # integrate error within the line window in same way as fallback integration
                # approximate by sqrt(sum(err^2 * dl^2)) across channels used in extraction
                # build mask of channels inside center +/- half_width
                dl = np.nanmedian(np.diff(spec_wave))
                half = max(half_width_um or (3*dl), MIN_HALF_WIDTH_UM)
                center_mask = (spec_wave >= (line_um - half)) & (spec_wave <= (line_um + half))
                if center_mask.sum() > 0:
                    err_ch = errmap[:, j, i]
                    ferr = np.sqrt(np.nansum((err_ch[center_mask] * dl) ** 2))
                else:
                    ferr = np.nan
                err_map[j, i] = ferr
            else:
                err_map[j, i] = fe
            ok_mask[j, i] = bool(ok)
    return flux_map, err_map, ok_mask

def resample_to_target(src, target_shape):
    """Resample 2D array src to target_shape using scipy.ndimage.zoom (nearest-ish)."""
    if src.shape == tuple(target_shape):
        return src
    # compute zoom factors (target / src)
    factors = (target_shape[0] / src.shape[0], target_shape[1] / src.shape[1])
    # use order=0 (nearest) to be safe for masks; for flux maps order=1 is acceptable
    return zoom(src, factors, order=1)

def save_map(arr, outpath_png, outpath_fits, title=None, cmap='viridis', vmin=None, vmax=None):
    fits.PrimaryHDU(np.asarray(arr, dtype=np.float32)).writeto(outpath_fits, overwrite=True)
    plt.figure(figsize=(5,4))
    plt.imshow(np.nan_to_num(arr), origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(fraction=0.046)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(outpath_png, dpi=150)
    plt.close()

# -------------------- orchestrator --------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    catalog = build_cube_catalog(RAW_DIR)
    if not catalog:
        print(f"[ERR] no usable cubes found in {RAW_DIR}")
        return

    # Extract maps for all lines (choose cube automatically)
    line_maps = {}
    for lname, wl in LINES.items():
        ci = choose_cube_for_line(catalog, wl)
        if ci is None:
            print(f"[SKIP] no cube candidate for {lname} ({wl:.2f} µm)")
            continue
        print(f"\n[LINE] {lname} ({wl:.3f} µm) -> {os.path.basename(ci.path)}")
        fmap, ferr, ok = extract_line_map_from_cube(ci, wl, half_width_um=None)
        # store with info
        line_maps[lname] = {"flux": fmap, "err": ferr, "ok": ok, "cube": ci}
        # save per-line previews
        base = os.path.splitext(os.path.basename(ci.path))[0]
        outdir = os.path.join(OUT_DIR, base)
        os.makedirs(outdir, exist_ok=True)
        save_map(fmap, os.path.join(outdir, f"{lname}_flux.png"), os.path.join(outdir, f"{lname}_flux.fits"), title=f"{base} {lname} flux")

    # Now compute ratios using DEFAULT_RATIOS
    for rname, (num_k, den_k) in DEFAULT_RATIOS.items():
        print(f"\n[RATIO] {rname}: {num_k} / {den_k}")
        if (num_k not in line_maps) or (den_k not in line_maps):
            print("  [INFO] One or both lines missing from extracted maps; skipping.")
            continue

        num = line_maps[num_k]["flux"]
        den = line_maps[den_k]["flux"]
        ne = line_maps[num_k]["err"]
        de = line_maps[den_k]["err"]
        cube_num = line_maps[num_k]["cube"]
        cube_den = line_maps[den_k]["cube"]

        # If different shapes, resample smaller to larger
        target_shape = num.shape
        if den.shape != target_shape:
            # choose a common target: the larger area in pix
            if den.shape[0]*den.shape[1] > num.shape[0]*num.shape[1]:
                target_shape = den.shape
            # resample both to target
            num = resample_to_target(num, target_shape)
            ne = resample_to_target(ne, target_shape)
            den = resample_to_target(den, target_shape)
            de = resample_to_target(de, target_shape)

        # compute ratio and propagated error
        with np.errstate(invalid='ignore', divide='ignore'):
            ratio = num / den
            denom_safe = den.copy()
            denom_safe[denom_safe == 0] = np.nan
            rerr = np.abs(ratio) * np.sqrt((ne / (num + 1e-30)) ** 2 + (de / (denom_safe + 1e-30)) ** 2)

        # S/N masks
        sn_num = np.abs(num) / (ne + 1e-30)
        sn_den = np.abs(den) / (de + 1e-30)
        sn_good = np.isfinite(ratio) & (sn_num >= MIN_SNR) & (sn_den >= MIN_SNR)

        # quantile fallback mask
        finite_ratio = ratio[np.isfinite(ratio)]
        if finite_ratio.size > 0:
            qthr = np.nanpercentile(finite_ratio, PERCENTILE_MASK)
            q_good = np.isfinite(ratio) & (ratio > qthr)
        else:
            q_good = np.zeros_like(ratio, dtype=bool)

        mask = clean_mask(sn_good | q_good, opening=2, closing=2, min_size=6)

        # save outputs
        outname = os.path.join(OUT_DIR, f"{rname}")
        os.makedirs(outname, exist_ok=True)
        save_map(ratio, os.path.join(outname, f"{rname}.png"), os.path.join(outname, f"{rname}.fits"), title=rname)
        save_map(rerr, os.path.join(outname, f"{rname}_err.png"), os.path.join(outname, f"{rname}_err.fits"), title=rname + " err")
        np.save(os.path.join(outname, f"{rname}_mask.npy"), mask.astype(np.uint8))
        save_map(mask.astype(float), os.path.join(outname, f"{rname}_mask.png"), os.path.join(outname, f"{rname}_mask.fits"), title=rname + " mask", cmap='gray')

        # diagnostics
        finite_frac = np.isfinite(ratio).sum() / ratio.size
        print(f"  Ratio {rname}: finite_frac={finite_frac:.3f}  min={np.nanmin(ratio):.3e} max={np.nanmax(ratio):.3e} mean={np.nanmean(ratio):.3e}")
        print(f"    mask pixels: {mask.sum()}  (S/N passed: {sn_good.sum()}, quantile passed: {q_good.sum()})")

    print("\nAll done.")

if __name__ == "__main__":
    main()
