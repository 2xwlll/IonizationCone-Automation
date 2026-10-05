#!/usr/bin/env python3
"""Render a wavelength-resolved synthetic AGN as a 2D HST/WFPC2 F502N image.

The saved training image is a single ``(y, x)`` array.  Spectral samples are
used internally only to integrate line and nebular-continuum flux through the
measured filter throughput.  A stellar/nuclear continuum, foreground dust,
PSF, and observation-matched noise are then added.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, zoom


DEFAULT_O3 = Path(
    "data/2d/MAST_2026-04-14T0111/HST/"
    "hst_5754_01_wfpc2_pc_f502n_u2m301/"
    "hst_5754_01_wfpc2_pc_f502n_u2m301_drz.fits"
)
DEFAULT_CONT = Path(
    "data/2d/MAST_continuum/mastDownload/HST/"
    "hst_5754_01_wfpc2_pc_f547m_u2m301/"
    "hst_5754_01_wfpc2_pc_f547m_u2m301_drz.fits"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("cube", type=Path)
    p.add_argument("--throughput", type=Path, default=Path(
        "configs/instruments/hst_wfpc2_pc_f502n.csv"))
    p.add_argument("--output", type=Path, default=Path(
        "data/cloudy/ngc1068_oiii_full_v1/ngc1068_f502n_pair.npz"))
    p.add_argument("--preview", type=Path, default=Path(
        "results/cloudy/ngc1068_f502n_pair_preview.png"))
    p.add_argument("--reference-f502n", type=Path, default=DEFAULT_O3)
    p.add_argument("--reference-f547m", type=Path, default=DEFAULT_CONT)
    p.add_argument("--seed", type=int, default=1068)
    p.add_argument("--target-psf-fwhm", type=float, default=0.10,
                   help="Final PSF FWHM in arcsec")
    p.add_argument("--input-psf-fwhm", type=float, default=0.075,
                   help="PSF already present in the input cube")
    p.add_argument("--pixel-scale", type=float, default=0.045528,
                   help="Output arcsec per pixel")
    p.add_argument("--input-pixel-scale", type=float, default=0.04,
                   help="Input cube arcsec per pixel")
    return p.parse_args()


def load_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    curve = np.loadtxt(path, delimiter=",", comments="#")
    return curve[:, 0], curve[:, 1]


def centered_reference(path: Path, size: int) -> np.ndarray:
    from astropy.io import fits
    from astropy.coordinates import SkyCoord
    from astropy.wcs import WCS
    import astropy.units as u

    with fits.open(path) as hdul:
        image = np.nan_to_num(hdul[1].data.astype(np.float64))
        # Fix the crop to the NGC 1068 nucleus. Bright-field source selection
        # can otherwise choose a star or a drizzle-chip boundary.
        nucleus = SkyCoord(ra=40.669621*u.deg, dec=-0.013294*u.deg)
        cx, cy = WCS(hdul[1].header).world_to_pixel(nucleus)
    cy, cx = int(round(float(cy))), int(round(float(cx)))
    half = size//2
    crop = image[cy-half:cy-half+size, cx-half:cx-half+size]
    if crop.shape != (size, size):
        raise ValueError(f"Could not take {size}x{size} nucleus crop from {path}")
    return crop


def match_pixel_scale(image: np.ndarray, factor: float, order: int) -> np.ndarray:
    """Resample an image and center crop/pad back to its original shape."""
    size = image.shape[0]
    scaled = zoom(image, factor, order=order, prefilter=order > 1)
    out = np.zeros_like(image)
    n = min(size, scaled.shape[0])
    src0 = (scaled.shape[0]-n)//2
    dst0 = (size-n)//2
    out[dst0:dst0+n, dst0:dst0+n] = scaled[src0:src0+n, src0:src0+n]
    return out


def robust_scale(image: np.ndarray) -> tuple[float, float, float]:
    """Return sky level, sky sigma, and bright-source scale."""
    h, w = image.shape
    yy, xx = np.indices(image.shape)
    r = np.hypot(xx-(w-1)/2, yy-(h-1)/2)
    sky = image[(r > 0.40*w) & (r < 0.49*w)]
    sky_level = float(np.median(sky))
    mad = float(np.median(np.abs(sky-sky_level)))
    sigma = max(1.4826*mad, 1e-8)
    positive = image[image > sky_level + 3*sigma]
    bright = float(np.percentile(positive, 99.5)) if positive.size else 1.0
    return sky_level, sigma, bright


def synthetic_stellar_continuum(
    size: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.indices((size, size), dtype=np.float64)
    x = (xx-(size-1)/2)/(size/2)
    y = (yy-(size-1)/2)/(size/2)
    pa = rng.uniform(0, np.pi)
    xp = x*np.cos(pa)+y*np.sin(pa)
    yp = -x*np.sin(pa)+y*np.cos(pa)
    q = rng.uniform(0.55, 0.82)
    r = np.sqrt(xp*xp+(yp/q)**2)
    bulge = np.exp(-3.2*np.power(np.maximum(r, 1e-4), 0.45))
    disk = np.exp(-2.7*r)
    nucleus = np.exp(-0.5*(r/0.018)**2)
    mottling = gaussian_filter(rng.normal(size=(size, size)), size/45)
    mottling /= max(mottling.std(), 1e-8)
    continuum = (0.62*bulge + 0.30*disk*(1+0.10*mottling)
                 + 1.25*nucleus)

    # Smooth foreground lanes attenuate both stellar and nuclear continuum.
    lanes = gaussian_filter(rng.normal(size=(size, size)), size/18)
    lanes = (lanes-lanes.min())/max(np.ptp(lanes), 1e-8)
    tau = rng.uniform(0.25, 0.85)*lanes
    attenuation = np.exp(-tau)
    return np.clip(continuum*attenuation, 0, None), attenuation


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    with np.load(args.cube) as data:
        cube = data["input"].astype(np.float64)
        mask = data["mask"].astype(np.float32)
        wavelength = data["wavelength_angstrom"].astype(np.float64)
    if cube.ndim != 3 or cube.shape[0] != wavelength.size:
        raise ValueError("Expected cube shape (wavelength, y, x)")

    filter_wave, filter_throughput = load_curve(args.throughput)
    throughput = np.interp(wavelength, filter_wave, filter_throughput,
                           left=0.0, right=0.0)
    dlambda = np.gradient(wavelength)
    # For a photon-counting detector, counts are proportional to F_lambda*T*lambda.
    weights = throughput*wavelength*dlambda
    gas = np.tensordot(weights, cube, axes=(0, 0))
    spatial_factor = args.input_pixel_scale/args.pixel_scale
    gas = match_pixel_scale(gas, spatial_factor, order=1)
    mask = match_pixel_scale(mask, spatial_factor, order=0)
    # Preserve an explicit nucleus bridge under pixel-scale conversion. This
    # makes a bicone one component even with strict 4-neighbour connectivity.
    center = mask.shape[0]//2
    mask[center-2:center+3, center-2:center+3] = 1.0
    gas /= max(float(np.percentile(gas, 99.8)), 1e-12)

    size = gas.shape[0]
    continuum, continuum_attenuation = synthetic_stellar_continuum(size, rng)
    continuum /= max(float(np.percentile(continuum, 99.8)), 1e-12)

    real_o3 = centered_reference(args.reference_f502n, size)
    real_cont = centered_reference(args.reference_f547m, size)
    ref_sky, ref_noise, ref_bright = robust_scale(real_o3)
    _, _, cont_bright = robust_scale(real_cont)
    # F547M-to-F502N count scaling used by the existing NGC 1068 reduction.
    photflam_ratio = 7.595041e-18/2.943716e-16
    continuum_peak = min(cont_bright*photflam_ratio, 0.75*ref_bright)
    gas_peak = max(ref_bright-continuum_peak, 4*ref_noise)
    clean = ref_sky + gas_peak*gas + continuum_peak*continuum

    extra_fwhm = np.sqrt(max(args.target_psf_fwhm**2-
                             args.input_psf_fwhm**2, 0.0))
    extra_sigma_px = extra_fwhm/args.pixel_scale/2.35482
    clean = gaussian_filter(clean, extra_sigma_px)

    # Drizzled HST reference pixels contain correlated, approximately Gaussian
    # background noise.  Match its robust amplitude and correlation scale.
    noise = gaussian_filter(rng.normal(size=clean.shape), 0.55)
    noise *= ref_noise/max(float(noise.std()), 1e-12)
    observed = (clean+noise).astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, input=observed, mask=mask, clean=clean.astype(np.float32),
        filter_integrated_gas=gas.astype(np.float32),
        stellar_continuum=(continuum_peak*continuum).astype(np.float32),
        continuum_attenuation=continuum_attenuation.astype(np.float32),
        wavelength_angstrom=wavelength.astype(np.float32),
        filter_throughput=throughput.astype(np.float32),
    )
    pair_root = args.output.parent/"f502n_training_pair"
    (pair_root/"images").mkdir(parents=True, exist_ok=True)
    (pair_root/"masks").mkdir(parents=True, exist_ok=True)
    np.save(pair_root/"images"/f"{args.output.stem}.npy", observed)
    np.save(pair_root/"masks"/f"{args.output.stem}.npy", mask)

    metadata = {
        "shape": list(observed.shape), "axis_order": ["y", "x"],
        "filter": "HST/WFPC2-PC.F502N", "pixel_scale_arcsec": args.pixel_scale,
        "input_pixel_scale_arcsec": args.input_pixel_scale,
        "target_psf_fwhm_arcsec": args.target_psf_fwhm,
        "reference_sky_counts": ref_sky, "reference_noise_counts": ref_noise,
        "reference_bright_counts": ref_bright,
        "continuum_peak_counts": continuum_peak, "gas_peak_counts": gas_peak,
        "seed": args.seed,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def shown(a: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(a, [5, 99.7])
        return np.arcsinh(np.clip((a-lo)/(hi-lo+1e-12), 0, None)*8)
    panels = [real_o3, gas, continuum_peak*continuum, clean, observed, mask]
    titles = ["Real NGC 1068 F502N", "Cloudy through F502N",
              "Stellar + nuclear continuum", "Synthetic clean",
              "Synthetic observed", "Geometric target mask"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for ax, image, title in zip(axes.flat, panels, titles):
        ax.imshow(image if title.endswith("mask") else shown(image), origin="lower",
                  cmap="gray" if title.endswith("mask") else "magma")
        ax.set_title(title); ax.set_axis_off()
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.preview, dpi=170)
    plt.close(fig)
    print(f"Saved 2D pair {observed.shape} to {args.output}")
    print(f"Saved preview to {args.preview}")


if __name__ == "__main__":
    main()
