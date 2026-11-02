"""NIRCam forward response on an intrinsic, common spatial model grid.

Output is relative unless input F_lambda is physically calibrated. No distance,
collecting area, exposure, detector rebinning, or noise is applied here.
"""

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

from src.simple_utils.synthetic_photometry import (
    bandpass_coverage, photon_bandpass_weights, validate_wavelengths,
)

DEFAULT_CATALOG = Path(__file__).resolve().parents[2] / "configs/instruments/jwst_nircam/catalog.json"


def load_filters(catalog_path=DEFAULT_CATALOG):
    path = Path(catalog_path)
    catalog = json.loads(path.read_text())
    return {entry["name"]: {
        **entry, "curve": np.loadtxt(path.parent / entry["file"], delimiter=",")
    } for entry in catalog["filters"]}


def redshift_flambda(cube, rest_wavelength_angstrom, redshift):
    """Stretch wavelength and F_lambda density, preserving integrated energy.

    Input physical normalization should already include 1/(4*pi*D_L**2) if
    starting from luminosity. Angular geometry must already be set separately.
    This function applies only wavelength stretch and its density Jacobian;
    it does not calculate luminosity distance or surface-brightness dimming.
    """
    wave = validate_wavelengths(rest_wavelength_angstrom)
    array = np.asarray(cube)
    if array.ndim != 3 or array.shape[0] != len(wave) or min(array.shape[1:]) < 1:
        raise ValueError("Expected nonempty (wavelength,y,x) F_lambda cube")
    if not np.isfinite(array).all():
        raise ValueError("Cube contains nonfinite flux")
    if not np.isfinite(redshift) or redshift < 0:
        raise ValueError("Galaxy redshift must be finite and nonnegative")
    stretch = 1 + redshift
    return array / stretch, wave * stretch


def gaussian_sigma_pixels(wavelength_angstrom, pixel_scale_arcsec):
    """Ideal 6.5-m diffraction-core FWHM=1.028*lambda/D, approximated as Gaussian.

    Not an empirical NIRCam PSF: excludes segmentation, wings, optical error,
    detector response, and pupil-stop changes. Use kernel_provider for these.
    """
    if not np.isfinite(pixel_scale_arcsec) or pixel_scale_arcsec <= 0:
        raise ValueError("Pixel scale must be finite and positive")
    return 1.028 * np.asarray(wavelength_angstrom) * 1e-10 / 6.5 * 206264.806247 / pixel_scale_arcsec / 2.354820045


def convolve_kernel(image, kernel):
    """Centered unit-sum PSF; zero exterior permits light to leave the cutout."""
    kernel = np.asarray(kernel, dtype=float)
    if (kernel.ndim != 2 or any(n % 2 != 1 for n in kernel.shape)
            or not np.isfinite(kernel).all() or np.any(kernel < 0)
            or kernel.sum() <= 0):
        raise ValueError("PSF kernel must be finite, nonnegative, nonzero and odd-sized 2D")
    return fftconvolve(image, kernel / kernel.sum(), mode="same")


def render_filters(cube, rest_wavelength_angstrom, *, redshift,
                   pixel_scale_arcsec, filters=None, catalog_path=DEFAULT_CATALOG,
                   minimum_coverage=0.999, kernel_provider=None):
    """Render selected filters (default all 29) with chromatic spatial PSFs.

    kernel_provider(name, observed_wavelength_angstrom, pixel_scale_arcsec)
    may supply monochromatic STPSF/empirical kernels, centered on the same
    grid and sky orientation as the model. Otherwise use an approximate
    Gaussian diffraction core per wavelength. Throughput is applied once.

    Source slices are piecewise-linear F_lambda samples; resolve line profiles
    and check spectral convergence. Chromatic convolution is sampled at those
    nodes. Every filter is checked for coverage before any images are rendered.
    No per-filter normalization: relative responses and ratios are retained.
    """
    observed_cube, observed_wave = redshift_flambda(cube, rest_wavelength_angstrom, redshift)
    sigmas = gaussian_sigma_pixels(observed_wave, pixel_scale_arcsec)
    if not np.isfinite(minimum_coverage) or not 0 < minimum_coverage <= 1:
        raise ValueError("Minimum coverage must lie in (0,1]")
    available = load_filters(catalog_path)
    selected = list(available) if filters is None else list(filters)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Select at least one filter, without duplicates")
    prepared = {}
    for name in selected:
        if name not in available:
            raise ValueError(f"Unknown NIRCam filter: {name}")
        fw, throughput = available[name]["curve"].T
        coverage = bandpass_coverage(observed_wave, fw, throughput)
        if coverage < minimum_coverage:
            raise ValueError(f"{name}: source covers {100*coverage:.4f}% of photon response; "
                             "extend the rest-frame spectrum for this redshift")
        prepared[name] = (photon_bandpass_weights(observed_wave, fw, throughput), coverage)

    images, metadata = {}, {}
    # Gaussian kernels do not depend on filter. Reuse each smoothed slice.
    if kernel_provider is None:
        images = {name: np.zeros(observed_cube.shape[1:], dtype=float) for name in selected}
        for channel, sigma in enumerate(sigmas):
            if not any(prepared[name][0][channel] > 0 for name in selected):
                continue
            blurred = gaussian_filter(observed_cube[channel], sigma, mode="constant", cval=0)
            for name in selected:
                images[name] += prepared[name][0][channel] * blurred
    else:
        for name, (weights, _) in prepared.items():
            image = np.zeros(observed_cube.shape[1:], dtype=float)
            for channel in np.flatnonzero(weights > 0):
                kernel = kernel_provider(name, observed_wave[channel], pixel_scale_arcsec)
                image += weights[channel] * convolve_kernel(observed_cube[channel], kernel)
            images[name] = image
    for name in selected:
        entry = available[name]
        metadata[name] = {key: value for key, value in entry.items() if key != "curve"}
        metadata[name].update({
            "redshift": float(redshift), "coverage": prepared[name][1],
            "model_pixel_scale_arcsec": float(pixel_scale_arcsec),
            "psf_model": "supplied monochromatic kernels" if kernel_provider else "approximate chromatic Gaussian diffraction core",
            "response_units": "relative; electrons/s/cm^2 if input is calibrated erg/s/cm^2/Angstrom per model pixel",
            "spatial_sampling": "common model grid; not detector-rebinned",
        })
    return images, metadata
