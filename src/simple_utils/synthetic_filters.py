"""Custom observed-frame passbands for existing narrow F_lambda cubes.

This module changes wavelength weighting only. It does not redshift, spatially
convolve, normalize, subtract continuum, or add components to a source cube.
"""

import numpy as np

from src.simple_utils.synthetic_photometry import integrate_bandpass, validate_wavelengths


def top_hat_passband(lower_angstrom, upper_angstrom, peak_throughput=1.0):
    """Exact rectangular support; integration clips at the two supplied edges."""
    if (not np.isfinite([lower_angstrom, upper_angstrom, peak_throughput]).all()
            or not 0 < lower_angstrom < upper_angstrom
            or not 0 < peak_throughput <= 1):
        raise ValueError("Require positive increasing edges and peak throughput in (0,1]")
    return np.array([lower_angstrom, upper_angstrom], dtype=float), np.full(2, peak_throughput)


def gaussian_passband(center_angstrom, fwhm_angstrom, *, truncate_sigma=4.0,
                      samples=801, peak_throughput=1.0):
    """Sample a Gaussian passband with explicit finite support (default +/-4 sigma).

    It is zero outside that support. This truncates about 0.0063% of the ideal
    Gaussian area at the default extent; coverage refers to this finite filter.
    """
    if (not np.isfinite([center_angstrom, fwhm_angstrom, truncate_sigma,
                         peak_throughput]).all()
            or center_angstrom <= 0 or fwhm_angstrom <= 0
            or truncate_sigma <= 0 or not 0 < peak_throughput <= 1
            or not isinstance(samples, (int, np.integer)) or samples < 3):
        raise ValueError("Require positive center, FWHM, extent, >=3 samples and peak in (0,1]")
    sigma = fwhm_angstrom / np.sqrt(8*np.log(2))
    lower, upper = center_angstrom-truncate_sigma*sigma, center_angstrom+truncate_sigma*sigma
    if lower <= 0:
        raise ValueError("Gaussian support extends to nonpositive wavelengths")
    wave = np.linspace(lower, upper, samples)
    response = peak_throughput*np.exp(-0.5*((wave-center_angstrom)/sigma)**2)
    return wave, response


def filter_cube(cube, wavelength_angstrom, filter_wavelength_angstrom, throughput):
    """Return an image, channel weights and coverage, without changing spatial PSF.

    Both wavelength axes must be in the same observed frame, in Angstrom.
    Input must be linear F_lambda samples, not log/asinh values or integrated
    flux per spectral bin. Reject any passband response outside source coverage.
    """
    wave = validate_wavelengths(wavelength_angstrom)
    fw = validate_wavelengths(filter_wavelength_angstrom)
    if fw[0] < wave[0] or fw[-1] > wave[-1]:
        raise ValueError("Synthetic passband extends outside the cube wavelength range; "
                         "choose a narrower passband")
    # Containment is exact above; permit only floating-point roundoff below.
    return integrate_bandpass(cube, wave, fw, throughput, minimum_coverage=1-1e-12)
