"""Photon-counting synthetic photometry on observed F_lambda samples.

Integrate piecewise-linear source and total-system throughput functions.
Two-point Gauss quadrature is exact for their product times wavelength on
each interval. This avoids incorrectly assigning a full bin to endpoints.
"""

from __future__ import annotations

import numpy as np

HC_ERG_ANGSTROM = 1.9864458571489286e-8


def validate_wavelengths(wavelength: np.ndarray) -> np.ndarray:
    wave = np.asarray(wavelength, dtype=np.float64)
    if (wave.ndim != 1 or len(wave) < 2 or not np.isfinite(wave).all()
            or np.any(wave <= 0) or np.any(np.diff(wave) <= 0)):
        raise ValueError("Wavelengths must be finite, positive, strictly increasing 1D samples")
    return wave


def photon_bandpass_weights(wavelength: np.ndarray, filter_wavelength: np.ndarray,
                            throughput: np.ndarray) -> np.ndarray:
    """Weights for integral F_lambda*T(lambda)*lambda/(h*c) d(lambda).

    All wavelengths are observed-frame Angstrom. For physical F_lambda in
    erg/s/cm^2/Angstrom, dot(weights, spectrum) gives photons/s/cm^2 detected
    through a total-system throughput. Collecting area and exposure time are
    additional factors. Normalized input flux gives only a relative response.
    No spectrum is extrapolated beyond its supplied wavelength limits.
    """
    wave = validate_wavelengths(wavelength)
    fw = validate_wavelengths(filter_wavelength)
    response = np.asarray(throughput, dtype=np.float64)
    if response.shape != fw.shape or not np.isfinite(response).all():
        raise ValueError("Throughput must be finite and match filter wavelengths")
    if np.any(response < 0) or np.any(response > 1):
        raise ValueError("Total-system throughput must lie in [0,1]")
    left, right = max(wave[0], fw[0]), min(wave[-1], fw[-1])
    weights = np.zeros(len(wave), dtype=np.float64)
    if left >= right:
        return weights
    knots = np.unique(np.r_[left, right, wave[(wave > left) & (wave < right)],
                            fw[(fw > left) & (fw < right)]])
    half = np.diff(knots)/2
    middle = (knots[1:]+knots[:-1])/2
    points = np.r_[middle-half/np.sqrt(3), middle+half/np.sqrt(3)]
    quadrature = np.r_[half, half]
    photon_response = (np.interp(points, fw, response)*points/HC_ERG_ANGSTROM
                       *quadrature)
    lower = np.clip(np.searchsorted(wave, points, side="right")-1, 0, len(wave)-2)
    fraction = (points-wave[lower])/(wave[lower+1]-wave[lower])
    np.add.at(weights, lower, photon_response*(1-fraction))
    np.add.at(weights, lower+1, photon_response*fraction)
    return weights


def bandpass_coverage(wavelength: np.ndarray, filter_wavelength: np.ndarray,
                      throughput: np.ndarray) -> float:
    """Fraction of the tabulated photon response covered for constant F_lambda.

    This does not guarantee that a truncated source line is complete. The
    underlying emission spectrum must also have adequate wavelength coverage.
    """
    full = photon_bandpass_weights(filter_wavelength, filter_wavelength,
                                   throughput).sum()
    if full <= 0:
        raise ValueError("Filter must have a nonzero integrated throughput")
    covered = photon_bandpass_weights(wavelength, filter_wavelength, throughput).sum()
    return float(np.clip(covered/full, 0, 1))


def integrate_bandpass(cube: np.ndarray, wavelength: np.ndarray,
                        filter_wavelength: np.ndarray, throughput: np.ndarray,
                        minimum_coverage: float = 0.999) -> tuple[np.ndarray, np.ndarray, float]:
    """Integrate a (wavelength,y,x) F_lambda cube into one (y,x) response map."""
    array = np.asarray(cube)
    if array.ndim != 3 or array.shape[0] != len(wavelength):
        raise ValueError("Expected cube shape (wavelength,y,x)")
    if not np.isfinite(array).all():
        raise ValueError("Cube contains nonfinite flux")
    coverage = bandpass_coverage(wavelength, filter_wavelength, throughput)
    if coverage < minimum_coverage:
        raise ValueError(
            f"Source covers only {100*coverage:.4f}% of the filter photon response; "
            "extend the internal spectrum before rendering the narrowband image")
    weights = photon_bandpass_weights(wavelength, filter_wavelength, throughput)
    return np.tensordot(weights, array, axes=(0, 0)), weights, coverage
