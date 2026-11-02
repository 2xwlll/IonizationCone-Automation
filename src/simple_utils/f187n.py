"""Small numerical building blocks for the first relative F187N prototype."""

import re

import numpy as np
from scipy.ndimage import map_coordinates

from src.simple_utils.synthetic_photometry import HC_ERG_ANGSTROM, photon_bandpass_weights

LINE_PATTERN = re.compile(
    r'(?P<label>[A-Za-z][A-Za-z0-9 +]*?)\s+(?P<wave>\d+\.\d+)(?P<unit>[Am])'
    r'\s+(?P<logflux>[+-]?\d+\.\d+)\s+(?P<ratio>\d+\.\d+)'
)


def printed_emergent_lines(text):
    """Recover rounded log fluxes from the final printed emergent table.

    Missing entries are unknown/censored, not measured zeros. The caller must
    track omissions. This does not improve Cloudy's iteration convergence.
    """
    if 'Emergent line intensities' not in text:
        raise ValueError('No emergent line table in Cloudy output')
    table = text.rsplit('Emergent line intensities', 1)[1]
    lines = {}
    for match in LINE_PATTERN.finditer(table):
        label = ' '.join(match['label'].split())
        wave = float(match['wave'])*(1e4 if match['unit']=='m' else 1)
        lines[(label, round(wave, 3))] = 10**float(match['logflux'])
    return lines


def continuum_from_cached_file(path, rest_wave):
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        fields = line.split()
        if len(fields) < 9:
            continue
        try:
            rows.append([float(fields[0]), float(fields[3]), float(fields[8])])
        except ValueError:
            continue
    data = np.asarray(rows)
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError(f'Invalid continuum: {path}')
    data = data[np.argsort(data[:, 0])]
    if data[0, 0] > rest_wave[0] or data[-1, 0] < rest_wave[-1]:
        raise ValueError('Cached continuum does not cover requested wavelengths')
    # Same outward line-free continuum convention as the existing optical grid.
    flambda = np.maximum(data[:, 1]-data[:, 2], 0)/data[:, 0]
    return np.interp(rest_wave, data[:, 0], flambda)


def line_response(rest_a, redshift, velocity_kms, sigma_kms, filter_wave, throughput):
    """Photon response per unit integrated line energy for a resolved Gaussian."""
    if rest_a <= 0 or redshift < 0 or sigma_kms <= 0:
        raise ValueError('Invalid line parameters')
    c = 299792.458
    center = rest_a*(1+redshift)*(1+velocity_kms/c)
    sigma = rest_a*(1+redshift)*sigma_kms/c
    wave = np.linspace(center-8*sigma, center+8*sigma, 401)
    profile = np.exp(-.5*((wave-center)/sigma)**2)/(np.sqrt(2*np.pi)*sigma)
    return float(photon_bandpass_weights(wave, filter_wave, throughput)@profile)


def resample_psf(kernel, old_scale, new_scale):
    """Center-preserving linear resampling of an odd optical kernel; no rotation."""
    kernel = np.asarray(kernel, dtype=float)
    if (kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]
            or kernel.shape[0] % 2 != 1 or np.any(kernel < 0)
            or not np.isfinite(kernel).all() or kernel.sum() <= 0
            or old_scale <= 0 or new_scale <= 0):
        raise ValueError('Invalid centered optical PSF or pixel scales')
    half = int(np.ceil((kernel.shape[0]-1)/2*old_scale/new_scale))
    y, x = np.indices((2*half+1, 2*half+1), dtype=float)
    center = (kernel.shape[0]-1)/2
    coordinates = [(y-half)*new_scale/old_scale+center, (x-half)*new_scale/old_scale+center]
    result = map_coordinates(kernel, coordinates, order=1, mode='constant', cval=0)
    return result/result.sum()


def rebin_surface_brightness(image, factor):
    """Average an oversampled surface-brightness grid into detector-sized pixels."""
    if image.ndim != 2 or factor < 1 or image.shape[0] % factor or image.shape[1] % factor:
        raise ValueError('Image dimensions must be divisible by the rebinning factor')
    return image.reshape(image.shape[0]//factor, factor, image.shape[1]//factor, factor).mean(axis=(1, 3))
