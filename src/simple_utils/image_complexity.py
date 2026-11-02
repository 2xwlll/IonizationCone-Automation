"""Descriptive component-map diagnostics, not a calibrated difficulty score."""

import numpy as np
from scipy.ndimage import gaussian_filter


def nonnegative_image(value, name, shape=None):
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or not array.size or (shape is not None and array.shape != shape):
        raise ValueError(f'{name}: expected matching nonempty 2D image')
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError(f'{name}: expected finite nonnegative values')
    return array


def component_metrics(cone, other, aperture, *, pixel_scale_arcsec,
                      smoothing_sigmas_arcsec=(0.1, 0.2, 0.4), pixel_area=None,
                      noise_sigma=None):
    """Measure linear, post-PSF components with a fixed boolean aperture.

    Patchiness = sum_A(area * abs(I - Gaussian_sigma(I))) / sum_A(area * I).
    Smooth the full map BEFORE selecting A, with reflective stamp boundaries.
    This measures scale-dependent structure, including gradients and cone edges;
    it is neither a pure cloud statistic nor necessarily bounded by one.

    Scalar/equal pixel areas cancel from ratios. An area map supports weighted
    flux/area ratios, but Gaussian smoothing assumes a uniform angular grid.
    S/N is a per-pixel median, never an integrated significance.
    """
    cone = nonnegative_image(cone, 'cone')
    other = nonnegative_image(other, 'other', cone.shape)
    aperture = np.asarray(aperture)
    if aperture.dtype != bool or aperture.shape != cone.shape:
        raise ValueError('aperture must be a matching boolean array')
    if not np.isfinite(pixel_scale_arcsec) or pixel_scale_arcsec <= 0:
        raise ValueError('pixel scale must be positive and finite')
    scales = np.asarray(smoothing_sigmas_arcsec, dtype=float)
    if scales.ndim != 1 or not len(scales) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('smoothing sigmas must be a nonempty list of finite positive values')
    area = np.ones(cone.shape) if pixel_area is None else np.broadcast_to(
        np.asarray(pixel_area, dtype=float), cone.shape)
    if not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError('pixel areas must be finite and positive')
    if noise_sigma is not None:
        noise_sigma = np.asarray(noise_sigma, dtype=float)
        if noise_sigma.shape != cone.shape:
            raise ValueError('noise sigma shape mismatch')
    weights = area[aperture]
    csum = float(np.sum(cone[aperture] * weights))
    bsum = float(np.sum(other[aperture] * weights))
    count = int(aperture.sum())
    structure = []
    for sigma in scales:
        values = {}
        for name, layer, total in [('cone', cone, csum), ('other', other, bsum)]:
            smooth = gaussian_filter(layer, sigma / pixel_scale_arcsec, mode='reflect', truncate=4.)
            residual = np.abs(layer - smooth)
            values[name] = float(np.sum(residual[aperture] * weights) / total) if total > 0 else None
        structure.append({'gaussian_sigma_arcsec': float(sigma), **values})
    noise_valid = np.zeros(cone.shape, dtype=bool)
    snr = None
    if noise_sigma is not None:
        noise_valid = aperture & np.isfinite(noise_sigma) & (noise_sigma > 0)
        if noise_valid.any():
            snr = float(np.median(cone[noise_valid] / noise_sigma[noise_valid]))
    return {
        'status': 'empty_aperture' if not count else ('zero_cone_signal' if csum == 0 else 'ok'),
        'aperture_pixels': count,
        'cone_weighted_sum': csum,
        'other_weighted_sum': bsum,
        'other_over_cone': bsum / csum if csum > 0 else None,
        'cone_fraction': csum / (csum + bsum) if csum + bsum > 0 else None,
        'other_dominant_area_fraction': float(np.sum(weights * (other[aperture] > cone[aperture]))
                                              / weights.sum()) if count else None,
        'structure': structure,
        'median_cone_signal_over_pixel_sigma': snr,
        'noise_valid_pixels': int(noise_valid.sum()),
    }


def psf_axis_fwhm(kernel, pixel_scale_arcsec):
    """First half-maximum crossings through the peak, interpolated in x and y.

    This measures the supplied kernel core, not its wings or a detector/drizzle
    PSF absent from that kernel. The arithmetic mean is an explicit approximation.
    """
    kernel = nonnegative_image(kernel, 'PSF')
    if not np.isfinite(pixel_scale_arcsec) or pixel_scale_arcsec <= 0:
        raise ValueError('PSF scale must be finite and positive')
    if kernel.max() <= 0:
        raise ValueError('PSF has no signal')
    y, x = np.unravel_index(np.argmax(kernel), kernel.shape)

    def width(profile, peak):
        half = profile[peak] / 2

        def crossing(values):
            below = np.flatnonzero(values <= half)
            if not len(below):
                raise ValueError('PSF does not contain both half-maximum crossings')
            j = int(below[0])
            return j - 1 + (values[j-1] - half) / (values[j-1] - values[j])

        return (crossing(profile[peak::-1]) + crossing(profile[peak:])) * pixel_scale_arcsec

    wx, wy = float(width(kernel[y, :], x)), float(width(kernel[:, x], y))
    return {'x_arcsec': wx, 'y_arcsec': wy, 'mean_arcsec': (wx + wy) / 2}
