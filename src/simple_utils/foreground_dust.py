"""Prescribed foreground screen controls, independent of illumination geometry."""
import numpy as np
from scipy.ndimage import gaussian_filter


def patchy_screen(shape, pixel_scale_arcsec, seed, cfg):
    """Threshold a correlated Gaussian field to an exact whole-stamp area fraction.

    Optical depth is floor + patch increment on covered intrinsic pixels. The
    Gaussian sigma sets a correlation scale, not a measured cloud radius. Use
    a separate RNG stream so dust changes never resample gas or geometry.
    """
    if cfg.get('model') != 'patchy_screen_v1':
        raise ValueError('Unknown foreground dust prescription')
    coverage = float(cfg['covering_fraction'])
    floor, patch = float(cfg['tau_floor_pa_alpha']), float(cfg['tau_patch_pa_alpha'])
    sigma = float(cfg['correlation_sigma_arcsec'])
    values = (coverage, floor, patch, sigma, pixel_scale_arcsec)
    if not np.isfinite(values).all() or not 0 <= coverage <= 1 or min(floor, patch) < 0 or sigma <= 0 or pixel_scale_arcsec <= 0:
        raise ValueError('Invalid foreground screen parameters')
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 74519]))
    field = gaussian_filter(rng.normal(size=shape), sigma/pixel_scale_arcsec, mode='reflect')
    count = int(round(coverage*field.size))
    covered = np.zeros(field.size, dtype=bool)
    if count:
        covered[np.argpartition(field.ravel(), field.size-count)[-count:]] = True
    covered = covered.reshape(shape)
    return floor + patch*covered, covered


def screen_summary(tau_mean, transmission_mean, covering_mean, aperture):
    """Area statistics of pre-PSF screen maps averaged onto output pixels.

    transmission_mean is mean(exp(-tau)) in each output pixel, not exp(-mean
    tau). These are not emission-weighted attenuation, masses, or post-PSF fits.
    """
    count = int(aperture.sum())
    if not count:
        return {'status': 'empty_aperture', 'pixels': 0}
    return {'status': 'ok', 'pixels': count,
            'mean_tau_pa_alpha': float(np.mean(tau_mean[aperture])),
            'mean_transmission_pa_alpha': float(np.mean(transmission_mean[aperture])),
            'covered_area_fraction': float(np.mean(covering_mean[aperture]))}
