"""Coarse empirical image constraints; no inference of physical gas fractions."""
import numpy as np
from scipy.optimize import nnls, lsq_linear

GROUPS = ('cone', 'host', 'stellar', 'nuclear')


def component_group(name):
    if name.startswith('starforming_'):
        return 'starforming'
    return next(group for group in GROUPS if name.startswith(group+'_'))


def annular_means(image, valid, radius, edges):
    means = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        pixels = valid & (radius >= lo) & (radius < hi)
        means.append(float(np.mean(image[pixels])) if pixels.any() else np.nan)
    return np.array(means)


def fit_annular_templates(templates, observed, valid, radius, edges, fixed_first=None, upper_bounds=None):
    """Fit nonnegative template amplitudes and baseline to annular averages.

    Equal fractional residual weighting avoids letting the bright nucleus alone
    determine the fit. Annuli are descriptive constraints, not independent
    Gaussian measurements; no statistical confidence intervals are inferred.
    """
    profiles = np.array([annular_means(layer, valid, radius, edges)
                         for layer in templates])
    target = annular_means(observed, valid, radius, edges)
    usable = np.isfinite(target) & np.isfinite(profiles).all(axis=0)
    design = np.column_stack((profiles[:,usable].T, np.ones(usable.sum())))
    if design.shape[0] < design.shape[1]:
        raise ValueError('Not enough populated annuli for template fit')
    weight = 1 / np.maximum(np.abs(target[usable]), 1.)
    def solve(matrix, values, upper):
        if upper is None:
            return nnls(matrix, values)[0]
        fit = lsq_linear(matrix, values, bounds=(np.zeros(matrix.shape[1]), upper), method='bvls')
        if not fit.success:
            raise RuntimeError(fit.message)
        return fit.x

    if fixed_first is None:
        coefficients = solve(design*weight[:,None], target[usable]*weight, upper_bounds)
    else:
        if fixed_first < 0:
            raise ValueError('Fixed template amplitude must be nonnegative')
        free = solve(design[:,1:]*weight[:,None],
                     (target[usable]-design[:,0]*fixed_first)*weight,
                     None if upper_bounds is None else np.asarray(upper_bounds)[1:])
        coefficients = np.r_[fixed_first, free]
    return coefficients, profiles, target, usable
