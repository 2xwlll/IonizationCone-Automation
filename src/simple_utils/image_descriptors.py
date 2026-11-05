"""Measured synthetic-image descriptors and explicit, provisional category cuts."""
import numpy as np
from src.simple_utils.image_complexity import component_metrics
from src.simple_utils.foreground_dust import screen_summary

CATEGORY_DEFINITION = {
    'version': 'f187n_descriptive_categories_v1',
    'contrast': 'inside nominal mask and valid 0.3<r<3.3 arcsec annulus: M<1 cone_brighter; 1<=M<3 other_1_to_3; M>=3 other_ge3; cone-free separate',
    'obscuration': 'area-mean pre-PSF transmission at Pa alpha: T>=0.8 weak; 0.5<=T<0.8 moderate; T<0.5 strong; nominal cone aperture for positives, whole annulus for negatives',
    'scope': 'illustrative grouping cuts, not calibrated task difficulty or empirical population classes; patchiness/resolution/noise remain separate continuous descriptors'}


def set_sampled_parameter(cfg, name, value):
    """Strict dotted paths for v3; retain existing v2 flat field routing."""
    if '.' in name:
        keys = name.split('.')
        target = cfg
        for key in keys[:-1]:
            target = target[key]
        if keys[-1] not in target:
            raise KeyError(name)
        target[keys[-1]] = value
    else:
        target = cfg['geometry'] if name in cfg['geometry'] else cfg['components']
        if name not in target:
            raise KeyError(name)
        target[name] = value


def describe_image(cone, other, mask, valid, scale, *, noise_sigma=None, dust_maps=None, dust_applied=True):
    yy, xx = np.indices(mask.shape)
    radius = np.hypot(xx-(mask.shape[1]-1)/2, yy-(mask.shape[0]-1)/2)*scale
    aperture = valid & (radius > .3) & (radius < 3.3)
    regions = {'whole_annulus': aperture, 'cone_annulus': aperture & mask}
    metrics = {name: component_metrics(cone, other, region, pixel_scale_arcsec=scale,
               noise_sigma=noise_sigma) if region.any() else None for name, region in regions.items()}
    dust = {'applied': bool(dust_applied), 'map_semantics':
        'area averages of pre-PSF foreground screen, not emission-weighted or a dust mass', 'regions': {}}
    if dust_maps is not None and dust_applied:
        tau, transmission, coverage = dust_maps
        for name, region in regions.items():
            dust['regions'][name] = screen_summary(tau, transmission, coverage, region) if region.any() else None
    present = bool(mask.any())
    ratio = metrics['cone_annulus']['other_over_cone'] if metrics['cone_annulus'] is not None else None
    contrast = ('cone_free' if not present else 'undefined' if ratio is None else
                'cone_brighter' if ratio < 1 else 'other_1_to_3' if ratio < 3 else 'other_ge3')
    dust_region = 'cone_annulus' if present else 'whole_annulus'
    summary = dust['regions'].get(dust_region)
    transmission = summary.get('mean_transmission_pa_alpha') if summary is not None else None
    obscuration = ('not_applied' if not dust_applied else 'undefined' if transmission is None else
                   'weak' if transmission >= .8 else 'moderate' if transmission >= .5 else 'strong')
    return metrics, dust, {'version': CATEGORY_DEFINITION['version'], 'contrast': contrast,
                           'obscuration': obscuration, 'obscuration_region': dust_region}
