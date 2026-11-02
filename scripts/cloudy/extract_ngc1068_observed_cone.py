#!/usr/bin/env python3
"""Measure candidate projected [O III] support and compare with real F187N.

Uses the downloaded HST continuum pair, not a U-Net or synthetic cone template.
Preserves signed residuals and reports sensitivity to analysis choices.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import warnings

os.environ.setdefault('MPLCONFIGDIR', '/tmp/agn-observed-cone-matplotlib')
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OIII = ROOT/'data/2d/MAST_2026-04-14T0111/HST/hst_5754_01_wfpc2_pc_f502n_u2m301/hst_5754_01_wfpc2_pc_f502n_u2m301_drz.fits'
CONT = ROOT/'data/2d/MAST_continuum/mastDownload/HST/hst_5754_01_wfpc2_pc_f547m_u2m301/hst_5754_01_wfpc2_pc_f547m_u2m301_drz.fits'
NIR = ROOT/'data/2d/jwst_ngc1068/jw03707-o128_t002_nircam_clear-f187n_i2d.fits'
CENTER = SkyCoord(40.66962166666667*u.deg, -.01329444444445471*u.deg)
SCALE = .04553
SIZE = 401


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def output_grid():
    w = WCS(naxis=2)
    w.wcs.crpix = [(SIZE+1)/2, (SIZE+1)/2]
    w.wcs.crval = [CENTER.ra.deg, CENTER.dec.deg]
    w.wcs.cdelt = [-SCALE/3600, SCALE/3600]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    yy, xx = np.indices((SIZE, SIZE))
    sky = w.pixel_to_world(xx, yy)
    east, north = CENTER.spherical_offsets_to(sky)
    return w, sky, east.arcsec, north.arcsec


def sample_image(path, sky):
    """Interpolate only a small WCS-selected cutout; require valid footprint."""
    with fits.open(path, memmap=True) as hdus:
        source_wcs = WCS(hdus['SCI'].header).celestial
        x, y = source_wcs.world_to_pixel(sky)
        x0, x1 = int(np.floor(x.min()))-2, int(np.ceil(x.max()))+3
        y0, y1 = int(np.floor(y.min()))-2, int(np.ceil(y.max()))+3
        if x0 < 0 or y0 < 0 or x1 > hdus['SCI'].data.shape[1] or y1 > hdus['SCI'].data.shape[0]:
            raise ValueError('Requested comparison grid extends outside source stamp')
        a = np.array(hdus['SCI'].data[y0:y1, x0:x1], dtype=float)
        valid = np.isfinite(a) & (hdus['WHT'].data[y0:y1, x0:x1] > 0)
        coords = [y-y0, x-x0]
        coverage = map_coordinates(valid.astype(float), coords, order=1, mode='constant', cval=0.)
        value = map_coordinates(np.where(valid, a, 0.), coords, order=1, mode='constant', cval=0.)
        value = np.divide(value, coverage, out=np.full_like(value, np.nan), where=coverage>.999)
        fields = ['BUNIT', 'PHOTFLAM', 'PHOTPLAM', 'EXPTIME', 'FILTER', 'DATE-OBS']
        meta = {key: hdus['SCI'].header.get(key, hdus[0].header.get(key)) for key in fields}
    return value, np.isfinite(value), {'path': str(path), 'sha256': digest(path), 'header': meta,
                                     'resampling': 'bilinear SCI interpolation using FITS celestial WCS; not a flux-calibrated or empirical astrometric registration'}


def smooth_valid(image, valid, sigma_arcsec):
    weight = gaussian_filter(valid.astype(float), sigma_arcsec/SCALE)
    smooth = gaussian_filter(np.where(valid, image, 0.), sigma_arcsec/SCALE)
    return np.divide(smooth, weight, out=np.full_like(smooth, np.nan), where=weight>.995)


def circular_regions(support):
    support = np.asarray(support, dtype=bool)
    if support.all() or not support.any():
        return []  # no distinct edge; never supply an assumed fallback cone
    starts = np.flatnonzero(support & ~np.roll(support, 1))
    regions = []
    for start in starts:
        length = 0
        while support[(start+length) % len(support)]:
            length += 1
        regions.append((int(start), length))
    return regions


def angular_measurement(image, valid, east, north, inner, outer, threshold):
    radius = np.hypot(east, north)
    pa = np.rad2deg(np.arctan2(east, north)) % 360
    edges = np.linspace(inner, outer, 13)
    angle_index = (pa // 3).astype(int) % 120
    polar = np.full((12, 120), np.nan)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        ring = valid & np.isfinite(image) & (radius >= lo) & (radius < hi)
        for j in range(120):
            pixels = ring & (angle_index == j)
            if pixels.sum() >= 2:
                polar[i, j] = np.median(image[pixels])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        profile = np.nanmedian(polar, axis=0)
    if not np.isfinite(profile).all():
        return {'status': 'insufficient_angular_coverage', 'primary': None, 'counter': None}, profile
    profile = gaussian_filter1d(profile, 5./3., mode='wrap')
    peak = float(profile.max())
    lobes = []
    if peak > 0:
        for start, length in circular_regions(profile > threshold*peak):
            width = length*3.
            if not 15 <= width <= 160:
                continue
            ids = (start+np.arange(length)) % 120
            lobes.append({'pa_east_of_north_deg': float((start+length/2)*3 % 360),
                          'opening_full_deg': width, 'opening_half_deg': width/2,
                          'edge_start_pa_deg': float(start*3),
                          'edge_end_pa_deg': float((start+length)*3 % 360),
                          'positive_profile_sum': float(np.maximum(profile[ids], 0).sum())})
    lobes.sort(key=lambda item: item['positive_profile_sum'], reverse=True)
    primary = lobes[0] if lobes else None
    counter = None
    if primary:
        for lobe in lobes[1:]:
            separation = abs((lobe['pa_east_of_north_deg']-primary['pa_east_of_north_deg']+180)%360-180)
            if separation >= 120:
                counter = lobe
                break
    return {'status': 'candidate' if primary else 'no_distinct_lobe', 'primary': primary,
            'counter': counter, 'all_lobes': lobes, 'profile_peak': peak}, profile


def sector_mask(east, north, lobe, inner, outer):
    if lobe is None:
        return np.zeros(east.shape, dtype=bool)
    pa = np.rad2deg(np.arctan2(east, north)) % 360
    offset = abs((pa-lobe['pa_east_of_north_deg']+180)%360-180)
    r = np.hypot(east, north)
    return (offset <= lobe['opening_half_deg']) & (r > inner) & (r < outer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT/args.output
    if output.exists():
        raise FileExistsError(output)
    wcs, sky, east, north = output_grid()
    oiii, vo, mo = sample_image(OIII, sky)
    cont, vc, mc = sample_image(CONT, sky)
    nir, vn, mn = sample_image(NIR, sky)
    valid = vo & vc
    radius = np.hypot(east, north)
    outer = valid & (radius > 6.) & (radius < 8.) & (cont > 0)
    alpha = float(np.median(oiii[outer]) / np.median(cont[outer]))
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('No positive empirical continuum scale')
    output.mkdir(parents=True, exist_ok=False)
    residual = oiii-alpha*cont  # preserve negative values
    default_smooth = smooth_valid(residual, valid, .1)
    nominal, profile = angular_measurement(default_smooth, valid, east, north, .7, 3.7, .2)
    if nominal['primary'] is None:
        raise ValueError('No candidate under nominal settings; no fallback geometry invented')
    scenarios = []
    frequency = np.zeros_like(oiii)
    shifts = [(0., 0.), (-.1, 0.), (.1, 0.), (0., -.1), (0., .1)]
    radial_ranges = [(.7, 3.7), (1., 3.), (.7, 5.)]
    # One-factor sweeps plus a joint continuum/threshold grid; not a posterior.
    settings = [(a, .1, .7, 3.7, t, 0., 0.) for a in [.8, 1., 1.2] for t in [.1, .2, .3, .4]]
    settings += [(1., s, lo, hi, .2, de, dn) for s in [.06, .1, .15]
                 for lo, hi in radial_ranges for de, dn in shifts]
    settings = list(dict.fromkeys(settings))
    for factor, sigma, lo, hi, threshold, de, dn in settings:
        smoothed = smooth_valid(oiii-alpha*factor*cont, valid, sigma)
        measurement, _ = angular_measurement(smoothed, valid, east-de, north-dn, lo, hi, threshold)
        record = {'continuum_factor': factor, 'smoothing_sigma_arcsec': sigma,
                  'inner_arcsec': lo, 'outer_arcsec': hi, 'relative_edge_threshold': threshold,
                  'center_east_shift_arcsec': de, 'center_north_shift_arcsec': dn, **measurement}
        scenarios.append(record)
        # Compare orientation/width on the SAME nominal radial interval.
        frequency += sector_mask(east-de, north-dn, measurement['primary'], .7, 3.7)
    frequency /= len(scenarios)
    primary_mask = sector_mask(east, north, nominal['primary'], .7, 3.7) & valid
    counter_mask = sector_mask(east, north, nominal['counter'], .7, 3.7) & valid
    measured = [s['primary'] for s in scenarios if s['primary'] is not None]
    pa0 = nominal['primary']['pa_east_of_north_deg']
    offsets = [((m['pa_east_of_north_deg']-pa0+180)%360-180) for m in measured]
    sensitivity = {'scenario_count': len(scenarios), 'candidate_count': len(measured),
                   'pa_offset_from_nominal_min_max_deg': [min(offsets), max(offsets)],
                   'full_opening_min_max_deg': [min(m['opening_full_deg'] for m in measured), max(m['opening_full_deg'] for m in measured)],
                   'meaning': 'range across explicit analysis choices, not statistical confidence or an intrinsic cone-angle distribution'}
    report = {
        'status': 'candidate observed projected [O III] emission support; reviewable reference, not a ground-truth torus boundary',
        'sources': [mo, mc, mn], 'code_sha256': digest(Path(__file__)),
        'center_ra_dec_deg': [CENTER.ra.deg, CENTER.dec.deg],
        'center_source': 'same adopted sky nucleus as saved F187N reference; not optimized to cone emission',
        'grid': {'shape': [SIZE, SIZE], 'pixel_scale_arcsec': SCALE, 'orientation': 'north up, east left'},
        'continuum': {'scale': alpha, 'method': 'ratio of medians F502N/F547M in 6 < r < 8 arcsec, valid footprint',
                      'photflam_ratio_from_headers': mc['header']['PHOTFLAM']/mo['header']['PHOTFLAM'],
                      'status': 'empirical continuum proxy; annulus contains galaxy light and may contain emission; not blank sky or an SED fit'},
        'nominal_settings': {'inner_arcsec': .7, 'outer_arcsec': 3.7, 'smoothing_sigma_arcsec': .1,
                             'angular_bins_deg': 3., 'angular_smoothing_sigma_deg': 5., 'relative_edge_threshold': .2,
                             'radial_bins': 12, 'profile': 'median over radial-bin medians of signed smoothed residual',
                             'primary_selection': 'largest positive angular-profile sum among contiguous 15-160 degree sectors; no angle prior'},
        'nominal': nominal, 'sensitivity': sensitivity, 'scenarios': scenarios,
        'limitations': [
            'F502N minus scaled F547M is a line-emission proxy, not proof every residual pixel is AGN cone gas.',
            'Relative brightness thresholds define apparent support; no detection significance or intrinsic opening is inferred.',
            'Both HST bands receive equal additional smoothing, not empirically matched PSFs; nucleus is excluded and smoothing is varied.',
            'FITS WCS alignment only; no field-star absolute registration between HST and JWST. Center shifts are sensitivity trials, not measured registration errors.',
            'JWST F187N remains total band light: no matched near-IR continuum image is downloaded for this analysis.',
            'Optical and near-IR tracers and epochs differ; optical support cannot determine F187N cone/contaminant flux fractions.',
            'Chosen outer radii are measurement apertures, not detections of the physical cone termination.',
            'Scenario frequency is an analysis-stability diagnostic, not a probability or an independent training label.',
        ],
        'synthetic_use': {
            'use': ['candidate sky orientation', 'apparent support-width sensitivity', 'observed spatial morphology and radial dependence for comparison'],
            'do_not_use': ['F187N component amplitude calibration', 'intrinsic half-angle without inclination/projection model', 'pasted real pixels labeled as independent simulations'],
            'evaluation': 'NGC 1068 becomes a calibration reference when used to tune generation; reserve other data for independent validation',
            'generator_changes_applied': False,
        },
        'literature_context': ['https://authors.library.caltech.edu/records/jf1m7-83d33', 'https://arxiv.org/abs/astro-ph/0007018'],
    }
    (output/'measurements.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    np.savez_compressed(output/'observed_reference.npz', f502n=oiii, f547m=cont, f187n=nir,
                        signed_line_proxy=residual, smoothed_line_proxy=default_smooth,
                        hst_valid=valid, f187n_valid=vn, primary_sector=primary_mask,
                        counter_sector=counter_mask, scenario_primary_frequency=frequency,
                        east_arcsec=east, north_arcsec=north, angular_profile=profile,
                        wcs_header=wcs.to_header().tostring(), metadata=json.dumps(report))
    fits.writeto(output/'signed_oiii_proxy.fits', residual, wcs.to_header(), overwrite=False)
    fits.writeto(output/'candidate_primary_sector.fits', primary_mask.astype(np.uint8), wcs.to_header(), overwrite=False)
    with (output/'sensitivity.csv').open('w', newline='') as handle:
        rows = [{k: s[k] for k in ['continuum_factor', 'smoothing_sigma_arcsec', 'inner_arcsec', 'outer_arcsec', 'relative_edge_threshold', 'center_east_shift_arcsec', 'center_north_shift_arcsec', 'status']} |
                {'pa_deg': s['primary']['pa_east_of_north_deg'] if s['primary'] else None,
                 'full_opening_deg': s['primary']['opening_full_deg'] if s['primary'] else None} for s in scenarios]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    plot(output, oiii, cont, nir, residual, primary_mask, counter_mask, frequency, report, profile)
    (output/'README.md').write_text(
        '# Observed NGC 1068 cone candidate\n\n'
        'See `comparison.png`, `sensitivity.png`, and `measurements.json`. The signed HST '
        'line proxy, WCS, candidate sectors, and analysis-stability map are in `observed_reference.npz`. '
        'The primary sector is a threshold-defined geometric envelope, not a pure-emission pixel label.\n\n'
        'No synthetic amplitude, geometry, or training data has been changed. Use this as a '
        'candidate observed morphology reference. F187N remains unsubtracted total band light; '
        'this analysis cannot measure its cone fraction or M. All aperture and processing '
        'choices and their limitations are recorded in `measurements.json`.\n')
    print(json.dumps({'nominal': nominal, 'sensitivity': sensitivity, 'continuum_scale': alpha}, indent=2))


def plot(output, oiii, cont, nir, residual, primary, counter, frequency, report, profile):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import AsinhNorm, SymLogNorm
    extent = np.array([SIZE/2, -SIZE/2, -SIZE/2, SIZE/2])*SCALE
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.7), constrained_layout=True)
    alpha = report['continuum']['scale']
    for i, (ax, image, title) in enumerate(zip(axes, [oiii, alpha*cont, residual, nir],
             ['Observed HST F502N', 'Scaled observed F547M', 'Signed [O III] proxy', 'Observed JWST F187N'])):
        view = image[90:-90, 90:-90]
        top = float(np.nanpercentile(view, 99))
        norm = SymLogNorm(linthresh=top/100, vmin=-top/10, vmax=top) if i == 2 else AsinhNorm(linear_width=top/40, vmin=0, vmax=top)
        ax.imshow(image, origin='lower', extent=extent, cmap='RdBu_r' if i == 2 else 'magma', norm=norm)
        ax.contour(primary, levels=[.5], colors='cyan', linewidths=1., extent=extent, origin='lower')
        if counter.any():
            ax.contour(counter, levels=[.5], colors='lime', linewidths=.8, extent=extent, origin='lower')
        ax.plot(0, 0, '+', color='white', ms=7)
        ax.set(title=title, xlabel='East offset (arcsec)', ylabel='North offset (arcsec)', xlim=(5, -5), ylim=(-5, 5))
    lobe = report['nominal']['primary']
    fig.suptitle(f"NGC 1068 observed candidate: PA {lobe['pa_east_of_north_deg']:.1f}° E of N; apparent full width {lobe['opening_full_deg']:.0f}°\n"
                 'Cyan: HST-derived sector; independent display scales; WCS alignment only; F187N is not continuum-subtracted')
    fig.savefig(output/'comparison.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    pa = np.arange(120)*3+1.5
    axes[0].plot(pa, profile/profile.max())
    axes[0].axhline(.2, linestyle='--', color='gray', label='Nominal edge threshold')
    axes[0].set(xlabel='PA east of north (degrees)', ylabel='Profile / peak', title='Measured angular profile', xlim=(0, 360))
    axes[0].legend(fontsize=8)
    rows = [s for s in report['scenarios'] if s['primary']]
    scatter = axes[1].scatter([s['primary']['pa_east_of_north_deg'] for s in rows],
                             [s['primary']['opening_full_deg'] for s in rows],
                             c=[s['relative_edge_threshold'] for s in rows], cmap='viridis')
    axes[1].set(xlabel='Candidate PA (degrees)', ylabel='Apparent full opening (degrees)', title='Analysis choices; not confidence bounds')
    fig.colorbar(scatter, ax=axes[1], label='Relative edge threshold')
    im = axes[2].imshow(frequency, origin='lower', extent=extent, vmin=0, vmax=1, cmap='viridis')
    axes[2].set(xlabel='East offset (arcsec)', ylabel='North offset (arcsec)', title='Fraction of trials covering each pixel', xlim=(4, -4), ylim=(-4, 4))
    fig.colorbar(im, ax=axes[2], label='Scenario frequency (not probability)')
    fig.savefig(output/'sensitivity.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    main()
