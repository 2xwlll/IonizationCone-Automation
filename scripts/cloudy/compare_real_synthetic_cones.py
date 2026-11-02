#!/usr/bin/env python3
"""Matched F187N angular-excess measurements, mask audit, and orientation audit."""
import argparse
import json
import os
from pathlib import Path
import sys
import warnings

os.environ.setdefault('MPLCONFIGDIR', '/tmp/agn-cone-comparison-matplotlib')
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates
from scipy.stats import binned_statistic_2d

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.cloudy.measure_f187n_complexity import load_components
from scripts.cloudy.extract_ngc1068_observed_cone import circular_regions, digest, sector_mask
from src.simple_utils.cone_geometry import resolve_cone_geometry


def sky_offsets(shape, center_xy, matrix_arcsec):
    y, x = np.indices(shape, dtype=float)
    return np.einsum('ij,jkl->ikl', matrix_arcsec,
                     np.array([x-center_xy[0], y-center_xy[1]]))


def smooth_image(image, valid, sigma_pixels):
    weight = gaussian_filter(valid.astype(float), sigma_pixels)
    value = gaussian_filter(np.where(valid, image, 0.), sigma_pixels)
    return np.divide(value, weight, out=np.full_like(value, np.nan), where=weight>.995)


def excess_profile(image, valid, east, north, inner=.7, outer=3.3):
    """Median over radial-bin fractional angular excess; not continuum subtraction.

    Each polar cell is divided by the full annulus median and has one subtracted.
    Baselines/continuum remain in both numerator and denominator. The operator
    is identical for observed and simulated total images, and does not see labels.
    """
    radius = np.hypot(east, north)
    pa = np.rad2deg(np.arctan2(east, north)) % 360
    good = valid & np.isfinite(image) & (radius>=inner) & (radius<outer)
    redges, aedges = np.linspace(inner, outer, 13), np.linspace(0., 360., 121)
    table = binned_statistic_2d(radius[good], pa[good], image[good], statistic='median',
                               bins=[redges, aedges]).statistic
    counts = np.histogram2d(radius[good], pa[good], bins=[redges, aedges])[0]
    table[counts<2] = np.nan
    ring_medians = []
    for i in range(12):
        ring = good & (radius>=redges[i]) & (radius<redges[i+1])
        value = float(np.median(image[ring])) if ring.any() else np.nan
        ring_medians.append(value)
        table[i] = table[i]/value-1 if np.isfinite(value) and value>0 else np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        profile = np.nanmedian(table, axis=0)
    if not np.isfinite(profile).all():
        raise ValueError('Incomplete polar coverage; do not fill missing angles')
    return gaussian_filter1d(profile, 5./3., mode='wrap'), table, ring_medians


def measure_lobes(profile, threshold):
    peak = float(np.max(profile))
    lobes = []
    if peak > 1e-12:
        for start, length in circular_regions(profile > threshold*peak):
            if 15 <= length*3 <= 160:
                ids = (start+np.arange(length)) % 120
                lobes.append({'pa_east_of_north_deg': float((start+length/2)*3 % 360),
                              'opening_full_deg': float(length*3), 'opening_half_deg': float(length*1.5),
                              'positive_profile_sum': float(np.maximum(profile[ids], 0).sum())})
    lobes.sort(key=lambda item: item['positive_profile_sum'], reverse=True)
    primary = lobes[0] if lobes else None
    counter = next((l for l in lobes[1:] if abs((l['pa_east_of_north_deg']-
                    primary['pa_east_of_north_deg']+180)%360-180)>=120), None) if primary else None
    return {'status': 'candidate' if primary else 'no_distinct_lobe', 'primary': primary,
            'counter': counter, 'all_lobes': lobes, 'peak_fractional_excess': peak,
            'relative_threshold': threshold}


def measured_mask(measurement, east, north, inner, outer):
    return (sector_mask(east, north, measurement['primary'], inner, outer) |
            sector_mask(east, north, measurement['counter'], inner, outer))


def mask_scores(measured, target, aperture):
    a, b = measured & aperture, target & aperture
    intersection, union = int((a & b).sum()), int((a | b).sum())
    return {'iou': intersection/union if union else None,
            'dice': 2*intersection/(int(a.sum())+int(b.sum())) if a.any() or b.any() else None}


def angle_error(pa, target):
    return float(abs((pa-target+90)%180-90))


def north_up(image, valid, center_xy, matrix_arcsec, scale, order=1):
    n = 257
    y, x = np.indices((n, n))
    east, north = -(x-128)*scale, (y-128)*scale
    delta = np.einsum('ij,jkl->ikl', np.linalg.inv(matrix_arcsec), np.array([east, north]))
    coords = [delta[1]+center_xy[1], delta[0]+center_xy[0]]
    weight = map_coordinates(valid.astype(float), coords, order=order, mode='constant', cval=0.)
    values = map_coordinates(np.where(valid, image, 0.), coords, order=order, mode='constant', cval=0.)
    return np.divide(values, weight, out=np.full_like(values, np.nan), where=weight>.999)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic', type=Path, default=Path('results/cloudy/ngc1068_f187n_soft_gas_v1/image_mask.npz'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    path = args.synthetic if args.synthetic.is_absolute() else ROOT/args.synthetic
    output = args.output if args.output.is_absolute() else ROOT/args.output
    if output.exists():
        raise FileExistsError('Use a fresh output directory')
    meta, cone, other, label, _, sigma, synthetic, checks = load_components(path)
    refpath = ROOT/meta['configuration_values']['reference']
    with np.load(refpath, allow_pickle=False) as archive:
        real = np.array(archive['science'], dtype=float)
        valid = np.array(archive['valid'], dtype=bool)
        reference = json.loads(str(archive['metadata']))
    if real.shape != synthetic.shape:
        raise ValueError('Reference and synthetic image shapes differ')
    scale = meta['pixel_scale_arcsec']
    if not np.isclose(scale, reference['pixel_scale_arcsec']):
        raise ValueError('Reference and synthetic pixel scales differ')
    matrix = np.asarray(reference['pixel_to_sky_matrix'])*3600.
    centers = {'observed': reference['nucleus_cutout_xy'],
               'synthetic': [(synthetic.shape[1]-1)/2, (synthetic.shape[0]-1)/2]}
    offsets = {name: sky_offsets(real.shape, center, matrix) for name, center in centers.items()}
    rawpath = Path(reference['source_fits'])
    with fits.open(rawpath, memmap=True) as hdus:
        w = WCS(hdus['SCI'].header).celestial
        x, y = map(float, w.world_to_pixel_values(reference['target_ra_deg'], reference['target_dec_deg']))
        x0, y0 = round(x)-real.shape[1]//2, round(y)-real.shape[0]//2
        raw = hdus['SCI'].data[y0:y0+real.shape[0], x0:x0+real.shape[1]]
        identical = bool(np.array_equal(raw, real, equal_nan=True))
        if not identical:
            raise ValueError('Cached reference differs from original FITS cutout: investigate before measuring')
        # Verify the saved local matrix against direct WCS evaluation near the center.
        coordinates = np.array([[x,y], [x+1,y], [x,y+1]])
        ra, dec = w.pixel_to_world_values(coordinates[:,0], coordinates[:,1])
        finite_diff = np.array([(ra[1:]-ra[0])*np.cos(np.deg2rad(dec[0]))*3600,
                                (dec[1:]-dec[0])*3600])
        if not np.allclose(finite_diff, matrix, rtol=1e-5, atol=1e-7):
            raise ValueError('Saved matrix does not reproduce local FITS WCS')
    transform = np.linalg.inv(matrix) @ np.diag([-scale, scale])
    orientation = {
        'cached_science_equals_original_unflipped_cutout': identical,
        'pixel_to_east_north_arcsec': matrix.tolist(), 'direct_wcs_finite_difference': finite_diff.tolist(),
        'native_positive_y_sky_pa_deg': float(np.rad2deg(np.arctan2(matrix[0,1], matrix[1,1])) % 360),
        'north_up_target_to_native_determinant': float(np.linalg.det(transform)),
        'interpretation': 'Native array axes are rotated relative to north/east. Positive determinant for the north-up transform means no reflection. Origin upper would reverse the displayed y-axis; inspected project previews use origin lower.',
        'display': 'north up, east left; native data and saved masks are unchanged',
    }
    images = {'observed': real, 'synthetic': synthetic, 'synthetic_cone_only': cone}
    measurements, profiles, masks, trials = {}, {}, {}, {}
    for name, image in images.items():
        group = 'observed' if name == 'observed' else 'synthetic'
        east, north = offsets[group]
        smoothed = smooth_image(image, valid, .1/scale)
        profile, polar, medians = excess_profile(smoothed, valid, east, north)
        nominal = measure_lobes(profile, .2)
        profiles[name] = profile
        masks[name] = measured_mask(nominal, east, north, .7, 3.3) & valid
        measurements[name] = {'nominal': nominal, 'radial_medians': medians,
                              'input': 'total F187N band image' if name != 'synthetic_cone_only' else 'known synthetic cone gas only; diagnostic, not an observed input'}
        family = []
        for blur in [.06, .1, .15]:
            smoothed = smooth_image(image, valid, blur/scale)
            for inner, outer in [(.7, 3.3), (1., 3.3), (.7, 2.5)]:
                for de, dn in [(0.,0.), (.1,0.), (-.1,0.), (0.,.1), (0.,-.1)]:
                    p, _, _ = excess_profile(smoothed, valid, east-de, north-dn, inner, outer)
                    for threshold in [.1, .2, .3, .4]:
                        m = measure_lobes(p, threshold)
                        family.append({'sigma_arcsec': blur, 'inner_arcsec': inner, 'outer_arcsec': outer,
                                       'center_east_shift_arcsec': de, 'center_north_shift_arcsec': dn, **m})
        trials[name] = family
        candidates = [row['primary'] for row in family if row['primary']]
        p0 = nominal['primary']['pa_east_of_north_deg'] if nominal['primary'] else None
        delta = [float((l['pa_east_of_north_deg']-p0+90)%180-90) for l in candidates] if p0 is not None else []
        measurements[name]['analysis_sensitivity'] = {
            'trial_count': len(family), 'candidate_count': len(candidates),
            'axis_offsets_mod180_min_max_deg': [min(delta), max(delta)] if delta else None,
            'full_opening_min_max_deg': [min(l['opening_full_deg'] for l in candidates), max(l['opening_full_deg'] for l in candidates)] if candidates else None,
            'interpretation': 'method sensitivity, not a confidence interval or a ranking of network difficulty'}
    east, north = offsets['synthetic']
    radius = np.hypot(east, north)
    aperture = valid & (radius>.7) & (radius<3.3)
    geo = meta['configuration_values']['geometry']
    resolved_geometry = resolve_cone_geometry(geo)
    if 'resolved_geometry' in meta and meta['resolved_geometry'] != resolved_geometry:
        raise ValueError('Saved geometry disagrees with analysis convention')
    expected_width = resolved_geometry['projected_full_opening_deg']
    gas_total = float(cone[aperture].sum())
    captured = float(cone[aperture & label].sum()/gas_total)
    mask_audit = {'aperture': 'valid pixels, 0.7 < r < 3.3 arcsec; geometric target and measured support evaluated identically',
                 'nominal_axis_pa_deg': geo['position_angle_east_of_north_deg'],
                 'nominal_projected_full_opening_deg': expected_width,
                 'resolved_geometry': resolved_geometry,
                 'cone_flux_fraction_inside_geometric_label': captured,
                 'cone_flux_fraction_outside_geometric_label': 1-captured,
                 'cone_fraction_of_components_inside_label': float(cone[aperture & label].sum()/(cone+other)[aperture & label].sum()),
                 'area_fraction_inside_label_other_brighter': float(np.mean(other[aperture & label]>cone[aperture & label])),
                 'measured_vs_geometric': {}, 'gas_visibility_footprints': []}
    for name in ['synthetic', 'synthetic_cone_only']:
        m = measurements[name]['nominal']
        mask_audit['measured_vs_geometric'][name] = {
            **mask_scores(masks[name], label, aperture),
            'primary_axis_error_mod180_deg': angle_error(m['primary']['pa_east_of_north_deg'],geo['position_angle_east_of_north_deg']) if m['primary'] else None,
            'primary_full_width_minus_geometric_deg': m['primary']['opening_full_deg']-expected_width if m['primary'] else None,
            'captured_cone_flux_fraction': float(cone[masks[name] & aperture].sum()/gas_total),
        }
    for threshold in [1., 3., 5.]:
        visible = np.isfinite(sigma) & (sigma>0) & (cone>=threshold*sigma)
        mask_audit['gas_visibility_footprints'].append({'threshold_C_over_pixel_sigma': threshold,
                                                       **mask_scores(label, visible, aperture)})
    # Compare sky-dependent measured label directly with analytic geometric construction.
    phi = np.deg2rad(geo['position_angle_east_of_north_deg'])
    axial, cross = east*np.sin(phi)+north*np.cos(phi), east*np.cos(phi)-north*np.sin(phi)
    analytic = (abs(cross)<=abs(axial)*np.tan(np.deg2rad(expected_width/2)))
    analytic &= radius < geo['radius_fraction']*real.shape[0]/2*scale
    mask_audit['stored_label_vs_analytic_geometry_in_aperture'] = mask_scores(label, analytic, aperture)
    report = {
        'status': 'matched image-based candidate-cone comparison; no U-Net run or real cone-component decomposition',
        'sources': {str(p):digest(p) for p in [path, refpath, rawpath]},
        'code_sha256': {str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__).resolve(),
                        ROOT/'scripts/cloudy/measure_f187n_complexity.py', ROOT/'scripts/cloudy/extract_ngc1068_observed_cone.py',
                        ROOT/'src/simple_utils/cone_geometry.py']},
        'synthetic_configuration': meta['configuration_values'], 'synthetic_component_checks': checks,
        'orientation': orientation, 'centers_native_xy': centers,
        'method': {'band':'F187N for both main inputs', 'native_pixel_scale_arcsec':scale,
                   'nominal_radial_range_arcsec':[.7,3.3], 'additional_smoothing_sigma_arcsec':.1,
                   'polar_grid':[12,120], 'angular_smoothing_sigma_deg':5.,
                   'profile':'median over radial bins of (angular-cell median/full-ring median - 1)',
                   'edge':'20 percent of positive profile peak; 15-160 degree contiguous lobes; no angle/label prior',
                   'limitations':['Radial normalization reveals anisotropy, not pure AGN line emission or continuum subtraction.',
                                  'Same added smoothing is not exact effective-PSF matching; original sampling and PSFs remain approximations.',
                                  'No signal/noise likelihood or independent-pixel significance is inferred.',
                                  'Real F187N has no independent exact mask; error scores apply only to the synthetic label.',
                                  'Higher image contrast or stability does not alone establish which input a trained network finds harder.',
                                  'Soft cone emission outside a nominal label is expected; a geometric label and visible gas footprint are different targets.']},
        'measurements':measurements, 'synthetic_mask_audit':mask_audit,
        'analysis_trials':trials,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output/'measurements.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(output/'comparison_arrays.npz', real=real, synthetic=synthetic, cone_gas=cone,
                        other=other, geometric_label=label, valid=valid,
                        **{f'{name}_measured_mask':value for name,value in masks.items()},
                        **{f'{name}_profile':value for name,value in profiles.items()},
                        metadata=json.dumps(report))
    plot_results(output, report, images, label, valid, centers, matrix, scale, masks, profiles)
    print(json.dumps({'orientation':orientation,'measurements':measurements,'mask_audit':mask_audit},indent=2))


def plot_results(output, report, images, label, valid, centers, matrix, scale, masks, profiles):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import AsinhNorm
    from matplotlib.lines import Line2D
    extent=np.array([128.5,-128.5,-128.5,128.5])*scale
    fig, axes=plt.subplots(1,3,figsize=(16,6.3))
    fig.subplots_adjust(left=.055,right=.99,bottom=.11,top=.77,wspace=.20)
    for ax,name,title in zip(axes,images,['Observed NGC 1068 F187N','Synthetic F187N: all components','Synthetic cone gas only (diagnostic)']):
        group='observed' if name=='observed' else 'synthetic'
        shown=north_up(images[name],valid,centers[group],matrix,scale)
        high=float(np.nanpercentile(shown,99.3))
        ax.imshow(shown,origin='lower',extent=extent,cmap='magma',norm=AsinhNorm(linear_width=high/40,vmin=0,vmax=high))
        detected=north_up(masks[name].astype(float),valid,centers[group],matrix,scale,order=0)
        if np.nansum(detected)>0:
            ax.contour(detected,levels=[.5],colors='cyan',linewidths=1.2,origin='lower',extent=extent)
        if group=='synthetic':
            truth=north_up(label.astype(float),valid,centers[group],matrix,scale,order=0)
            ax.contour(truth,levels=[.5],colors='lime',linestyles='--',linewidths=1.2,origin='lower',extent=extent)
        ax.plot(0,0,'+',color='white')
        ax.set(title=title,xlabel='East offset (arcsec)',ylabel='North offset (arcsec)',xlim=(3.7,-3.7),ylim=(-3.7,3.7))
        m=report['measurements'][name]['nominal'];p=m['primary']
        text=(f"Primary PA {p['pa_east_of_north_deg']:.1f}° | full width {p['opening_full_deg']:.0f}°" if p else 'No distinct lobe')
        ax.text(.02,.02,text,transform=ax.transAxes,color='white',fontsize=9,bbox=dict(facecolor='black',alpha=.65))
    fig.legend(handles=[Line2D([],[],color='cyan',label='Measured angular-excess envelope'),
                        Line2D([],[],color='lime',linestyle='--',label='Synthetic geometric target')],loc='upper center',ncol=2,bbox_to_anchor=(.5,.89))
    fig.suptitle('Same F187N measurement; north up and east left\nIndependent display stretches; envelopes are not pure-emission pixel masks',y=.99)
    fig.savefig(output/'cone_comparison.png',dpi=160,bbox_inches='tight');plt.close(fig)

    fig, axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    angles=np.arange(120)*3+1.5
    for name,profile in profiles.items():
        if name != 'synthetic_cone_only':
            axes[0].plot(angles,profile,label=name.replace('_',' '))
        if profile.max()>0:
            axes[1].plot(angles,profile/profile.max(),label=name.replace('_',' '))
    axes[0].set(xlabel='PA east of north (degrees)',ylabel='Fractional excess over radial median',title='Directional contrast (not component fraction)')
    axes[1].axhline(.2,color='gray',linestyle='--')
    axes[1].set(xlabel='PA east of north (degrees)',ylabel='Profile / positive peak',title='Same relative threshold for each image')
    for ax in axes:ax.legend(fontsize=8);ax.set_xlim(0,360)
    fig.savefig(output/'angular_profiles.png',dpi=160);plt.close(fig)

    real=images['observed'];high=float(np.nanpercentile(real[valid],99.3))
    norm=AsinhNorm(linear_width=high/40,vmin=0,vmax=high)
    fig, axes=plt.subplots(1,3,figsize=(14,4.7),constrained_layout=True)
    for ax,origin,title in zip(axes[:2],['lower','upper'],['Native array, origin=lower (existing preview)','Native array, origin=upper (flip demonstration)']):
        ax.imshow(np.where(valid,real,np.nan),origin=origin,cmap='magma',norm=norm)
        cx,cy=centers['observed']
        for skyvector,text,color in [([0,1.3],'N','cyan'),([1.3,0],'E','lime')]:
            dx,dy=np.linalg.solve(matrix,skyvector)
            ax.annotate('',xy=(cx+dx,cy+dy),xytext=(cx,cy),
                        arrowprops=dict(arrowstyle='->',color=color,lw=1.5))
            ax.text(cx+1.15*dx,cy+1.15*dy,text,color=color,fontsize=12,
                    ha='center',va='center',bbox=dict(facecolor='black',alpha=.5,pad=1))
        ax.set(title=title,xlabel='Native x pixel',ylabel='Native y pixel')
    shown=north_up(real,valid,centers['observed'],matrix,scale)
    axes[2].imshow(shown,origin='lower',extent=extent,cmap='magma',norm=norm)
    for endpoint,text,color in [((0,1.3),'N','cyan'),((1.3,0),'E','lime')]:
        axes[2].annotate('',xy=endpoint,xytext=(0,0),arrowprops=dict(arrowstyle='->',color=color,lw=1.5))
        axes[2].text(endpoint[0]*1.15,endpoint[1]*1.15,text,color=color,fontsize=12,
                     ha='center',va='center',bbox=dict(facecolor='black',alpha=.5,pad=1))
    axes[2].set(title='WCS-oriented display: north up, east left',xlabel='East offset (arcsec)',ylabel='North offset (arcsec)')
    fig.suptitle('The saved real cutout matches the original FITS pixels exactly; only the display convention changes')
    fig.savefig(output/'orientation_audit.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    main()
