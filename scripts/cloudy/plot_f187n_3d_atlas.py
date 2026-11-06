#!/usr/bin/env python3
"""Read-only 3D descriptor surfaces with their synthetic F187N images beside them."""
import argparse
import gc
import html
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/agn-f187n-3d-mpl')
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import AsinhNorm
from PIL import Image
from src.simple_utils.f187n_pilot_dataset import validate_manifest, sha256, write_json
from scripts.cloudy.measure_f187n_complexity import load_components


FIELDS = {
    'fraction': ('Other-light fraction B / (C+B)', 'B / (C+B)', 'viridis', (0., 1.)),
    'cone_residual': ('Cone spatial structure', 'log10(1 + R_C)', 'inferno', None),
    'other_residual': ('Other-light spatial structure', 'log10(1 + R_B)', 'inferno', None),
    'tau': ('Foreground optical depth', 'Mean tau in output pixel', 'cividis', None),
    'transmission': ('Foreground transmission', 'Mean exp(-tau) in output pixel', 'viridis', (0., 1.)),
    'signal_sigma': ('Cone signal / pixel noise amplitude', 'log10(1 + C / sigma)', 'magma', None),
}


def patch_value(row, component):
    metrics = row['complexity_regions']['cone_annulus']
    return next(s[component] for s in metrics['structure'] if np.isclose(s['gaussian_sigma_arcsec'], .2))


def measured_value(row, key):
    metrics = row['complexity_regions']['cone_annulus']
    if key == 'M':
        return metrics['other_over_cone']
    if key == 'P_B':
        return patch_value(row, 'other')
    if key == 'T':
        return row['dust_diagnostics']['regions']['cone_annulus']['mean_transmission_pa_alpha']
    if key == 'snr':
        return metrics['median_cone_signal_over_pixel_sigma']
    raise KeyError(key)


def representative_rows(rows, key, count):
    ordered = sorted((r for r in rows if measured_value(r, key) is not None),
                     key=lambda r: (measured_value(r, key), r['id']))
    if not ordered:
        raise ValueError('No defined measurements for '+key)
    indices = np.unique(np.rint(np.linspace(0, len(ordered)-1, min(count, len(ordered)))).astype(int))
    return [ordered[i] for i in indices]


def spatial_fields(cone, other, noise, aperture, scale, dust_tau, dust_transmission):
    """Full-resolution diagnostic fields, before any rendering subsampling.

    Mean_A(R_I) equals the stored normalized absolute residual P_I. The local
    fraction B/(C+B) is bounded, but its mean is NOT the summed ratio M.
    """
    total = cone+other
    fraction = np.divide(other, total, out=np.full_like(total, np.nan), where=total > 0)
    raw_residuals = {}
    for name, layer in [('cone_residual', cone), ('other_residual', other)]:
        mean = float(layer[aperture].mean())
        smooth = gaussian_filter(layer, .2/scale, mode='reflect', truncate=4.)
        raw_residuals[name] = abs(layer-smooth)/mean if mean > 0 else np.full_like(layer, np.nan)
    signal_sigma = np.divide(cone, noise, out=np.full_like(cone, np.nan),
                             where=np.isfinite(noise) & (noise > 0))
    values = dict(fraction=fraction, tau=dust_tau, transmission=dust_transmission,
                  signal_sigma=np.log10(1+signal_sigma),
                  **{key: np.log10(1+value) for key, value in raw_residuals.items()})
    summaries = {name: float(np.mean(value[aperture])) for name, value in raw_residuals.items()}
    return values, summaries


def load_example(root, row, manifest):
    source = root/row['source_archive']
    if sha256(source) != row['source_sha256']:
        raise ValueError('Changed source archive: '+str(source))
    meta, cone, other, mask, valid, noise, observed, checks = load_components(source)
    if not np.array_equal(mask, np.load(root/row['mask'], allow_pickle=False).astype(bool)):
        raise ValueError('Source mask disagrees with training target')
    with np.load(source, allow_pickle=False) as saved:
        try:
            tau = np.array(saved['foreground_tau_pa_alpha_mean'])
            transmission = np.array(saved['foreground_transmission_pa_alpha_mean'])
            coverage = np.array(saved['foreground_covering_fraction'])
        except KeyError as exc:
            raise ValueError('This atlas requires the saved foreground-screen maps from recipe v3') from exc
    scale = manifest['pixel_scale_arcsec']
    if not np.isclose(scale, meta['pixel_scale_arcsec']):
        raise ValueError('Pixel scale mismatch')
    yy, xx = np.indices(mask.shape, dtype=float)
    delta = np.array([xx-(mask.shape[1]-1)/2, yy-(mask.shape[0]-1)/2])
    east, north = np.einsum('ij,jkl->ikl', np.array(manifest['matrix_arcsec']), delta)
    radius = np.hypot(delta[0], delta[1])*scale
    aperture = valid & mask & (radius > .3) & (radius < 3.3)
    if not aperture.any():
        raise ValueError('Empty nominal-cone aperture')
    fields, residuals = spatial_fields(cone, other, noise, aperture, scale, tau, transmission)
    measured = row['complexity_regions']['cone_annulus']
    actual = {'M': float(other[aperture].sum()/cone[aperture].sum()),
              'P_C': residuals['cone_residual'], 'P_B': residuals['other_residual'],
              'T': float(transmission[aperture].mean()), 'tau': float(tau[aperture].mean()),
              'coverage': float(coverage[aperture].mean())}
    expected = {'M': measured['other_over_cone'], 'P_C': patch_value(row, 'cone'),
                'P_B': patch_value(row, 'other'),
                'T': row['dust_diagnostics']['regions']['cone_annulus']['mean_transmission_pa_alpha'],
                'tau': row['dust_diagnostics']['regions']['cone_annulus']['mean_tau_pa_alpha'],
                'coverage': row['dust_diagnostics']['regions']['cone_annulus']['covered_area_fraction']}
    for key, value in expected.items():
        if not np.isclose(actual[key], value, rtol=1e-10, atol=1e-12):
            raise ValueError('Atlas recomputation disagrees with saved '+key)
    return dict(row=row, east=east, north=north, aperture=aperture, valid=valid,
                observed=observed, fields=fields, actual=actual, checks=checks)


def image_panel(ax, example, norm, title=None):
    ax.pcolormesh(example['east'], example['north'],
                  np.ma.array(example['observed'], mask=~example['valid']),
                  shading='nearest', cmap='magma', norm=norm, rasterized=True)
    ax.contour(example['east'], example['north'], example['aperture'],
               levels=[.5], colors='#00d9f5', linewidths=.85)
    ax.set(xlim=(3.6, -3.6), ylim=(-3.6, 3.6), xlabel='East offset (arcsec)', ylabel='North offset (arcsec)')
    ax.set_aspect('equal')
    ax.set_title(title or example['row']['id']+' | synthetic noisy input', fontsize=9)
    ax.tick_params(labelsize=8)


def surface_panel(ax, example, key, limits, stride):
    title, zlabel, cmap, _ = FIELDS[key]
    shape = example['aperture'].shape
    iy = np.unique(np.r_[np.arange(0, shape[0], stride), shape[0]-1])
    ix = np.unique(np.r_[np.arange(0, shape[1], stride), shape[1]-1])
    selection = np.ix_(iy, ix)
    values = np.where(example['aperture'], example['fields'][key], np.nan)
    surface = ax.plot_surface(example['east'][selection], example['north'][selection], values[selection],
        cmap=cmap, vmin=limits[0], vmax=limits[1], rcount=len(iy), ccount=len(ix),
        linewidth=0, antialiased=False, shade=False, rasterized=True)
    ax.contour(example['east'], example['north'], example['aperture'].astype(float),
               levels=[.5], zdir='z', offset=limits[0], colors='#00bcd4', linewidths=.7)
    ax.set(xlim=(3.6, -3.6), ylim=(-3.6, 3.6), zlim=limits,
           xlabel='East (arcsec)', ylabel='North (arcsec)', zlabel=zlabel)
    ax.set_title(title, fontsize=10, pad=4)
    ax.view_init(elev=33, azim=-132)
    ax.set_box_aspect((1, 1, .65))
    ax.tick_params(labelsize=7, pad=0)
    ax.xaxis.label.set_size(8); ax.yaxis.label.set_size(8); ax.zaxis.label.set_size(8)
    return surface


def caption(example, page):
    row, a = example['row'], example['actual']
    if page == 'contrast':
        return ('M = %.3f | cone fraction = %.1f%% | area B>C = %.1f%%' %
                (a['M'], 100*row['complexity_regions']['cone_annulus']['cone_fraction'],
                 100*row['complexity_regions']['cone_annulus']['other_dominant_area_fraction']))
    if page == 'patchiness':
        return 'P_C = %.3f | P_B = %.3f | Gaussian sigma = 0.2 arcsec' % (a['P_C'], a['P_B'])
    if page == 'dust':
        return 'Mean tau = %.3f | mean T = %.3f | covered area = %.1f%%' % (a['tau'], a['T'], 100*a['coverage'])
    return ('Median C/sigma = %.2f | extent / PSF = %.1f | optical PSF = %.2f pixels' %
            (measured_value(row, 'snr'), row['extent_over_optical_psf_fwhm'],
             row['optical_psf_core_fwhm_arcsec']/example['pixel_scale']))


def atlas_page(rows, examples, keys, page, title, explanation, norm, stride, label):
    n = len(rows)
    fig = plt.figure(figsize=(6.0*(len(keys)+1), 4.6*n+1.5))
    grid = fig.add_gridspec(n, len(keys)+1, left=.025, right=.98, bottom=.10, top=.88,
                           wspace=.10, hspace=.38)
    limits = {}
    for key in keys:
        fixed = FIELDS[key][3]
        maximum = max(float(np.nanmax(examples[r['id']]['fields'][key][examples[r['id']]['aperture']])) for r in rows)
        limits[key] = fixed or (0., max(maximum*1.04, .1))
    for i, row in enumerate(rows):
        example = examples[row['id']]
        for j, key in enumerate(keys):
            surface_panel(fig.add_subplot(grid[i, j], projection='3d'), example, key, limits[key], stride)
        ax = fig.add_subplot(grid[i, -1])
        image_panel(ax, example, norm,
            example['row']['id']+' | synthetic noisy input\n'+caption(example, page))
    fig.suptitle(title+'\n'+label, fontsize=15, y=.975)
    fig.text(.5, .935, explanation, ha='center', va='top', fontsize=10)
    fig.text(.5, .024, 'Cyan: valid nominal cone at 0.3 < r < 3.3 arcsec. Images share an asinh brightness scale; east left, north up.\n'
             'Surfaces sampled every %d pixels for display; annotations and checks use every valid pixel.' % stride,
             ha='center', fontsize=9)
    return fig


def assemble_pdf(output, pages):
    """Package already rendered PNGs, avoiding expensive 3D vector re-rendering.

    Pages are raster figures at their original dimensions (135 dpi), with
    scientific source fields and full-resolution checks recorded separately.
    Commit the PDF name only after a complete document has been closed.
    """
    temporary = output/'atlas.pending.pdf'
    with PdfPages(temporary) as pdf:
        for name, _ in pages:
            with Image.open(output/(name+'.png')) as image:
                pixels = np.asarray(image.convert('RGB'))
            height, width = pixels.shape[:2]
            fig = plt.figure(figsize=(width/135, height/135))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(pixels); ax.set_axis_off()
            pdf.savefig(fig, dpi=135)
            plt.close(fig)
            del pixels, fig, ax
            gc.collect()
    temporary.rename(output/'f187n_3d_descriptor_atlas.pdf')


def space_page(rows, gallery, examples, norm, label, resolution=False):
    fig = plt.figure(figsize=(15, 11))
    grid = fig.add_gridspec(len(gallery), 2, width_ratios=(1.7, 1), top=.88, bottom=.08,
                           left=.035, right=.97, hspace=.35, wspace=.15)
    ax = fig.add_subplot(grid[:, 0], projection='3d')
    for row in rows:
        if resolution:
            xyz = [row['extent_over_optical_psf_fwhm'], measured_value(row, 'snr'), patch_value(row, 'cone')]
        else:
            xyz = [measured_value(row, 'M'), patch_value(row, 'other'), measured_value(row, 'T')]
        if any(v is None or not np.isfinite(v) for v in xyz):
            continue
        number = next((i+1 for i, selected in enumerate(gallery) if selected['id'] == row['id']), None)
        ax.scatter(*xyz, s=80 if number else 22, color='#d55e00' if number else '#0072b2', alpha=.85)
        if number:
            ax.text(*xyz, '  '+str(number), fontsize=12)
    if resolution:
        ax.set(xlabel='Projected extent / optical PSF FWHM', ylabel='Median cone / pixel sigma',
               zlabel='Cone structure P_C (sigma=0.2 arcsec)')
        title = 'Resolution and signal visibility in 3D'
    else:
        ax.set(xlabel='Other / cone light M', ylabel='Other-light structure P_B (sigma=0.2 arcsec)',
               zlabel='Area-mean foreground transmission T', zlim=(0, 1))
        title = 'Measured image-descriptor space in 3D'
    ax.view_init(elev=26, azim=-55)
    for i, row in enumerate(gallery):
        image_panel(fig.add_subplot(grid[i, 1]), examples[row['id']], norm, str(i+1)+': '+row['id'])
    fig.suptitle(title+'\n'+label, fontsize=16, y=.97)
    fig.text(.5, .91, 'Each point is one full positive synthetic realization. Numbered points match the images. No fitted response surface.',
             ha='center', fontsize=10)
    fig.text(.5, .025, 'Descriptors are separate measurements, not a combined difficulty score. Cone-free examples have undefined M/extent and are excluded.',
             ha='center', fontsize=9)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--examples', type=int, choices=(1, 2, 3, 4), default=3)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--label', default=None)
    args = parser.parse_args()
    if args.stride < 1 or args.stride > 16:
        parser.error('Surface sampling stride must be 1..16 pixels')
    root = args.dataset.resolve()
    manifest = validate_manifest(root)
    rows = [r for r in manifest['samples'] if r['variant'] == 'full' and r.get('cone_present', True)]
    groups = len({r['group'] for r in rows})
    label = args.label or ('F187N synthetic examples | %d independent groups | descriptive diagnostics' % groups)
    selections = {key: representative_rows(rows, key, args.examples) for key in ('M', 'P_B', 'T', 'snr')}
    chosen = {r['id']: r for selected in selections.values() for r in selected}
    examples = {name: load_example(root, row, manifest) for name, row in chosen.items()}
    for example in examples.values():
        example['pixel_scale'] = manifest['pixel_scale_arcsec']
    values = np.concatenate([e['observed'][e['valid']] for e in examples.values()])
    top = float(np.percentile(values, 99.5))
    bottom = min(0., float(np.percentile(values, .1)))
    norm = AsinhNorm(linear_width=max(top*.03, 1e-12), vmin=bottom, vmax=top)
    args.output.mkdir(parents=True, exist_ok=False)
    pages = []
    specs = [
        ('01_contrast', 'M', ['fraction'], 'contrast', 'Competing light: a spatial field and its summed contrast',
         'Height = B/(C+B). M = sum(B)/sum(C) is annotated separately; M is not the average of the local fractions.'),
        ('02_patchiness', 'P_B', ['cone_residual', 'other_residual'], 'patchiness', 'Spatial structure of cone and other light',
         'R_I = |I - Gaussian_sigma(I)| / mean_A(I); P_I = mean_A(R_I). Heights use log10(1+R_I). Smooth before masking.'),
        ('03_dust', 'T', ['tau', 'transmission'], 'dust', 'Foreground dust: optical depth, transmission and coverage',
         'Pre-PSF screen maps: mean(tau) and mean(exp(-tau)) within each output pixel. Coverage is annotated; no dust mass inference.'),
        ('04_visibility', 'snr', ['signal_sigma'], 'visibility', 'Cone signal visibility and spatial resolution',
         'Height = log10(1+C/pixel_sigma). The reported median is per pixel; it is not an integrated detection significance.'),
    ]
    pdf_path = args.output/'f187n_3d_descriptor_atlas.pdf'
    for name, selector, keys, page, title, explanation in specs:
        fig = atlas_page(selections[selector], examples, keys, page, title, explanation, norm, args.stride, label)
        fig.savefig(args.output/(name+'.png'), dpi=135)
        plt.close(fig)
        del fig
        gc.collect()
        pages.append((name, title))
        print('Rendered '+name, flush=True)
    for resolution, name in [(False, '05_descriptor_space'), (True, '06_resolution_space')]:
        fig = space_page(rows, selections['M'], examples, norm, label, resolution=resolution)
        fig.savefig(args.output/(name+'.png'), dpi=135)
        plt.close(fig)
        del fig
        gc.collect()
        pages.append((name, 'Resolution and visibility space' if resolution else 'Image-descriptor space'))
        print('Rendered '+name, flush=True)
    assemble_pdf(args.output, pages)
    write_json(args.output/'atlas.json', {
        'schema': 'f187n_3d_descriptor_atlas_v1', 'label': label, 'source_dataset': str(root),
        'manifest_sha256': sha256(root/'manifest.json'), 'groups': groups,
        'selection': {k: [r['id'] for r in v] for k, v in selections.items()},
        'selection_rule': 'low, middle and high measured rank per panel; examples are not matched single-factor experiments',
        'aperture': 'reference-valid nominal mask, 0.3<r<3.3 arcsec; baseline and noise excluded from C/B and patchiness',
        'smoothing_sigma_arcsec': .2, 'surface_stride_pixels': args.stride,
        'pdf_export': 'PNG figures assembled as raster PDF pages at 135 dpi; raw measurements use full-resolution arrays',
        'display': {'normalization': 'shared asinh, display only', 'vmin': bottom, 'vmax': top,
                    'vmax_rule': '99.5 percentile across selected input images; brighter display pixels saturate',
                    'orientation': 'sky matrix from manifest; east left/north up in 2D; central +/-3.6 arcsec shown'},
        'examples': {key: {'source_archive': e['row']['source_archive'],
            'source_sha256': e['row']['source_sha256'], 'recomputed_measurements': e['actual'],
            'saved_measurements_verified': True, 'component_checks': e['checks']} for key, e in examples.items()},
        'limitations': 'Example images and descriptors only; no U-Net accuracy inference, fitted population function, or universal difficulty scalar. Dust fields are pre-PSF screen area averages; component/SNR maps are post-PSF.',
        'code_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__), ROOT/'src/simple_utils/image_complexity.py']},
        'artifact_sha256': {p.name: sha256(p) for p in args.output.iterdir() if p.is_file()}})
    links = ''.join('<section><h2>'+html.escape(title)+'</h2><a href="'+name+'.png"><img src="'+name+'.png" alt="'+html.escape(title)+'"></a></section>' for name, title in pages)
    (args.output/'index.html').write_text('<!doctype html><html><meta charset="utf-8"><title>F187N 3D descriptor atlas</title>'
        '<style>body{font:16px system-ui;max-width:1500px;margin:35px auto;padding:0 22px;background:#f4f6f8;color:#192330}img{width:100%;background:white}section{margin:35px 0}a{color:#145b92}</style>'
        '<h1>'+html.escape(label)+'</h1><p><a href="f187n_3d_descriptor_atlas.pdf">Download the six-page PDF</a>. '
        'Static scientific 3D views with matching synthetic images. Cyan outlines show the measurement aperture. '
        'These examples are descriptive measurements, not a calibrated difficulty score or evidence of U-Net accuracy.</p>'+links+'</html>')
    print('Saved six-page 3D atlas:', pdf_path)


if __name__ == '__main__':
    main()
