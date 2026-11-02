#!/usr/bin/env python3
"""Characterize saved F187N component maps without rerendering or training."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/agn-complexity-matplotlib')
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.simple_utils.image_complexity import component_metrics, psf_axis_fwhm


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_components(path):
    """Require a reconstructable F187N archive; ignore redundant aggregate maps."""
    with np.load(path, allow_pickle=False) as saved:
        meta = json.loads(str(saved['metadata']))
        if meta['filter'] != 'F187N':
            raise ValueError(f'{path}: this adapter supports F187N only')
        names = meta['component_names']
        if not names or len(set(names)) != len(names) or set(names) & set(meta.get('aggregate_names', [])):
            raise ValueError('Ambiguous or duplicate component membership')
        clean = np.array(saved['clean'], dtype=float)
        layers = {name: np.array(saved[name], dtype=float) for name in names}
        if clean.ndim != 2 or not np.isfinite(clean).all():
            raise ValueError('Expected finite 2D clean image')
        for name, layer in layers.items():
            if layer.shape != clean.shape or not np.isfinite(layer).all():
                raise ValueError(f'Invalid component {name}')
        baseline = float(meta['galaxy_plus_background_baseline_mjysr'])
        reconstruction = sum(layers.values()) + baseline
        error = float(np.max(np.abs(clean - reconstruction)))
        if not np.isfinite(baseline) or not np.allclose(clean, reconstruction, rtol=1e-10, atol=1e-10):
            raise ValueError('Saved components plus baseline do not reconstruct clean image')
        clipped = {}
        for name, layer in layers.items():
            # FFT convolution can leave roundoff-level negative pixels in faint tails.
            tolerance = 1e-12 * max(float(np.max(np.abs(layer))), np.finfo(float).tiny)
            if np.any(layer < -tolerance):
                raise ValueError(f'Material negative component values in {name}')
            negative = layer < 0
            clipped[name] = {'pixels': int(negative.sum()),
                             'absolute_pixel_sum': float(-layer[negative].sum())}
            layer[negative] = 0.
        cone_names = [n for n in names if n.startswith('cone_')]
        other_names = [n for n in names if n not in cone_names]
        cone = sum((layers[n] for n in cone_names), np.zeros_like(clean))
        other = sum((layers[n] for n in other_names), np.zeros_like(clean))
        if not np.allclose(clean, cone + other + baseline, rtol=1e-10, atol=1e-10):
            raise ValueError('Roundoff cleanup changed reconstruction materially')
        mask = np.array(saved['mask'])
        if mask.shape != clean.shape or not np.isin(mask, [0, 1]).all():
            raise ValueError('Expected matching binary nominal mask')
        valid = np.array(saved['reference_valid'], dtype=bool)
        noise = np.array(saved['reference_error'], dtype=float)
        observed = np.array(saved['input'], dtype=float)
        if any(x.shape != clean.shape for x in [valid, noise, observed]):
            raise ValueError('Saved reference/image shapes differ')
    return meta, cone, other, mask.astype(bool), valid, noise, observed, {
        'cone_components': cone_names, 'other_components': other_names,
        'baseline_excluded': baseline, 'reconstruction_max_absolute_error': error,
        'roundoff_negative_clipping': clipped,
    }


def characterize(path, args):
    meta, cone, other, mask, valid, noise, observed, checks = load_components(path)
    scale = float(meta['pixel_scale_arcsec'])
    yy, xx = np.indices(cone.shape)
    center = [(cone.shape[1] - 1) / 2, (cone.shape[0] - 1) / 2]
    radius = np.hypot(xx - center[0], yy - center[1]) * scale
    annulus = (radius > args.inner_radius) & (radius < args.outer_radius)
    if args.valid_pixels == 'reference':
        annulus &= valid
    regions = {'cone_annulus': annulus & mask, 'whole_annulus': annulus}
    # In reference-valid pixels the renderer uses reference_error as sigma.
    # Do not infer fallback sigma values for invalid reference pixels.
    noise = np.where(valid, noise, np.nan)
    metrics = {name: component_metrics(cone, other, region,
               pixel_scale_arcsec=scale, smoothing_sigmas_arcsec=args.smoothing_sigmas,
               pixel_area=(scale / 206264.806247)**2, noise_sigma=noise)
               for name, region in regions.items()}
    psf_path = resolve(meta['psf'])
    with np.load(psf_path, allow_pickle=False) as psf:
        widths = psf_axis_fwhm(psf['kernel'], float(psf['pixel_scale_arcsec']))
    extent = float(radius[mask].max()) if mask.any() else None
    resolution = {
        'extent_definition': 'maximum center-to-mask-pixel-center radius; one-sided projected radial extent, not tip-to-tip',
        'projected_mask_extent_arcsec': extent,
        'mask_touches_stamp_edge': bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()),
        'psf_path': str(psf_path), 'psf_sha256': sha256(psf_path),
        'psf_core_fwhm': widths,
        'psf_width_definition': 'mean of interpolated x/y half-maximum widths through peak of saved optical kernel; excludes detector integration and drizzle',
        'extent_over_optical_psf_fwhm': extent / widths['mean_arcsec'] if extent is not None else None,
        'optical_psf_fwhm_in_image_pixels': widths['mean_arcsec'] / scale,
        'distance_mpc': args.distance_mpc, 'distance_source': args.distance_source,
        'projected_mask_extent_pc': (4.8481368 * args.distance_mpc * extent
                                     if args.distance_mpc is not None and extent is not None else None),
    }
    result = {
        'source': str(path), 'source_sha256': sha256(path),
        'source_status': meta.get('status'), 'source_limitations': meta.get('limitations', []),
        'configuration': meta['configuration_values'], 'grid': meta.get('grid'),
        'filter': meta['filter'], 'redshift': meta['redshift'], 'pixel_scale_arcsec': scale,
        'center_xy_pixels': center, 'center_definition': 'generator grid center, not fitted observed nucleus',
        'label_semantics': meta.get('cone_label_semantics', meta.get('label')),
        'component_checks': checks, 'resolution': resolution, 'regions': metrics,
    }
    return result, (observed, cone, other, mask, regions['cone_annulus'], scale)


def make_plots(output, results, maps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import AsinhNorm

    for i, (result, layers) in enumerate(zip(results, maps)):
        observed, cone, other, mask, region, scale = layers
        fraction = np.divide(cone, cone + other, out=np.zeros_like(cone), where=cone + other > 0)
        limit = max(float(np.percentile(np.concatenate([cone.ravel(), other.ravel()]), 99.5)), 1e-10)
        norm = AsinhNorm(linear_width=limit / 50, vmin=0, vmax=limit)
        extent = np.array([-cone.shape[1]/2, cone.shape[1]/2, -cone.shape[0]/2, cone.shape[0]/2]) * scale
        fig, axes = plt.subplots(1, 4, figsize=(15, 4.1), constrained_layout=True)
        for ax, layer, title in zip(axes, [observed, cone, other, fraction],
                                   ['Saved noisy input', 'Cone gas C', 'Other components B', 'Cone fraction C / (C + B)']):
            if ax is axes[0]:
                top = max(float(np.nanpercentile(observed, 99.5)), 1e-10)
                im = ax.imshow(layer, origin='lower', extent=extent, cmap='magma',
                               norm=AsinhNorm(linear_width=top / 50, vmin=0, vmax=top))
            elif ax is axes[3]:
                im = ax.imshow(layer, origin='lower', extent=extent, cmap='viridis', vmin=0, vmax=1)
            else:
                im = ax.imshow(layer, origin='lower', extent=extent, cmap='magma', norm=norm)
            if region.any() and not region.all():
                ax.contour(region, levels=[.5], colors='cyan', linewidths=.6, origin='lower', extent=extent)
            ax.set(title=title, xlabel='Image x offset (arcsec)', ylabel='Image y offset (arcsec)')
            fig.colorbar(im, ax=ax, shrink=.75, label='fraction' if ax is axes[3] else 'empirical MJy/sr')
        m = result['regions']['cone_annulus']
        ratio = m['other_over_cone']
        label = f'{ratio:.3f}' if ratio is not None else 'undefined'
        fig.suptitle(f"{Path(result['source']).parent.name} | M = {label} | cyan: measurement aperture\n"
                     'Component ratios exclude baseline and noise; C and B share a display scale')
        fig.savefig(output / f'sample_{i:02d}.png', dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for i, result in enumerate(results):
        m = result['regions']['cone_annulus']
        label = Path(result['source']).parent.name.replace('ngc1068_f187n_', '')
        if m['other_over_cone'] is not None:
            axes[0].barh(i, m['other_over_cone'])
            axes[0].text(m['other_over_cone'], i, f"  {m['other_over_cone']:.2f}", va='center')
        axes[1].plot([v['gaussian_sigma_arcsec'] for v in m['structure']],
                     [v['other'] for v in m['structure']], marker='o', label=label)
    axes[0].set_yticks(range(len(results)), [Path(r['source']).parent.name.replace('ngc1068_f187n_', '') for r in results])
    axes[0].set(xlabel='Other / cone light M (baseline excluded)', title='Competing light within cone aperture')
    axes[0].margins(x=.2)
    axes[1].set(xlabel='Gaussian smoothing sigma (arcsec)', ylabel='Normalized absolute residual',
                title='Structure in other components within same aperture')
    axes[1].legend(fontsize=8)
    fig.suptitle('F187N descriptive diagnostics — illustrative models, not a difficulty ranking')
    fig.savefig(output / 'comparison.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', nargs='+', type=Path, help='Saved image_mask.npz files')
    parser.add_argument('--output', type=Path, required=True, help='Fresh result directory')
    parser.add_argument('--inner-radius', type=float, default=.3, help='Strict inner radius in arcsec')
    parser.add_argument('--outer-radius', type=float, default=3.7, help='Strict outer radius in arcsec')
    parser.add_argument('--smoothing-sigmas', nargs='+', type=float, default=[.1, .2, .4])
    parser.add_argument('--valid-pixels', choices=['reference', 'all'], default='reference')
    parser.add_argument('--distance-mpc', type=float, help='Adopted angular-diameter distance; not inferred from z')
    parser.add_argument('--distance-source', help='Required provenance for an adopted distance')
    args = parser.parse_args()
    if not np.isfinite([args.inner_radius, args.outer_radius]).all() or not 0 <= args.inner_radius < args.outer_radius:
        parser.error('Require finite 0 <= inner radius < outer radius')
    if (args.distance_mpc is None) != (args.distance_source is None):
        parser.error('Supply both --distance-mpc and --distance-source, or neither')
    if args.distance_mpc is not None and (not np.isfinite(args.distance_mpc) or args.distance_mpc <= 0 or not args.distance_source.strip()):
        parser.error('Distance must be positive/finite and its provenance nonempty')
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    sources = [resolve(p).resolve() for p in args.sources]
    if len(set(sources)) != len(sources):
        parser.error('Duplicate source archives')
    results, maps = [], []
    for path in sources:
        result, layers = characterize(path, args)
        results.append(result)
        maps.append(layers)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        'schema': 'f187n_component_complexity_v1',
        'status': 'descriptive diagnostics; no scalar difficulty calibration, new rendering, or training',
        'aperture': {'inner_radius_arcsec_exclusive': args.inner_radius,
                     'outer_radius_arcsec_exclusive': args.outer_radius, 'valid_pixels': args.valid_pixels,
                     'regions': 'cone_annulus = nominal mask intersected with annulus; whole_annulus ignores mask'},
        'definitions': {
            'M': 'sum_A(B * pixel_solid_angle) / sum_A(C * pixel_solid_angle)',
            'weighted_sum_units': 'empirical MJy/sr times sr = empirical MJy; no absolute CLOUDY flux calibration',
            'patchiness': 'sum_A(abs(I - Gaussian_sigma(I)) * area) / sum_A(I * area); full-map smoothing, reflect boundary, truncate=4 sigma',
            'patchiness_limit': 'includes smooth gradients and cone boundaries; not pure cloud clumpiness or bounded by 1',
            'noise': 'median C/reference_error over reference-valid aperture pixels with positive finite ERR; no integrated significance or covariance independence assumed',
            'resolution': 'mask radial extent / saved optical kernel core FWHM; approximate, excludes detector integration and drizzle',
            'geometry_limit': 'mask geometry is prescribed, not inferred from these measurements',
        },
        'smoothing_sigmas_arcsec': args.smoothing_sigmas,
        'implementation_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in
                                  [Path(__file__).resolve(), ROOT/'src/simple_utils/image_complexity.py']},
        'samples': results,
    }
    (output / 'metrics.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    rows = []
    for i, result in enumerate(results):
        for region, m in result['regions'].items():
            row = {'sample': Path(result['source']).parent.name, 'source': result['source'],
                   'region': region, **{k: v for k, v in m.items() if k != 'structure'},
                   'extent_over_optical_psf_fwhm': result['resolution']['extent_over_optical_psf_fwhm']}
            for v in m['structure']:
                for component in ['cone', 'other']:
                    row[f'{component}_structure_sigma_{v["gaussian_sigma_arcsec"]:g}_arcsec'] = v[component]
            rows.append(row)
    with (output / 'metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    make_plots(output, results, maps)
    (output / 'README.md').write_text(
        '# F187N component complexity diagnostics\n\n'
        'Read `metrics.json` for definitions, provenance, aperture rules, limitations, and all values. '
        '`metrics.csv` has one row per sample and aperture. `comparison.png` compares contamination '
        'and spatial structure; `sample_XX.png` shows input and separate components.\n\n'
        'These are descriptive measurements of existing illustrative simulations, not real-galaxy '
        'decompositions, calibrated difficulty scores, or a new training population. The population '
        'v3 examples and soft-gas example use different component recipes, so their differences '
        'cannot be attributed to a single parameter. No existing image, mask, or spectrum changed.\n\n'
        'See `configs/cloudy/F187N_COMPLEXITY.md` in the repository for reproduction and interpretation.\n')
    print(f'Saved {len(results)} sample reports to {output}')
    for result in results:
        m = result['regions']['cone_annulus']
        print(Path(result['source']).parent.name, 'M=', m['other_over_cone'],
              'cone fraction=', m['cone_fraction'])


if __name__ == '__main__':
    main()
