"""Slide-ready F187N spectral comparison: synthetic galaxy and NGC 1068 model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/f187n_galaxy_agn_plot')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.jwst_utils.nircam import load_filters
from src.simple_utils.wavelengths import cloudy_air_to_vacuum

Z = 0.003793
R = 3000.
GAL = ROOT / 'results/cloudy/f187n_spectral_reference_v2/spectra.npz'
GRID = ROOT / 'data/cloudy/ngc1068_agn_spectral_pilot_v1/grid.npz'
AGN_FIT = ROOT / 'results/cloudy/ngc1068_spectral_constraints_v1/fit.json'
AGN_CONT = ROOT / 'results/cloudy/ngc1068_spectral_constraints_v1/modeled_continuum.npz'
LINE_GUIDES = [
    ('He II 1.864', ('he2_18636',), 18636.5, '#72529a', .87, -.0030),
    ('He I 1.869–1.870', ('he1_18685', 'he1_18697'), 18685.4, '#238467', .74, -.0018),
    ('Pa α + He II blend', ('pa_alpha', 'he2_18743', 'he2_18757'), 18751., '#654d71', .82, .0007),
]


def smooth(wave, flux):
    dlog = np.diff(np.log(wave)).mean()
    grid = np.exp(np.linspace(np.log(wave[0]), np.log(wave[-1]), len(wave)))
    values = np.interp(grid, wave, flux)
    blurred = gaussian_filter1d(values, 1. / (2.35482 * R * dlog), mode='nearest')
    return np.interp(wave, grid, blurred)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'results/cloudy/f187n_galaxy_vs_agn_v5')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f'Use a fresh output folder: {out}')

    galaxy_data = np.load(GAL)
    grid_data = np.load(GRID)
    fit = json.loads(AGN_FIT.read_text())
    best = fit['best_line_scenario']
    index = int(best['model_index'])
    model_id = str(grid_data['model_id'][index])
    names = [str(name) for name in grid_data['line_names']]
    red_vac = cloudy_air_to_vacuum(grid_data['rest_line_angstrom'])

    # Use the published NGC 1068 nuclear line fit. Pa-alpha is predicted by
    # CLOUDY because the source paper reports it as unmeasurable.
    line_scale = float(best['scale_in_table_flux_units_per_cloudy_line_unit']) * 1e-13
    tau = float(best['tau_at_pa_alpha'])
    ref = 18756.119850349412
    line_flux = grid_data['line_flux'][index] * line_scale * np.exp(
        -tau * (red_vac / ref) ** (-1.3))

    galaxy_wave = galaxy_data['observed_vacuum_angstrom']
    selected = (galaxy_wave >= 18580.) & (galaxy_wave <= 19050.)
    wave = galaxy_wave[selected]
    galaxy_cont = galaxy_data['synthetic_continuum'][selected]
    galaxy = galaxy_data['synthetic_relative_flambda'][selected]

    # Continuum is the existing phenomenological fit to local continuum
    # proxies from published IRTF/SpeX nuclear measurements, not raw pixels.
    cont_data = np.load(AGN_CONT)
    agn_cont = np.interp(wave / (1 + Z), cont_data['rest_angstrom'],
                         cont_data['modeled_flambda']) / (1 + Z)
    agn_lines = np.zeros_like(wave)
    line_records = []
    for label, keys, air, color, *_ in LINE_GUIDES:
        for key in keys:
            if key not in names:
                raise ValueError(f'Missing CLOUDY transition {key}')
            i = names.index(key)
            center = float(red_vac[i]) * (1 + Z)
            flux = float(line_flux[i])
            sigma = center * 85. / 299792.458
            if np.isfinite(flux):
                agn_lines += flux * np.exp(-.5 * ((wave - center) / sigma) ** 2) / (
                    np.sqrt(2 * np.pi) * sigma)
            line_records.append({
                'label': label, 'cloudy_id': key, 'rest_air_angstrom': air,
                'rest_vacuum_angstrom': float(red_vac[i]),
                'observed_vacuum_um': center / 1e4,
                'integrated_flux_erg_s_cm2': flux,
                'status': 'CLOUDY line; Pa alpha is predicted, not measured'
                if key == 'pa_alpha' else 'CLOUDY line in published-line-constrained model',
            })
    agn = agn_cont + agn_lines
    galaxy, galaxy_cont = smooth(wave, galaxy), smooth(wave, galaxy_cont)
    agn, agn_cont = smooth(wave, agn), smooth(wave, agn_cont)
    norm_g = float(np.interp(18740., wave, galaxy_cont))
    norm_a = float(np.interp(18740., wave, agn_cont))
    galaxy, galaxy_cont = galaxy / norm_g, galaxy_cont / norm_g
    agn, agn_cont = agn / norm_a, agn_cont / norm_a

    filter_meta = next(row for row in load_filters().values() if row['name'] == 'F187N')
    blue, red = filter_meta['half_power_blue_um'], filter_meta['half_power_red_um']
    fw, throughput = filter_meta['curve'].T
    throughput = throughput / throughput.max()
    out.mkdir(parents=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=.075, right=.99, top=.77, bottom=.16, wspace=.11)
    colors = ['#007fa3', '#a85828']
    panels = [
        (axes[0], galaxy, galaxy_cont, 'Synthetic galaxy', colors[0]),
        (axes[1], agn, agn_cont, 'NGC 1068 nuclear model', colors[1]),
    ]
    ymax = 1.16 * max(float(galaxy.max()), float(agn.max()))
    for ax, spectrum, continuum, title, color in panels:
        ax.axvspan(blue, red, color='#7589a2', alpha=.13, zorder=0)
        ax.plot(wave / 1e4, continuum, color='#555555', lw=1.45,
                ls=(0, (4, 3)), zorder=2)
        ax.plot(wave / 1e4, spectrum, color=color, lw=2.5, zorder=3)
        for label, keys, air, line_color, ypos, xoff in LINE_GUIDES:
            centers = [float(red_vac[names.index(key)]) * (1 + Z) / 1e4 for key in keys]
            x = float(np.mean(centers))
            ax.axvline(x, color=line_color, alpha=.23, lw=1.1, zorder=1)
            display_label = label.replace(' + ', ' / ').replace('1.869–1.870', '1.869–70')
            ax.annotate(display_label, (x, ymax * .035), (x + xoff, ymax * ypos),
                        ha='center', va='center', fontsize=12, color=line_color,
                        arrowprops={'arrowstyle': '-', 'color': line_color,
                                    'lw': 1.1, 'shrinkA': 0, 'shrinkB': 2})
        ax.axvline(blue, color='#38495b', lw=1.25, ls='--')
        ax.axvline(red, color='#38495b', lw=1.25, ls='--')
        ax.set_title(title, fontsize=22, pad=16, weight='semibold')
        ax.set_xlim(1.858, 1.900)
        ax.set_ylim(0, ymax)
        ax.grid(axis='y', alpha=.16)
        ax.tick_params(labelsize=14)
        ax.text(blue, ymax * .015, '1.863', ha='center', va='bottom', fontsize=12,
                color='#38495b')
        ax.text(red, ymax * .015, '1.885', ha='center', va='bottom', fontsize=12,
                color='#38495b')
    axes[0].set_ylabel(r'Relative $F_\lambda$  (continuum = 1)', fontsize=17)
    for ax in axes:
        ax.set_xlabel('Observed wavelength (μm)', fontsize=17, labelpad=12)
    handles = [
        plt.Line2D([], [], color='#7589a2', lw=8, alpha=.3, label='F187N half-power band'),
        plt.Line2D([], [], color='#555555', lw=1.5, ls=(0, (4, 3)), label='Continuum'),
        plt.Line2D([], [], color='#414141', lw=2.5, label='Continuum + emission lines'),
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .91),
               ncol=3, frameon=False, fontsize=13)
    fig.suptitle('F187N spectral comparison', fontsize=25, weight='semibold', y=.985)
    fig.text(.5, .045,
             'AGN: published NGC 1068 near-IR constraints + CLOUDY; Pa α predicted',
             ha='center', fontsize=12, color='#414141')
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(out / f'f187n_galaxy_vs_agn.{ext}', dpi=240)
    plt.close(fig)

    np.savez_compressed(out / 'spectra.npz', wave_angstrom=wave,
                        galaxy=galaxy, galaxy_continuum=galaxy_cont,
                        agn=agn, agn_continuum=agn_cont,
                        filter_wave_angstrom=fw, filter_throughput=throughput)
    (out / 'important_lines.json').write_text(json.dumps(line_records, indent=2) + '\n')
    report = {
        'status': 'Slide-ready shape comparison; NGC 1068 AGN scenario is model-constrained, not an observed spectrum',
        'agn_model_id': model_id, 'agn_grid_index': index,
        'line_constraint_fit': best,
        'continuum_constraint_fit': fit['best_continuum_scenario'],
        'filter': {'name': 'F187N', 'pivot_um': filter_meta['pivot_um'],
                   'blue_half_power_um': blue, 'red_half_power_um': red,
                   'source_url': filter_meta['source_url']},
        'redshift': Z, 'display_resolution_R': R,
        'assumed_intrinsic_line_sigma_kms': 85.,
        'continuum_normalization_um': 1.874,
        'limitations': [
            'AGN constraints are from ground-based IRTF/SpeX published measurements, not JWST spectroscopy.',
            'Continuum is inferred from published flux/equivalent-width points, not sampled spectral pixels.',
            'Pa alpha was not reliably measured in the source and is a CLOUDY prediction.',
            'Single-zone CLOUDY fit and phenomenological continuum are a first-pass model, not a posterior or validated spectrum.',
            'The two curves are independently continuum-normalized; their amplitudes are not comparable.',
            'R=3000 and 85 km/s intrinsic width are display assumptions; F187N is an imaging bandpass.',
        ],
        'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (GAL, GRID, AGN_FIT, AGN_CONT, Path(__file__))},
    }
    (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    (out / 'README.md').write_text(f'''# F187N spectral comparison\n\nThe PNG/PDF/SVG are the slide figure. The two spectra are independently normalized to their continua at 1.874 µm; vertical amplitudes cannot be compared. The blue-gray band marks the official F187N half-power range, {blue:.3f}–{red:.3f} µm, on the observed-wavelength axis.\n\nThe NGC 1068 nuclear example is more constrained than the previous illustrative template: CLOUDY line ratios are scaled to published nuclear near-IR line constraints, with a phenomenological continuum fit to local continuum proxies. The measurements are ground-based IRTF/SpeX data from 2007, not JWST spectra. Pa α was compromised by telluric absorption/order coverage in that study, so its plotted strength is a CLOUDY prediction. The source data audit is `data/spectra/ngc1068_martins2010_v1/manifest.json`; source paper: [Martins et al. 2010](https://academic.oup.com/mnras/article/406/4/2168/1017118). The continuum proxy fit has unknown covariance and is descriptive, not a formal likelihood.\n\nThe line-free continuum model and selected CLOUDY transitions are shown at display R={R:g}; intrinsic line sigma is assumed to be 85 km/s. F187N is an imaging filter, so this plotted spectrum illustrates which features contribute within its passband; it does not represent a resolved JWST spectrum. The filter pivot and half-power limits ({blue:.3f}, {red:.3f} µm) follow [STScI NIRCam filter documentation]({filter_meta['source_url']}).\n''')
    print(json.dumps({'output': str(out), 'model': model_id,
                      'agn_log_u': float(grid_data['log_u'][index]),
                      'agn_tau': tau, 'filter_half_power_um': [blue, red],
                      'agn_peak': float(agn.max()), 'galaxy_peak': float(galaxy.max())}, indent=2))


if __name__ == '__main__':
    main()
