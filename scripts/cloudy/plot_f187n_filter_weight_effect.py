"""Compare a flat F187N-window image with the actual throughput-weighted image."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/f187n-weight-effect-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.jwst_utils.nircam import convolve_kernel, load_filters
from src.simple_utils.f187n import line_response, rebin_surface_brightness
from src.simple_utils.nuclear_sed import nuclear_flambda_shape
from src.simple_utils.synthetic_photometry import photon_bandpass_weights
from src.simple_utils.wavelengths import cloudy_air_to_vacuum

EXPORT = ROOT / 'results/cloudy/ngc1068_f187n_spectral_export_v1'
FILTERED = ROOT / 'results/cloudy/ngc1068_f187n_soft_gas_v1/image_mask.npz'


def render_response(state, cfg, agn, hii, fw, response, blue, red):
    """Stream the saved spatial spectral model into one photon-weighted image."""
    z = cfg['redshift']
    over = cfg['oversample']
    tau = state['tau']
    exponent = cfg['components']['dust_exponent']
    populations = (
        ('cone', agn, state['cone_weight'], state['cone_index'], state['velocity'],
         cfg['velocity']['sigma_kms'], False),
        ('host', agn, state['host_weight'], state['host_index'], state['host_velocity'],
         cfg['velocity']['sigma_kms'], False),
        ('starforming', hii, state['hii_spatial'] * float(state['hii_normalization']),
         np.zeros_like(state['cone_index']), state['hii_velocity'],
         cfg['star_forming']['sigma_kms'], True),
    )
    photon_weights = photon_bandpass_weights(fw, fw, response)
    map_fine = np.zeros_like(tau)

    for pop, grid, spatial, model_index, velocity, sigma, is_hii in populations:
        flux = np.nan_to_num(grid['line_flux'], nan=0.)
        for j, stored_wave in enumerate(grid['rest_line_angstrom']):
            rest = float(stored_wave) if is_hii else float(stored_wave)
            if not is_hii and cfg.get('convert_cached_air_lines_to_vacuum', False):
                rest = float(cloudy_air_to_vacuum(rest))
            if response.max() <= 0:
                continue
            # Match the renderer: precompute on its velocity grid, then interpolate
            # the response for every source pixel.
            vel_grid = np.linspace(-400, 400, 81) if is_hii else np.linspace(-1200, 1200, 241)
            resp = np.array([line_response(rest, z, v, sigma, fw, response)
                             for v in vel_grid])
            local_response = np.interp(velocity, vel_grid, resp)
            # Dust uses observed center; for H II and gas this is the same frame convention.
            local_center = rest * (1 + z) * (1 + velocity / 299792.458)
            screen = np.exp(-tau * (local_center / (18751. * (1 + z))) ** (-exponent))
            map_fine += spatial * flux[model_index, j] * local_response * screen

        # Continuum response: integrate the full sampled F_lambda spectrum.
        continuum = grid['continuum_rest_flambda']
        for j, wave in enumerate(fw):
            weight = photon_weights[j]
            if weight == 0:
                continue
            values = continuum[model_index, j] / (1 + z)
            screen = np.exp(-tau * (wave / (18751. * (1 + z))) ** (-exponent))
            map_fine += weight * spatial * values * screen

    # Stellar continuum is flat in F_lambda in the existing model.
    flat_weights = photon_bandpass_weights(fw, fw, response)
    actual_filter = load_filters()['F187N']['curve']
    actual_weights = photon_bandpass_weights(actual_filter[:, 0], actual_filter[:, 0], actual_filter[:, 1])
    # The filter curve and model grids are sampled identically in this export.
    band_ratio = float(flat_weights.sum() / actual_weights.sum())
    map_fine += state['stellar_band'] * band_ratio

    # Apply the same explicit nuclear continuum shape to a changed response.
    nwave, transmission = load_filters()['F187N']['curve'].T
    nuclear_shape = nuclear_flambda_shape(nwave / (1 + z), cfg['nuclear_sed']) / (1 + z)
    unweighted_nuclear = photon_bandpass_weights(fw, fw, response) @ np.interp(fw, nwave, nuclear_shape)
    weighted_nuclear = photon_bandpass_weights(fw, fw, np.interp(fw, nwave, transmission)) @ np.interp(fw, nwave, nuclear_shape)
    map_fine += state['nuclear_band'] * float(unweighted_nuclear / weighted_nuclear)

    detector = rebin_surface_brightness(convolve_kernel(map_fine, state['kernel']), over)
    return detector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'results/cloudy/ngc1068_f187n_filter_weight_effect_v1')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f'Use a fresh output folder: {out}')

    state_file = np.load(EXPORT / 'spectral_state.npz')
    state = {key: state_file[key] for key in state_file.files}
    cfg = json.loads(str(state['configuration']))
    with np.load(cfg['grid']) as f:
        agn = {key: f[key] for key in f.files}
    with np.load(cfg['star_forming']['grid']) as f:
        hii = {key: f[key] for key in f.files}
    with np.load(FILTERED) as f:
        saved_after = f['relative_clean']
        reference_valid = f['reference_valid']
        reference_meta = json.loads(str(f['metadata']))

    filt = load_filters()['F187N']
    fw, actual_t = filt['curve'].T
    blue, red = filt['half_power_blue_um'] * 1e4, filt['half_power_red_um'] * 1e4
    # Flat response over the documented half-power interval, matched to the
    # mean system throughput there so the map difference shows spectral shape
    # weighting rather than the filter's overall throughput loss.
    in_band = (fw >= blue) & (fw <= red)
    top_hat = in_band.astype(float) * float(np.mean(actual_t[in_band]))
    before = render_response(state, cfg, agn, hii, fw, top_hat, blue, red)
    after = render_response(state, cfg, agn, hii, fw, actual_t, blue, red)
    amplitude = float(reference_meta['empirical_scale'])
    before *= amplitude
    after *= amplitude
    saved_reference = saved_after * amplitude
    saved_check = float(np.max(np.abs(after - saved_reference)) /
                        max(np.max(np.abs(saved_reference)), 1e-30))
    # Reference image is already filter-weighted; independently rebuilt map should agree.
    if saved_check > 0.03:
        raise ValueError(f'Rebuilt F187N image differs from saved filtered image by {saved_check:.3%}')

    lo = min(float(np.percentile(before[reference_valid], 1)),
             float(np.percentile(after[reference_valid], 1)))
    hi = max(float(np.percentile(before[reference_valid], 99.7)),
             float(np.percentile(after[reference_valid], 99.7)))
    norm = AsinhNorm(linear_width=max((hi - lo) / 35., 1e-12), vmin=lo, vmax=hi)
    cmap = 'magma'
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2), constrained_layout=True)
    for ax, image, title in zip(
            axes, (before, after),
            ('Flat response across F187N band', 'F187N throughput-weighted')):
        shown = ax.imshow(image, origin='lower', cmap=cmap, norm=norm,
                          interpolation='nearest')
        ax.set_title(title, fontsize=19, weight='semibold', pad=12)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect('equal')
    colorbar = fig.colorbar(shown, ax=axes, shrink=.83, pad=.025)
    colorbar.set_label('Relative photon-weighted surface brightness', fontsize=13)
    fig.suptitle('Same synthetic galaxy, with and without F187N throughput weighting',
                 fontsize=22, weight='semibold')
    out.mkdir(parents=True)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(out / f'f187n_filter_weight_effect.{ext}', dpi=240)
    plt.close(fig)
    np.savez_compressed(out / 'comparison_images.npz', flat_response=before,
                        f187n_weighted=after, valid=reference_valid,
                        filter_wave_angstrom=fw, throughput=actual_t,
                        filter_half_power_angstrom=[blue, red])
    report = {
    'comparison': 'flat response over F187N half-power interval, normalized to its mean transmission, versus actual F187N throughput',
        'half_power_um': [blue / 1e4, red / 1e4],
        'pivot_um': filt['pivot_um'],
        'same_galaxy_components_psf_pixel_grid_and_empirical_amplitude': True,
        'f187n_reconstruction_max_relative_difference_from_saved_image': saved_check,
        'units': 'relative photon-weighted surface brightness; same scale for both panels',
        'filter_source': filt['source_url'],
    'caveat': 'Flat-response panel is a diagnostic top-hat comparison normalized to the mean F187N transmission in the half-power interval, not an unfiltered broadband observation.',
    }
    (out / 'README.md').write_text(f'''# F187N throughput weighting comparison\n\nLeft: the same synthetic spatial model integrated with a flat response from {blue/1e4:.3f} to {red/1e4:.3f} µm. Its response level is normalized to the mean F187N transmission inside that interval, so the two panels share a meaningful intensity scale. Right: integrated with the actual wavelength-dependent NIRCam F187N throughput curve. Components, PSF, output pixel grid, and empirical model scale are held fixed.\n\nThis isolates the effect of wavelength-dependent filter weighting. The left panel is a diagnostic top-hat comparison, not a broadband image or a detector-level pre-filter exposure. The existing synthetic model is already F187N-specific and empirically scaled. The passband half-power limits are from [STScI NIRCam filter documentation]({filt['source_url']}).\n\nThe right-hand reconstruction agrees with the saved F187N image to {saved_check:.3%} maximum relative difference.\n''')
    print(json.dumps({'output': str(out), **report}, indent=2))


if __name__ == '__main__':
    main()
