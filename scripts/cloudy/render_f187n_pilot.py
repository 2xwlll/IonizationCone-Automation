"""Render a relative Cloudy F187N pilot with explicit components and a cone mask.

Wavelength integration streams over lines/continuum rather than allocating a
large spectral cube. This is a memory-limited pilot, not an observation fit.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import warnings

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ioncone-matplotlib')
import numpy as np
from scipy.ndimage import gaussian_filter, label

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.cloudy.generate_oiii_cube import nearest_models
from src.jwst_utils.nircam import convolve_kernel
from src.simple_utils.gas_structure import soft_agn_emission_weight, nuclear_spatial_profile
from src.simple_utils.wavelengths import cloudy_air_to_vacuum
from src.simple_utils.hii_population import render_hii_population
from src.simple_utils.nuclear_sed import nuclear_filter_response
from src.simple_utils.f187n_calibration import component_group
from src.simple_utils.f187n import line_response, rebin_surface_brightness, resample_psf
from src.simple_utils.synthetic_photometry import photon_bandpass_weights
from src.simple_utils.cone_geometry import resolve_cone_geometry, LEGACY
from src.simple_utils.foreground_dust import patchy_screen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/cloudy/ngc1068_f187n_pilot_v2.json'))
    parser.add_argument('--export-spectral-state', action='store_true',
                        help='Save pre-integration population state for reproducible spectral diagnostics')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    geometry_metadata = resolve_cone_geometry(cfg['geometry'])
    if geometry_metadata['projection'] == LEGACY:
        warnings.warn('Reproducing legacy approximate cone geometry; use circular_cone_envelope_v2 for new physical-angle experiments')
    output = Path(cfg['output'])
    if output.exists():
        raise FileExistsError('Choose a new pilot output directory')
    with np.load(cfg['grid'], allow_pickle=False) as f:
        grid = {name:f[name] for name in f.files}
    with np.load(cfg['reference'], allow_pickle=False) as f:
        real, error, valid = f['science'], f['error'], f['valid']
        reference = json.loads(f['metadata'].item())
    with np.load(cfg['psf'], allow_pickle=False) as f:
        kernel, kernel_scale = f['kernel'], float(f['pixel_scale_arcsec'])
    n, over = cfg['spatial_pixels'], cfg['oversample']
    if real.shape != (n,n):
        raise ValueError('Reference dimensions and requested output must agree')
    scale = reference['pixel_scale_arcsec']
    fine_scale = scale/over
    kernel = resample_psf(kernel, kernel_scale, fine_scale)
    rng = np.random.default_rng(cfg['seed'])
    high = n*over
    yy,xx = np.indices((high, high), dtype=float)
    x = (xx-(high-1)/2)/(high/2)
    y = (yy-(high-1)/2)/(high/2)
    matrix = np.asarray(reference['pixel_to_sky_matrix'])
    matrix /= np.sqrt(np.sum(matrix[:,0]**2))
    east = matrix[0,0]*x+matrix[0,1]*y
    north = matrix[1,0]*x+matrix[1,1]*y
    geo = cfg['geometry']
    pa = np.deg2rad(geo['position_angle_east_of_north_deg'])
    axial = east*np.sin(pa)+north*np.cos(pa)
    cross = east*np.cos(pa)-north*np.sin(pa)
    radius = np.hypot(x,y)
    half = geometry_metadata['projected_half_opening_rad']
    cone = (np.abs(cross) <= np.abs(axial)*np.tan(half)) & (radius < geo['radius_fraction'])
    cone |= radius < 3/high
    texture_cfg = cfg.get('texture_controls', {})
    cone_sigma = texture_cfg['cone_correlation_sigma_arcsec']/fine_scale if 'cone_correlation_sigma_arcsec' in texture_cfg else 2*over
    fields = [gaussian_filter(rng.normal(size=(high,high)), width) for width in (cone_sigma,8*over)]
    fields = [field/max(field.std(),1e-12) for field in fields]
    texture = np.exp(texture_cfg.get('cone_log_contrast', .8)*fields[0]+.35*fields[1])
    cone_weight = cone*np.exp(-2.5*radius)*(.3+texture)
    if cfg.get('gas_structure'):
        cone_weight=soft_agn_emission_weight(axial,cross,radius,half,
                    geo['radius_fraction'],texture,fields[1],high/2*fine_scale,
                    cfg['gas_structure'])
    disk_radius = np.hypot(east,north/.55)
    host_weight = cfg['components']['host_low_u_gas_fraction']*np.exp(-3*disk_radius)*np.exp(.4*fields[1])
    independent_host = cfg.get('host_geometry') is not None
    if independent_host:
        host_cfg = cfg['host_geometry']
        host_pa = np.deg2rad(host_cfg['position_angle_east_of_north_deg'])
        host_major = east*np.sin(host_pa)+north*np.cos(host_pa)
        host_minor = east*np.cos(host_pa)-north*np.sin(host_pa)
        disk_radius = np.hypot(host_major,host_minor/host_cfg['axis_ratio'])
        host_sigma = texture_cfg['host_correlation_sigma_arcsec']/fine_scale if 'host_correlation_sigma_arcsec' in texture_cfg else 2*over
        host_fields = [gaussian_filter(rng.normal(size=(high,high)), width) for width in (host_sigma,8*over)]
        host_fields = [field/max(field.std(),1e-12) for field in host_fields]
        host_weight = (cfg['components']['host_low_u_gas_fraction']*
                       np.exp(-host_cfg['radial_decay']*disk_radius)*
                       np.exp(texture_cfg.get('host_log_contrast', .6)*host_fields[0]+.4*host_fields[1]))
        # Independent projected population: no cone mask or exclusion here.
    else:
        # Preserve the original v1 recipe when explicitly using its config.
        host_weight *= ~cone
    density = np.clip(3+.5*fields[1],2,4)
    log_u = np.clip(-1.55-1.45*np.log10(1+8*radius),-3,-1.5)
    metallicity = np.clip(1.2-.3*radius,.7,1.5)
    dust = np.where(fields[1]>0,1.,.5)
    pah = np.where((dust==1)&(log_u < -2.2),.1,0)

    def indices(ionization, nh, model_dust=dust, model_pah=pah):
        result = np.empty((high,high), dtype=np.int32)
        # Avoid allocating an (image,image,81-model) distance cube in RAM.
        for start in range(0,high,24):
            rows = slice(start,start+24)
            result[rows] = nearest_models(grid,ionization[rows],nh[rows],metallicity[rows],model_dust[rows],model_pah[rows])
        return result

    cone_index = indices(log_u,density)
    host_index = indices(np.full_like(log_u,-3),np.clip(density-.5,2,4))
    velocity = np.sign(axial)*cfg['velocity']['maximum_kms']*np.sin(np.deg2rad(geo['inclination_deg']))*np.minimum(radius/.75,1)
    host_velocity = 150*east
    if independent_host:
        host_dust = np.where(host_fields[1]>0,1.,.5)
        host_pah = np.where(host_dust==1,.1,0)
        host_index = indices(np.full_like(log_u,host_cfg['log_u']),
                             np.clip(host_cfg['log_nh']+.4*host_fields[1],2,4),
                             host_dust,host_pah)
        host_velocity = host_cfg['rotation_kms']*host_major/np.sqrt(disk_radius**2+.1**2)
    if cfg.get('foreground_dust'):
        tau, dust_covered = patchy_screen((high,high),fine_scale,cfg['seed'],cfg['foreground_dust'])
        dust_model = 'patchy_screen_v1'
    else:
        tau = cfg['components']['foreground_tau_at_pa_alpha']*(fields[1]-fields[1].min())/np.ptp(fields[1])
        dust_covered = tau > 0
        dust_model = 'legacy_gas_correlated_screen'
    fw, throughput = grid['filter_wavelength_angstrom'], grid['throughput']
    z = cfg['redshift']
    populations = ('cone','host')
    gas_maps = {f'{population}_{kind}':np.zeros((high,high))
                for population in populations for kind in ('pa_alpha','other_lines','nebular_continuum')}

    vel_grid = np.linspace(-1200,1200,241)
    missing = np.isnan(grid['line_flux'])
    # Zero-filled censored lines provide a lower-bound contribution, recorded
    # explicitly. The archive retains NaN, never reclassifying missing as zero.
    line_flux = np.nan_to_num(grid['line_flux'],nan=0)
    for j, rest in enumerate(grid['rest_line_angstrom']):
        if cfg.get('convert_cached_air_lines_to_vacuum',False):
            rest=float(cloudy_air_to_vacuum(rest))
        responses = np.array([line_response(rest,z,v,cfg['velocity']['sigma_kms'],fw,throughput) for v in vel_grid])
        if max(responses) <= 0:
            continue
        cone_response = np.interp(velocity,vel_grid,responses)
        host_response = np.interp(host_velocity,vel_grid,responses)
        key = 'pa_alpha' if grid['line_labels'][j]=='H 1' and abs(grid['rest_line_angstrom'][j]-18751)<.1 else 'other_lines'
        for population, weight_map, model_index, response, local_velocity in (
                ('cone',cone_weight,cone_index,cone_response,velocity),
                ('host',host_weight,host_index,host_response,host_velocity)):
            # Each gas population has its own spectrum and Doppler shift.
            dust_velocity = local_velocity if independent_host else velocity
            center = rest*(1+z)*(1+dust_velocity/299792.458)
            attenuation = np.exp(-tau*(center/(18751*(1+z)))**(-cfg['components']['dust_exponent']))
            gas_maps[f'{population}_{key}'] += weight_map*line_flux[model_index,j]*response*attenuation
    weights = photon_bandpass_weights(fw,fw,throughput)
    for j,(wave,weight) in enumerate(zip(fw,weights)):
        # Density Jacobian for observed F_lambda. Foreground screen is shared;
        # intrinsic nebular emission follows its emitting population, not all gas.
        values = grid['continuum_rest_flambda'][:,j]/(1+z)
        attenuation = np.exp(-tau*(wave/(18751*(1+z)))**(-cfg['components']['dust_exponent']))
        for population,weight_map,model_index in (
                ('cone',cone_weight,cone_index),('host',host_weight,host_index)):
            gas_maps[f'{population}_nebular_continuum'] += weight*weight_map*values[model_index]*attenuation
    hii_diagnostics = {} if args.export_spectral_state else None
    if cfg.get('star_forming'):
        sf=cfg['star_forming']
        if cfg.get('calibration'):
            raise ValueError('A new star-forming component requires a new calibration, not the old four-component fit')
        grid_path=Path(sf['grid'])
        manifest=json.loads((grid_path.parent/'manifest.json').read_text())
        if not manifest['status'].startswith('completed stellar-ionized'):
            raise ValueError('H II grid has not completed the calculation checks')
        with np.load(grid_path,allow_pickle=False) as f:
            hii_grid={name:f[name] for name in f.files}
        if not np.array_equal(hii_grid['filter_wavelength_angstrom'],fw):
            raise ValueError('H II and AGN grids must use the same observed passband sampling')
        star_maps=render_hii_population(hii_grid,sf,east*high/2*fine_scale,
                                        north*high/2*fine_scale,z,tau,
                                        cfg['components']['dust_exponent'],over,rng,
                                        diagnostics=hii_diagnostics)
        agn_p99=float(np.percentile(sum(gas_maps.values()),99))
        hii_p99=float(np.percentile(sum(star_maps.values()),99))
        normalization=sf['relative_to_agn_gas_p99']*agn_p99/max(hii_p99,1e-30)
        gas_maps.update({f'starforming_{name}':layer*normalization for name,layer in star_maps.items()})
    gas = sum(gas_maps.values())
    gas_p99 = max(float(np.percentile(gas,99)),1e-12)
    # Parametric continuum amplitudes are assumptions, not stellar-population fits.
    bulge = np.exp(-3.5*np.sqrt(radius**2+.015**2))
    stellar = cfg['components']['stellar_bulge_relative_to_gas_p99']*gas_p99*bulge*np.exp(-tau)
    nuclear = np.zeros_like(gas)
    nuclear_color_factor = 1.
    if cfg.get('nuclear_sed'):
        response=nuclear_filter_response(fw,throughput,z,cfg['nuclear_sed'])
        # The existing amplitude now denotes the flat-F_lambda reference-band
        # normalization. Apply the explicit spectral shape through the filter.
        flat_response=float(photon_bandpass_weights(fw,fw,throughput).sum())/(1+z)
        nuclear_color_factor=response/flat_response
    nuclear[high//2,high//2] = (cfg['components']['nuclear_peak_relative_to_gas_p99']*
                              gas_p99*nuclear_color_factor/kernel.max())
    if cfg.get('nuclear_structure'):
        structure=cfg['nuclear_structure']
        total_nuclear=nuclear.sum()
        nuclear=total_nuclear*nuclear_spatial_profile(nuclear.shape,fine_scale,
                    structure['extended_fraction'],structure['sigma_arcsec'])
    components = dict(**gas_maps,stellar_continuum=stellar,nuclear_continuum=nuclear)
    rendered = {name:rebin_surface_brightness(convolve_kernel(layer,kernel),over) for name,layer in components.items()}
    calibration = None
    if cfg.get('calibration'):
        calibration = json.loads(Path(cfg['calibration']).read_text())
        variation = cfg.get('population_amplitude_factors', {})
        rendered = {name:layer*calibration['amplitudes_mjysr'][component_group(name)] /
                    calibration['template_scales_relative'][component_group(name)] *
                    variation.get(component_group(name),1.) for name,layer in rendered.items()}
    clean_relative = sum(rendered.values())
    # Aggregate spectral maps remain available for existing consumers, but must
    # not be added again when reconstructing the image from component_names.
    aggregates = {kind:sum(layer for name,layer in rendered.items()
                            if name in (f'cone_{kind}',f'host_{kind}',f'starforming_{kind}'))
                  for kind in ('pa_alpha','other_lines','nebular_continuum')}

    mask = rebin_surface_brightness(cone.astype(float),over) > .5
    mask[n//2-1:n//2+2,n//2-1:n//2+2] = True
    _,count = label(mask,structure=np.ones((3,3)))
    if count != 1:
        raise ValueError('Geometric target mask is disconnected')
    ry,rx = np.indices((n,n))
    r = np.hypot(rx-(n-1)/2,ry-(n-1)/2)*scale
    fitting = valid & (r>.3)
    baseline = reference['outer_median_science_mjysr']
    real_span = np.percentile(real[fitting],90)-np.percentile(real[fitting],10)
    sim_span = np.percentile(clean_relative[fitting],90)-np.percentile(clean_relative[fitting],10)
    amplitude = float(real_span/max(sim_span,1e-12))
    if calibration is not None:
        # Freeze the reference calibration across variants. Never rescale each
        # new galaxy to the real image's brightness percentiles.
        amplitude = 1.
        baseline = calibration['baseline_mjysr']
    clean = clean_relative*amplitude+baseline
    noise = gaussian_filter(rng.normal(size=(n,n)),cfg['noise']['correlation_sigma_pixels'])
    noise /= noise.std()
    sigma = np.where(valid,error,reference['outer_median_error_mjysr'])
    observed = clean+noise*sigma
    metadata = {
        'status':('exploratory population member with coarse reference calibration; not a physical decomposition'
                  if calibration else 'relative pilot with empirical single-band amplitude matching; not a fitted NGC 1068 model'),
        'configuration':str(args.config),'configuration_values':cfg,'redshift':z,'shape':[n,n],
        'calibration':cfg.get('calibration'),
        'relative_clean_units':'empirical MJy/sr' if calibration else 'relative photon-weighted units',
        'pixel_scale_arcsec':scale,'intrinsic_pixel_scale_arcsec':fine_scale,'oversample':over,
        'filter':'F187N','grid':cfg['grid'],'psf':cfg['psf'],
        'component_names':list(rendered),'aggregate_names':list(aggregates),'empirical_scale':amplitude,
        'independent_host_geometry':independent_host,
        'gas_structure':cfg.get('gas_structure'),'nuclear_structure':cfg.get('nuclear_structure'),
        'foreground_dust':{'model':dust_model,'controls':cfg.get('foreground_dust'),
            'screen_applies_to':'cone gas, host gas, H II emission and stellar continuum before PSF; nuclear continuum retains its separate normalized SED prescription',
            'map_definition':'pre-PSF intrinsic optical depth and coverage; coarse maps are area averages, transmission is mean(exp(-tau))',
            'limitations':'Prescribed foreground screen, no dust mass inference, scattering, thermal re-emission or energy balance; not a torus model'},
        'texture_controls':texture_cfg,
        'cone_label_semantics':'nominal projected cone geometry; soft AGN emission may extend outside it',
        'resolved_geometry':geometry_metadata,
        'nuclear_sed':cfg.get('nuclear_sed'), 'nuclear_sed_color_factor':nuclear_color_factor,
        'star_forming':cfg.get('star_forming'),
        'cached_agn_air_lines_converted_to_vacuum':cfg.get('convert_cached_air_lines_to_vacuum',False),
        'host_emission_fraction_inside_projected_cone_before_psf':float(
            sum(layer[cone].sum() for name,layer in gas_maps.items() if name.startswith('host_'))/
            max(sum(layer.sum() for name,layer in gas_maps.items() if name.startswith('host_')),1e-30)),
        'spectral_populations':{
            'cone':'AGN grid, cone geometry, radial outflow',
            'host':'Low-U AGN grid; separate projected disk and rotation, NOT star-forming H II models'
                   if independent_host else 'Legacy low-U gas excluded inside cone',
            'starforming':'Separate stellar-ionized grid and independently placed H II emission' if cfg.get('star_forming') else 'Not included',
            'stellar_continuum':'Assumed band-integrated bulge',
            'nuclear_continuum':'Assumed unresolved band-integrated source'},

        'galaxy_plus_background_baseline_mjysr':baseline,
        'units':'MJy/sr scale assigned empirically; not absolute Cloudy observer calibration',
        'noise':'Gaussian correlated template using real ERR amplitudes; not a full detector simulation',
        'label':'sharp projected geometric cone, independent of intrinsic emission transitions, brightness, continuum, PSF and noise',
        'missing_printed_line_entries_zero_filled_in_renderer':int(missing.sum()),
        'limitations':['Soft illumination and compact nuclear structure, when enabled, are phenomenological and not fitted',
                       'Dedicated H II component, when enabled, uses a blackbody source and incomplete faint-line list',
                       'Star-forming geometry and relative amplitudes are illustrative, not measured',
                       'Cached one-iteration Cloudy convergence warnings remain unresolved',
                       'Printed line intensities rounded; censored weak lines give incomplete line contribution',
                       'Host gas uses low-U AGN models, not a dedicated star-forming H II grid',
                       'Stellar and nuclear continua have assumed shapes and amplitudes',
                       'PSF detector NRCB4 but reference wavefront and center, no fitted sky rotation or drizzling',
                       'No date-specific detector effects, saturation or persistence simulation',
                       'No physically calibrated luminosity/covering factors or fitted geometry'],
    }
    output.mkdir(parents=True)
    if args.export_spectral_state:
        spectral_state = dict(cone_weight=cone_weight, host_weight=host_weight,
            cone_index=cone_index, host_index=host_index, velocity=velocity,
            host_velocity=host_velocity, tau=tau, kernel=kernel,
            stellar_band=stellar, nuclear_band=nuclear,
            configuration=json.dumps(cfg))
        if cfg.get('star_forming'):
            spectral_state.update(hii_spatial=hii_diagnostics['spatial'],
                hii_velocity=hii_diagnostics['velocity'], hii_normalization=normalization)
        np.savez_compressed(output/'spectral_state.npz', **spectral_state)
    np.savez_compressed(output/'image_mask.npz',input=observed.astype(np.float32),clean=clean,
                        relative_clean=clean_relative,mask=mask.astype(np.float32),
                        reference_science=real,reference_error=error,reference_valid=valid,
                        foreground_tau_pa_alpha_intrinsic=tau,
                        foreground_covered_intrinsic=dust_covered,
                        foreground_tau_pa_alpha_mean=rebin_surface_brightness(tau,over),
                        foreground_transmission_pa_alpha_mean=rebin_surface_brightness(np.exp(-tau),over),
                        foreground_covering_fraction=rebin_surface_brightness(dust_covered.astype(float),over),
                        **{name:layer*amplitude for name,layer in {**rendered,**aggregates}.items()},
                        intrinsic_cone_gas=rebin_surface_brightness(sum(
                            layer for name,layer in gas_maps.items() if name.startswith('cone_')),over),
                        intrinsic_host_gas=rebin_surface_brightness(sum(
                            layer for name,layer in gas_maps.items() if name.startswith('host_')),over),
                        metadata=json.dumps(metadata))
    for folder,array in [('images',observed.astype(np.float32)),('masks',mask.astype(np.float32))]:
        (output/'training_pair'/folder).mkdir(parents=True)
        np.save(output/'training_pair'/folder/'f187n_1069.npy',array)
    (output/'configuration.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import AsinhNorm
    fig,axes = plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
    high_display = float(np.percentile(real[valid],99.5))
    gas_units = 'empirical MJy/sr' if calibration else 'relative'
    panels = [(np.ma.array(real,mask=~valid),'Real F187N: MJy/sr',high_display),
              (observed,'Pilot: empirical MJy/sr scale',high_display),
              (mask,'Geometric training target',1),
              (aggregates['pa_alpha'],f'Cloudy Pa alpha: {gas_units}',np.percentile(aggregates['pa_alpha'],99.5)),
              (aggregates['other_lines']+aggregates['nebular_continuum'],f'Other gas lines + nebular continuum: {gas_units}',np.percentile(aggregates['other_lines']+aggregates['nebular_continuum'],99.5)),
              (rendered['stellar_continuum']+rendered['nuclear_continuum'],f'Assumed stellar + nuclear continuum: {gas_units}',np.percentile(rendered['stellar_continuum']+rendered['nuclear_continuum'],99.5))]
    for axis,(array,title,top) in zip(axes.flat,panels):
        top = max(float(top),1e-12)
        axis.imshow(array,origin='lower',cmap='gray' if title=='Geometric training target' else 'magma',
                    norm=AsinhNorm(linear_width=top*.03,vmin=0,vmax=top))
        axis.set_title(title,fontsize=11)
        axis.set_axis_off()
    fig.suptitle('F187N pilot: physical gas emission with illustrative galaxy components\nGeometry and continuum are not fitted; this is not a validated segmentation dataset')
    fig.savefig(output/'preview.png',dpi=150)
    plt.close(fig)
    cone_gas = sum(layer for name,layer in rendered.items() if name.startswith('cone_'))
    host_gas = sum(layer for name,layer in rendered.items() if name.startswith('host_'))
    gas_panels=[(cone_gas,'AGN cone gas'),(host_gas,'Low-U host gas')]
    if cfg.get('star_forming'):
        sf_gas=sum(layer for name,layer in rendered.items() if name.startswith('starforming_'))
        gas_panels.append((sf_gas,'Stellar-ionized H II gas'))
    combined=sum(layer for layer,title in gas_panels)
    gas_panels.extend([(combined,'Combined gas'),(mask,'Cone target only')])
    fig,axes=plt.subplots(1,len(gas_panels),figsize=(4*len(gas_panels),4),constrained_layout=True)
    top=max(float(np.percentile(combined,99.5)),1e-12)
    for axis,(array,title) in zip(axes,gas_panels):
        axis.imshow(array,origin='lower',cmap='magma' if title!='Cone target only' else 'gray',
                    norm=AsinhNorm(linear_width=top*.03,vmin=0,vmax=top) if title!='Cone target only' else None)
        axis.set_title(title)
        axis.set_axis_off()
    fig.suptitle('Separate emitting populations; host gas can overlap the cone in projection')
    fig.savefig(output/'components.png',dpi=150)

    plt.close(fig)
    print(f'Saved pilot and unchanged geometric mask: {output}')


if __name__=='__main__':
    main()
