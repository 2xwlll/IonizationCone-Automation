#!/usr/bin/env python3
"""Line atlas and honest observational-reference/model spectral comparison.

No sampled observed F187N spectrum is available in the imported data. Never
invent observed line profiles or treat flux/EW continuum proxies as spectra.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/agn-spectrum-matplotlib')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.jwst_utils.nircam import convolve_kernel
from src.simple_utils.nuclear_sed import nuclear_flambda_shape
from src.simple_utils.synthetic_photometry import photon_bandpass_weights
from src.simple_utils.wavelengths import cloudy_air_to_vacuum

OUT = ROOT/'results/cloudy/f187n_spectral_reference_v2'
EXPORT = ROOT/'results/cloudy/ngc1068_f187n_spectral_export_v1'
CURRENT = ROOT/'results/cloudy/ngc1068_f187n_soft_gas_v1/image_mask.npz'
MARTINS = 'https://academic.oup.com/mnras/article/406/4/2168/1017118'
CORONAL = 'https://academic.oup.com/mnras/article/430/3/2411/1751793'
FILTER = 'https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-filters'


def velocity_histogram(velocity, energy, bins):
    """Adjoint of the renderer's piecewise-linear response interpolation."""
    t = np.clip((velocity-bins[0])/(bins[1]-bins[0]), 0, len(bins)-1)
    i = np.minimum(np.floor(t).astype(int), len(bins)-2)
    f = t-i
    return (np.bincount(i.ravel(), weights=(energy*(1-f)).ravel(), minlength=len(bins))+
            np.bincount((i+1).ravel(), weights=(energy*f).ravel(), minlength=len(bins)))


def integrated_profile(wave, rest, redshift, sigma_kms, velocities, energies):
    """Sum Gaussian profiles retaining spatially varying velocity and energy."""
    sigma = rest*(1+redshift)*sigma_kms/299792.458
    centers = rest*(1+redshift)*(1+velocities/299792.458)
    result = np.zeros_like(wave)
    for center, energy in zip(centers, energies):
        lo, hi = np.searchsorted(wave, [center-8*sigma, center+8*sigma])
        result[lo:hi] += energy*np.exp(-.5*((wave[lo:hi]-center)/sigma)**2)/(np.sqrt(2*np.pi)*sigma)
    return result


def reconstruct(state, cfg, grid, hii, wave, aperture=None, *, intrinsic_acceptance=None):
    z = cfg['redshift']; over = cfg['oversample']
    fw = grid['filter_wavelength_angstrom']; throughput = grid['throughput']
    # Adjoint PSF includes light lost outside the exact saved image footprint.
    if aperture is not None and intrinsic_acceptance is not None:
        raise ValueError('Choose detector aperture or independent intrinsic acceptance, not both')
    if intrinsic_acceptance is not None:
        spatial_acceptance=np.asarray(intrinsic_acceptance,dtype=float)
        if spatial_acceptance.shape!=state['tau'].shape or not np.isfinite(spatial_acceptance).all() or np.any((spatial_acceptance<0)|(spatial_acceptance>1)):
            raise ValueError('Intrinsic acceptance must be a finite probability map on the source grid')
    elif aperture is None:
        spatial_acceptance=np.ones_like(state['tau'])
    else:
        aperture=np.asarray(aperture,dtype=float)
        if aperture.shape!=(cfg['spatial_pixels'],cfg['spatial_pixels']) or not np.isfinite(aperture).all() or np.any(aperture<0):
            raise ValueError('Aperture must be a finite, nonnegative detector-grid weight map')
        spatial_acceptance=np.repeat(np.repeat(aperture,over,axis=0),over,axis=1)
    # An independently supplied source-grid acceptance already includes its
    # instrument PSF. Do not blur it again with the JWST imaging PSF.
    acceptance = (spatial_acceptance if intrinsic_acceptance is not None else
                  convolve_kernel(spatial_acceptance, state['kernel'][::-1, ::-1]))/over**2
    tau = state['tau']; exponent = cfg['components']['dust_exponent']
    spectra = {}; fw_continua = {}
    for population in ('cone','host','starforming'):
        sf = population == 'starforming'
        g = hii if sf else grid
        index = cfg['star_forming']['model_index'] if sf else state[f'{population}_index']
        spatial = state['hii_spatial']*float(state['hii_normalization']) if sf else state[f'{population}_weight']
        velocity = state['hii_velocity'] if sf else state['velocity' if population=='cone' else 'host_velocity']
        bins = np.linspace(-400,400,81) if sf else np.linspace(-1200,1200,241)
        sigma = cfg['star_forming']['sigma_kms'] if sf else cfg['velocity']['sigma_kms']
        flux = np.nan_to_num(g['line_flux'], nan=0.)
        pa = np.zeros_like(wave); other = np.zeros_like(wave)
        for j, stored in enumerate(g['rest_line_angstrom']):
            rest = float(stored if sf else cloudy_air_to_vacuum(stored))
            # Follow the original code's exact screen convention for each population.
            center_for_screen = rest*(1+velocity/299792.458)
            if not sf:
                center_for_screen *= 1+z
            denominator = 18751. if sf else 18751.*(1+z)
            attenuation = np.exp(-tau*(center_for_screen/denominator)**(-exponent))
            energy = spatial*acceptance*flux[index,j]*attenuation
            bins_energy = velocity_histogram(velocity, energy, bins)
            profile = integrated_profile(wave, rest, z, sigma, bins, bins_energy)
            is_pa = g['line_labels'][j]=='H 1' and abs(rest-18756.11985)<.15
            if is_pa: pa += profile
            else: other += profile
        spectra[f'{population}_pa_alpha'] = pa
        spectra[f'{population}_other_lines'] = other
        continuum = np.empty_like(fw)
        for j, w in enumerate(fw):
            values = g['continuum_rest_flambda'][index,j]/(1+z)
            screen = np.exp(-tau*(w/(18751.*(1+z)))**(-exponent))
            continuum[j] = np.sum(spatial*acceptance*values*screen)
        fw_continua[f'{population}_nebular_continuum'] = continuum
        spectra[f'{population}_nebular_continuum'] = np.interp(wave, fw, continuum)
        print('Reconstructed', population, flush=True)
    response_weights = photon_bandpass_weights(fw, fw, throughput)
    for name in ('stellar','nuclear'):
        shape = np.ones_like(fw)
        if name == 'nuclear':
            shape = nuclear_flambda_shape(fw/(1+z),cfg['nuclear_sed'])/(1+z)
        band_sum = np.sum(state[f'{name}_band']*acceptance)
        fw_continua[f'{name}_continuum'] = band_sum*shape/(response_weights@shape)
        spectra[f'{name}_continuum'] = np.interp(wave,fw,fw_continua[f'{name}_continuum'])
    return spectra, fw_continua


def reference_continuum(rest_wave, fit):
    b = fit['best_continuum_scenario']
    cfg = dict(reference_rest_angstrom=18756.11985, hot_dust_temperature_k=b['dust_temperature_k'],
        hot_dust_fraction_at_reference=1.,powerlaw_alpha_nu=-.5,tau_at_reference=0.,extinction_exponent=1.3)
    basis = np.array([nuclear_flambda_shape(rest_wave,cfg), (rest_wave/18756.11985)**(-1.5),
        nuclear_flambda_shape(rest_wave,{**cfg,'hot_dust_temperature_k':4500.})])
    return np.asarray(b['coefficients'])@basis*1e-14*np.exp(-b['tau_at_reference']*(rest_wave/18756.11985)**(-1.3))


def line_catalog(z, fw, throughput):
    entries = [
        ('He II',18636.5,'hard','High-excitation ionized gas; supports hard ionizing radiation, not uniquely AGN','Model transition; no measured flux in imported table',MARTINS),
        ('He I',18685.4,'star','Ionized gas in AGN and H II regions; not an AGN-only tracer','Model transition; no measured flux in imported table',MARTINS),
        ('He I',18697.2,'star','Ionized gas in AGN and H II regions; not an AGN-only tracer','Model transition; no measured flux in imported table',MARTINS),
        ('He II',18743.1,'hard','High-excitation gas; blended with Pa alpha at the adopted widths','Model transition; no measured flux in imported table',MARTINS),
        ('H I Pa alpha',18751.,'hydrogen','Recombining ionized gas: AGN cone, host and star-forming H II regions','Not reliably measured by Martins et al. (2010); order gap and telluric absorption',MARTINS),
        ('He II',18757.,'hard','High-excitation gas; blended with Pa alpha at the adopted widths','Censored in some cached AGN models; not an observed detection',MARTINS),
        ('Fe II blend',18890.,'iron','Partially ionized gas; AGN photoionization or shocks can contribute; blend identity from CLOUDY','Model blend; not a measured diagnostic in imported table',MARTINS),
        ('[Si VI]',19624.7,'coronal','AGN coronal gas, nucleus and extended coronal-line region; useful AGN diagnostic outside main F187N band','Measured in Martins et al. Table 1 and spatially studied by Mazzalay et al. (2013)',CORONAL),
        ('[O III]',5006.84,'coronal','Optical narrow-line gas / ionization cones; outside NIRCam F187N','Original optical tracer; not in F187N',MARTINS),
    ]
    rows = []
    for label,air,kind,origin,status,url in entries:
        rest = float(cloudy_air_to_vacuum(air)); obs=rest*(1+z)
        rows.append(dict(line=label,stored_air_angstrom=air,rest_vacuum_um=rest/1e4,
            observed_vacuum_um=obs/1e4,classification=kind,physical_origin=origin,
            measurement_status=status,throughput_fraction_of_peak_at_v0=float(np.interp(obs,fw,throughput,left=0,right=0)/max(throughput)),
            source_url=url,atomic_wavelength_source='Cached CLOUDY line inventory; air-to-vacuum conversion, not a published observed centroid',
            local_inventory='results/cloudy/f187n_physics_inventory_v1/line_inventory.csv'))
    return rows


def plots(wave, fw, throughput, ref, model, catalog, norm_model, norm_ref):
    colors = dict(hard='#6b3fa0',star='#16805d',hydrogen='#ba5a10',iron='#9c4660',coronal='#b62424')
    main = (1.858,1.905)
    height = max(2.,float(np.max(model[(wave/1e4>main[0])&(wave/1e4<main[1])]/norm_model))*1.22)
    plt.rcParams.update({'font.size':12,'axes.labelsize':13,'axes.titlesize':14,'savefig.facecolor':'white'})
    for overlay in (False,True):
        fig = plt.figure(figsize=(14,10))
        gs = fig.add_gridspec(3,1,height_ratios=(1.,3.3,1.6),hspace=.28)
        strip, ax, context = [fig.add_subplot(gs[i]) for i in range(3)]
        fig.subplots_adjust(left=.09,right=.91,top=.87,bottom=.15)
        fig.suptitle('NGC 1068 — F187N spectral reference'+(' + current synthetic galaxy' if overlay else ''),fontsize=20,y=.97)
        fig.text(.5,.924,'Expected line positions + published continuum constraints; not a measured F187N spectrum',ha='center',fontsize=12,color='#653232')
        strip.set(xlim=main,ylim=(0,3.9)); strip.axis('off')
        positions = [(1.8665,2.1),(1.8718,3.35),(1.8766,1.7),(1.8804,3.35),(1.8864,2.1),(1.893,3.35),(1.900,2.1)]
        for row,(text_x,text_y) in zip(catalog[:7],positions):
            x=row['observed_vacuum_um']; c=colors[row['classification']]
            text = row['line'].replace('Pa alpha','Pa α')+'\n'+f'{x:.5f} μm'
            strip.annotate(text,xy=(x,.05),xytext=(text_x,text_y),ha='center',va='center',fontsize=10.5,color=c,
                arrowprops=dict(arrowstyle='-',color=c,lw=.9))
            ax.axvline(x,color=c,lw=.8,alpha=.22,zorder=0)
        ax.plot(wave/1e4,ref/norm_ref,'--',color='#3a3a3a',lw=2,label='Continuum model constrained by published flux/EW proxies')
        if overlay:
            ax.plot(wave/1e4,model/norm_model,color='#007eae',lw=1.8,label='Current synthetic galaxy: all components, full modeled field')
        ax.set(xlim=main,ylim=(0,height),ylabel='Relative $F_\lambda$\n(continuum = 1 at 1.874 μm)')
        ax.grid(alpha=.15); ax.legend(loc='upper left',fontsize=10,framealpha=.97)
        ax.text(.02,.73,'Line positions above are identifications,\nnot measured line strengths.',transform=ax.transAxes,fontsize=10,color='#555555')
        ax.text(.02,.12,'Pa α is shared by AGN-ionized and stellar-ionized gas.\nHe II indicates higher excitation; it is not an AGN-only label.',transform=ax.transAxes,fontsize=10,color='#555555',bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
        right=ax.twinx(); right.fill_between(fw/1e4,throughput/max(throughput),color='#818b9b',alpha=.13,zorder=0)
        right.set(ylim=(0,1.05),ylabel='F187N throughput / peak'); right.tick_params(colors='#687383'); right.set_zorder(ax.get_zorder()-1); ax.patch.set_visible(False)
        context.fill_between(fw/1e4,throughput/max(throughput),color='#bcc5d3',alpha=.7)
        context.plot(fw/1e4,throughput/max(throughput),color='#687383',lw=1.2)
        context.axvspan(*main,color='#009ec2',alpha=.08,label='Zoomed interval above')
        context.set(xlim=(1.80,2.00),ylim=(0,1.65),xlabel='Observed vacuum wavelength (μm)',ylabel='Throughput / peak')
        for row,text_x in [(catalog[4],1.843),(catalog[7],1.958)]:
            x=row['observed_vacuum_um']; t=np.interp(x,fw/1e4,throughput)/max(throughput)
            context.plot(x,t,'o',color=colors[row['classification']],ms=5)
            text = row['line'].replace('Pa alpha','Pa α')+f'  {x:.5f} μm'
            if row['classification']=='coronal': text+='\nAGN coronal tracer; outside main band'
            context.annotate(text,xy=(x,t),xytext=(text_x,1.36),fontsize=10,ha='center',va='center',color=colors[row['classification']],arrowprops=dict(arrowstyle='-',lw=.8,color=colors[row['classification']]))
        context.grid(alpha=.15)
        fig.text(.09,.085,'Same axes and continuum normalization in both figures. Full-field model versus nuclear continuum reference: shape illustration, not an aperture-matched fit.',fontsize=10)
        fig.text(.09,.057,'Continuum reference: Martins et al. 2010, Table 1 flux/EW → phenomenological fit. No measured Pa α profile or in-band line flux is supplied by that source.',fontsize=10)
        fig.text(.09,.029,'Line wavelengths: CLOUDY air → vacuum, z = 0.003793, zero bulk velocity markers. Synthetic profiles retain model velocities; stellar continuum shape is assumed flat $F_\lambda$.',fontsize=10)
        name='02_reference_with_current_synthetic' if overlay else '01_ngc1068_f187n_reference'
        for extension in ('png','pdf','svg'):
            fig.savefig(OUT/f'{name}.{extension}',dpi=200)
        plt.close(fig)


def main():
    if OUT.exists(): raise FileExistsError('Choose a fresh output directory')
    OUT.mkdir(parents=True)
    state=dict(np.load(EXPORT/'spectral_state.npz'))
    cfg=json.loads(str(state['configuration']))
    grid=dict(np.load(ROOT/cfg['grid'])); hii=dict(np.load(ROOT/cfg['star_forming']['grid']))
    wave=np.arange(grid['filter_wavelength_angstrom'][0],grid['filter_wavelength_angstrom'][-1]+.05,.1)
    fw=grid['filter_wavelength_angstrom']; t=grid['throughput']
    spectra,continua=reconstruct(state,cfg,grid,hii,wave)
    original=np.load(CURRENT); rerender=np.load(EXPORT/'image_mask.npz')
    for name in json.loads(str(original['metadata']))['component_names']+['mask','input','relative_clean']:
        np.testing.assert_array_equal(original[name],rerender[name])
    weights=photon_bandpass_weights(wave,fw,t)
    coarse_weights=photon_bandpass_weights(fw,fw,t)
    amplitude=json.loads(str(original['metadata']))['empirical_scale']
    checks={}
    for name,spectrum in spectra.items():
        # Continuum maps were integrated on the original filter knots. Use those
        # original knots for round-trip comparison, not an approximately sampled interpolant.
        predicted=float(coarse_weights@continua[name]) if name in continua else float(weights@spectrum)
        actual=float(original[name].sum()/amplitude)
        error=abs(predicted/actual-1)
        checks[name]={'spectral_band_photon_sum':predicted,'saved_image_relative_sum':actual,'relative_error':error}
        if error>3e-4: raise AssertionError((name,error))
    fit=json.loads((ROOT/'results/cloudy/ngc1068_spectral_constraints_v1/fit.json').read_text())
    ref=reference_continuum(wave/(1+cfg['redshift']),fit)/(1+cfg['redshift'])
    model=sum(spectra.values()); cont=sum(spectra[k] for k in continua)
    norm_model=float(np.interp(18740.,wave,cont)); norm_ref=float(np.interp(18740.,wave,ref))
    catalog=line_catalog(cfg['redshift'],fw,t)
    with (OUT/'important_lines.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(catalog[0])); writer.writeheader(); writer.writerows(catalog)
    np.savez_compressed(OUT/'spectra.npz',observed_vacuum_angstrom=wave,
        synthetic_relative_flambda=model,synthetic_continuum=cont,
        reference_continuum_proxy_model_flambda=ref,synthetic_normalization=norm_model,
        reference_normalization=norm_ref,filter_wavelength_angstrom=fw,throughput=t,**spectra)
    with (OUT/'normalized_spectra.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['observed_vacuum_um','reference_continuum_model_NOT_observed_spectrum','current_synthetic_flambda','current_synthetic_continuum','throughput_fraction_of_peak'])
        writer.writerows(zip(wave/1e4,ref/norm_ref,model/norm_model,cont/norm_model,np.interp(wave,fw,t)/max(t)))
    plots(wave,fw,t,ref,model,catalog,norm_model,norm_ref)
    (OUT/'checks.json').write_text(json.dumps({'rerender_all_components_bitwise_identical':True,'component_band_roundtrip':checks},indent=2)+'\n')
    inputs=[CURRENT,EXPORT/'spectral_state.npz',ROOT/cfg['grid'],ROOT/cfg['star_forming']['grid'],ROOT/'results/cloudy/ngc1068_spectral_constraints_v1/fit.json',Path(__file__)]
    (OUT/'provenance.json').write_text(json.dumps({'sources':[MARTINS,CORONAL,FILTER],
        'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        'status':'Line atlas plus continuum-proxy reference, not an observed spectral measurement; overlaid current image-generator spectrum is unchanged in band-integrated signal.',
        'normalization':'Each continuum equals one at observed 1.874 um; full-field synthetic vs nuclear reference is not a flux or aperture-matched comparison.',
        'stellar_shape':'Flat observed F_lambda chosen to extend the existing band-integrated stellar component spectrally, preserving its exact F187N signal; no SSP absorption features implied.',
        'limits':json.loads(str(original['metadata']))['limitations']},indent=2)+'\n')
    guide = ['# NGC 1068: F187N spectral reference and current synthetic model',
        '', '## What these two matching figures mean', '',
        'The first figure is an expected-line atlas and continuum-proxy reference, NOT a measured F187N spectrum. The second adds the spectrum reconstructed from the exact current soft-gas image model. Axes, throughput curve, line markers and continuum normalization are identical. The imported observations do not supply reliable in-band line fluxes or a Pa-alpha profile. Do not describe the dashed reference as a measured NGC 1068 spectrum.', '',
        'The dashed continuum is the existing phenomenological fit to nuclear flux/EW proxies from Martins et al. (2010). Those sparse proxies constrain a continuum model across the near infrared; they are not sampled spectral pixels within F187N. No observed line heights have been invented.', '',
        'The blue curve integrates the same 2D model field that produced the current images, including the actual gas-model assignments, line velocities and widths, extinction, independent H II component and nuclear SED. Adjoint PSF weighting accounts for flux leaving the finite image. All re-rendered components, image pixels and the mask are bitwise identical to the prior model. Re-integrating the extracted components through F187N recovers every image component to better than 0.0001 percent (see checks.json).', '',
        'The original stellar component defines only a band-integrated amplitude. A flat observed F_lambda is assumed here to extend it into a spectrum while exactly preserving its F187N contribution. This is not a fitted stellar-population spectrum. Nebular continuum and the nuclear color retain their existing spectral prescriptions.', '',
        'Each curve is normalized to its own continuum at observed 1.874 um. This compares spectral shape, not absolute flux. The synthetic full field is about 7.9 x 7.9 arcsec; the published nuclear extraction is 0.8 x 1.6 arcsec. Epoch, seeing, aperture and registration differ. This is not an aperture-matched model fit or proof of spectral agreement.', '',
        '## Important line guide', '',
        'Wavelengths below are vacuum, at model z=0.003793 and zero additional bulk velocity. In-image velocity fields broaden and shift the actual synthetic profiles. Relative throughput is the value at the zero-velocity center, not a predicted line-flux fraction or a detection probability.', '',
        '| Line | Rest vacuum (um) | Observed (um) | F187N T/Tpeak | Physical origin / status |',
        '|---|---:|---:|---:|---|']
    for row in catalog:
        guide.append(f"| {row['line']} | {row['rest_vacuum_um']:.6f} | {row['observed_vacuum_um']:.6f} | {row['throughput_fraction_of_peak_at_v0']:.3g} | {row['physical_origin']}; {row['measurement_status']} |")
    guide += ['', '## Where to find the lines and observations', '',
        f'- [Martins et al. 2010, Tables 1–3 and Section 3.1]({MARTINS}): measured nuclear and regional near-IR lines, including [Si VI], [Fe II] 1.257/1.643 um, hydrogen Pa beta and Br gamma. Section 3.1 and Figure 4 explain why Pa alpha is not a reliable measured constraint in this source.',
        f'- [Mazzalay et al. 2013: Resolving the coronal line region of NGC 1068]({CORONAL}): spatially resolved [Si VI] and other coronal-line diagnostics. [Si VI] is outside F187N\'s main band at the adopted redshift.',
        f'- [STScI NIRCam filter documentation and official throughput curves]({FILTER}): the actual F187N transmission response.',
        '- Local observed line tables: `data/spectra/ngc1068_martins2010_v1/nuclear_lines.csv` and `regional_lines.csv`. These are transcribed line measurements, not downloaded sampled spectra.',
        '- Local modeled wavelengths and availability: `results/cloudy/f187n_physics_inventory_v1/line_inventory.csv`; underlying AGN grid `data/cloudy/ngc1068_f187n_pilot_v1/grid.npz` and H II grid `data/cloudy/hii_f187n_calculated_v2/grid.npz`. He I/He II entries around 1.87 um and the Fe II blend are model transitions, not confirmed individual detections in the imported NGC 1068 tables.',
        '- `important_lines.csv` provides wavelengths, physical origin, measured/model status and links. Published nominal wavelengths and model vacuum wavelengths must not be treated as identical precision centroids.', '',
        '## Remaining model limitations', '',
        'The current image uses the older AGN grid with convergence warnings and an incomplete printed line inventory. The new converged three-model spectral pilot and atomic-data control have not replaced this image grid. Continuum shapes, morphology and relative population amplitudes remain assumed. Missing weak lines are censored in the source grid and zero-filled only by the renderer; this does not establish that real emission is absent. No shock grid or complete stellar-population absorption spectrum is included.', '',
        'Outputs: two PNGs for slides, matching vector PDF/SVG files, spectra.npz, normalized_spectra.csv, important_lines.csv, checks.json and provenance.json. Spectra are BEFORE multiplication by the throughput; the filter is shown separately to avoid confusing a galaxy spectrum with its detector response. The saved image is a photon-weighted integral, not a wavelength-resolved observation.', '',
        'Reproduce the spectral state using the export configuration and `render_f187n_pilot.py --export-spectral-state`; then run `plot_f187n_spectral_reference.py` with a fresh OUT directory. Originals remain untouched.', '']
    (OUT/'README.md').write_text('\n'.join(guide))
    print('Saved',OUT,flush=True)


if __name__=='__main__':main()
