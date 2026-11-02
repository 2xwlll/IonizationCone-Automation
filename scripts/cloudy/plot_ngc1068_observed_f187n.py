#!/usr/bin/env python3
"""Plot actual ESO spectral samples, retaining the telluric limitation explicitly."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR','/tmp/agn-observed-spectrum-mpl')
import numpy as np
from scipy.ndimage import gaussian_filter1d
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.simple_utils.wavelengths import cloudy_air_to_vacuum
from scripts.cloudy.plot_f187n_spectral_reference import line_catalog

OUT=ROOT/'results/cloudy/ngc1068_observed_f187n_spectra_v1'
OBS=ROOT/'data/spectra/ngc1068_eso_xshooter_v1/3273695/spectrum.fits'
SYN=ROOT/'results/cloudy/f187n_spectral_reference_v2/spectra.npz'
RELEASE='https://www.eso.org/sci/observing/phase3/data_releases/xshooter_echelle.pdf'
ARCHIVE='https://archive.eso.org/datalink/links?ID=ivo://eso.org/ID?ADP.2022-08-12T12:26:18.491'


def load_observation(path):
    with fits.open(path) as hd:
        hd.verify('exception')
        header=hd[0].header.copy(); tabhead=hd[1].header.copy();data=hd[1].data[0]
        if tabhead['TUCD1']!='em.wl;obs.atmos' or tabhead['TUNIT1']!='nm':
            raise ValueError('Wavelength medium or units need a fresh audit')
        if header['SPECSYS']!='TOPOCENT':raise ValueError('Not a topocentric spectrum')
        if 'angstrom' not in tabhead['TUNIT2'].lower():raise ValueError('Unexpected flux-density units')
        air=np.asarray(data['WAVE'],float)*10
        vacuum=cloudy_air_to_vacuum(air)
        correction=float(header['ESO QC VRAD HELICOR'])
        helio=vacuum*(1+correction/299792.458)
        jacobian=np.gradient(helio,air)
        # Preserve spectral-energy integrals under the change of wavelength variable.
        flux=np.asarray(data['FLUX'],float)/jacobian
        error=np.asarray(data['ERR'],float)/jacobian
        quality=np.asarray(data['QUAL']).copy()
        valid=(quality==0)&np.isfinite(flux)&np.isfinite(error)&(error>0)&np.isfinite(helio)
        if not np.all(np.diff(helio)>0):raise ValueError('Non-monotonic wavelength axis')
    return helio,flux,error,quality,valid,header,tabhead


def make_figures(wave,flux,error,good,header,sw,synthetic,scale,fw,throughput,lines):
    unit=1e-15; zoom=(1.858,1.905); context=(1.75,2.002)
    observed=np.where(good,flux/unit,np.nan)
    uncertainty=np.where(good,error/unit,np.nan)
    model=synthetic*scale/unit
    main_obs=(wave/1e4>zoom[0])&(wave/1e4<zoom[1])&good
    main_syn=(sw/1e4>zoom[0])&(sw/1e4<zoom[1])
    top=max(float(np.max(flux[main_obs]/unit)),float(model[main_syn].max()))*1.3
    bottom=min(-.1,float(np.min(flux[main_obs]/unit))*1.15)
    broad_obs=(wave/1e4>context[0])&(wave/1e4<context[1])&good
    broad_syn=(sw/1e4>context[0])&(sw/1e4<context[1])
    broad_top=max(float(np.max(flux[broad_obs]/unit)),float(model[broad_syn].max()))*1.4
    colors={'hard':'#6b3fa0','star':'#157b59','hydrogen':'#b85c08','iron':'#9b4560','coronal':'#b52727'}
    plt.rcParams.update({'font.size':12,'axes.labelsize':13,'savefig.facecolor':'white'})
    geometry=[]
    for overlay in (False,True):
        fig=plt.figure(figsize=(14,10))
        gs=fig.add_gridspec(3,1,height_ratios=[.9,3.4,2.0],hspace=.30)
        strip,ax,wide=[fig.add_subplot(gs[i]) for i in range(3)]
        fig.subplots_adjust(left=.095,right=.90,top=.85,bottom=.17)
        fig.suptitle('NGC 1068 — measured spectrum'+(' + current synthetic model' if overlay else ''),fontsize=20,y=.97)
        fig.text(.5,.925,'ESO/VLT X-shooter • 2022-07-30 • NIR slit spectrum • ADP.2022-08-12T12:26:18.491',ha='center',fontsize=12)
        fig.text(.5,.891,'Atmospheric absorption is NOT corrected: the F187N-range profile is not an intrinsic galaxy spectrum.',ha='center',fontsize=12,color='#9a3224')
        strip.set(xlim=zoom,ylim=(0,3.9));strip.axis('off')
        positions=[(1.8665,2.1),(1.8718,3.35),(1.8766,1.7),(1.8804,3.35),(1.8864,2.1),(1.893,3.35),(1.900,2.1)]
        for row,(tx,ty) in zip(lines[:7],positions):
            x=row['observed_vacuum_um'];label=row['line'].replace('Pa alpha','Pa α')
            if row['line']!='H I Pa alpha':label+='*'
            strip.annotate(label+'\n'+f'{x:.5f} μm',xy=(x,.05),xytext=(tx,ty),ha='center',va='center',fontsize=10.5,color=colors[row['classification']],arrowprops=dict(arrowstyle='-',color=colors[row['classification']],lw=.8))
            ax.axvline(x,color=colors[row['classification']],alpha=.2,lw=.8)
        for panel,limits,ymax in [(ax,zoom,top),(wide,context,broad_top)]:
            # This broad caution band is not a measured transmission curve or a fabricated correction.
            panel.axvspan(1.80,1.98,facecolor='#eab984',alpha=.13,zorder=-2)
            panel.plot(wave/1e4,observed,color='#202020',lw=.85,label='Measured X-shooter samples (QUAL = 0)')
            panel.fill_between(wave/1e4,observed-uncertainty,observed+uncertainty,color='#777777',alpha=.20,linewidth=0,label='Pipeline ±1σ statistical error')
            if overlay:
                panel.plot(sw/1e4,model,color='#007eae',lw=1.65,label='Current synthetic model (no Earth atmosphere)')
            panel.set(xlim=limits,ylim=(bottom,ymax),yscale='symlog',ylabel=r'$F_\lambda$ ($10^{-15}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)')
            panel.set_yscale('symlog',linthresh=.1,linscale=.5)
            panel.grid(alpha=.16)
        ax.legend(loc='upper left',fontsize=9.3,framealpha=.96)
        ax.text(.02,.08,'Expected line positions; * = model-transition guide, not a detection.\nFlagged samples are gaps; no continuum or emission peaks have been filled in.',transform=ax.transAxes,fontsize=9.5,bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
        wide.set_xlabel('Heliocentric vacuum wavelength (μm)')
        wide.axvspan(*zoom,color='#009ec2',alpha=.08)
        twin=wide.twinx();twin.plot(fw/1e4,throughput/max(throughput),color='#758295',ls='--',lw=1.4)
        twin.set(ylim=(0,1.45),ylabel='F187N throughput / peak')
        wide.axvspan(1.985,1.995,color='#3b8877',alpha=.12)
        wide.text(1.786,.92,'Broad atmospheric-absorption caution region',transform=wide.get_xaxis_transform(),color='#945c29',fontsize=9)
        for row,tx,ty in [(lines[4],1.857,.75),(lines[7],1.966,.52)]:
            x=row['observed_vacuum_um'];wide.axvline(x,color=colors[row['classification']],alpha=.5,lw=.8)
            wide.annotate(row['line'].replace('Pa alpha','Pa α')+'\n'+f'{x:.5f} μm',xy=(x,.10),xycoords=wide.get_xaxis_transform(),xytext=(tx,ty),textcoords=wide.get_xaxis_transform(),ha='center',fontsize=10,color=colors[row['classification']],arrowprops=dict(arrowstyle='-',color=colors[row['classification']],lw=.8))
        fig.text(.095,.11,'Identical measured samples, flags, axes and labels in both figures. Symlog flux axis retains negative samples; wavelength gaps are not interpolated.',fontsize=10)
        fig.text(.095,.081,'Model overlay: full 7.9″ field, approximate X-shooter spectral blur, one scale from 1.985–1.995 μm (green); no in-band line fit or aperture match.',fontsize=10)
        fig.text(.095,.052,'Air wavelengths → vacuum (Peck–Reeder convention), then heliocentric correction from FITS header. Flux-density Jacobian applied. Redshift guides: z = 0.003793.',fontsize=10)
        fig.text(.095,.023,'Source: ESO XSHOOTER_ECHELLE product, programme 109.22YE.001. Ground-based atmospheric attenuation prevents a direct intrinsic-spectrum validation in F187N.',fontsize=10)
        name='02_measured_ngc1068_with_synthetic' if overlay else '01_measured_ngc1068_labeled'
        for ext in ('png','pdf','svg'):fig.savefig(OUT/f'{name}.{ext}',dpi=200)
        geometry.append(dict(main_xlim=list(ax.get_xlim()),main_ylim=list(ax.get_ylim()),context_xlim=list(wide.get_xlim()),context_ylim=list(wide.get_ylim())))
        plt.close(fig)
    assert geometry[0]==geometry[1]
    return geometry[0]


def main():
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    wave,flux,error,qual,good,header,tabhead=load_observation(OBS)
    source=np.load(SYN);sw=source['observed_vacuum_angstrom'];fw=source['filter_wavelength_angstrom'];t=source['throughput']
    # Spectral blur is a display observation operator; it does not alter the galaxy model.
    sigma=18827.2618/float(header['SPEC_RES'])/2.354820045/float(np.median(np.diff(sw)))
    syn=gaussian_filter1d(source['synthetic_relative_flambda'],sigma,mode='nearest')
    cont=gaussian_filter1d(source['synthetic_continuum'],sigma,mode='nearest')
    anchor=good&(wave>=19850)&(wave<=19950)
    if anchor.sum()<10:raise ValueError('Insufficient valid anchor samples')
    scale=float(np.median(flux[anchor]))/float(np.median(np.interp(wave[anchor],sw,cont)))
    if scale<=0:raise ValueError('Nonpositive comparison normalization')
    lines=line_catalog(.003793,fw,t)
    for row in lines:
        if row['line']=='H I Pa alpha':row['measurement_status']='Published NGC 1068 detection (May & Steiner 2017), but archive profile is not telluric-corrected; marker is not a fitted centroid'
        if row['line'] in ('He I','He II','Fe II blend'):row['source_url']='https://gitlab.nublado.org/cloudy/cloudy';row['measurement_status']='Model transition only; not claimed as a detection in this X-shooter spectrum'
    geometry=make_figures(wave,flux,error,good,header,sw,syn,scale,fw,t,lines)
    used=(wave>=17500)&(wave<=20020)
    with (OUT/'measured_spectral_samples.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['heliocentric_vacuum_um','flux_erg_s_cm2_A','statistical_error_erg_s_cm2_A','original_quality_flag','retained_for_plot']);w.writerows(zip(wave[used]/1e4,flux[used],error[used],qual[used],good[used]))
    with (OUT/'line_guides.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(lines[0]));w.writeheader();w.writerows(lines)
    np.savez_compressed(OUT/'plotted_data.npz',observed_heliocentric_vacuum_angstrom=wave,observed_flux=flux,observed_statistical_error=error,original_quality=qual,observed_valid=good,synthetic_wave=sw,synthetic_flux_scaled=syn*scale,synthetic_continuum_scaled=cont*scale,filter_wave=fw,throughput=t)
    manifest=dict(status='Actual archived observed samples; not telluric-corrected, not an intrinsic-spectrum fit',
        source_file=str(OBS.relative_to(ROOT)),source_url=ARCHIVE,source_sha256=hashlib.sha256(OBS.read_bytes()).hexdigest(),
        synthetic_source=str(SYN.relative_to(ROOT)),synthetic_source_sha256=hashlib.sha256(SYN.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        date_obs=header['DATE-OBS'],object=header['OBJECT'],program=header['ESO OBS PROG ID'],
        slit=header['ESO QC SLITW'],spectral_resolving_power=header['SPEC_RES'],
        wavelength_input={'column':'WAVE','units':tabhead['TUNIT1'],'ucd':tabhead['TUCD1'],'frame':header['SPECSYS']},
        wavelength_output='Heliocentric vacuum Angstrom; air-to-vacuum Peck & Reeder convention, plus lambda*(1+heliocentric_correction/c)',
        heliocentric_correction_kms=header['ESO QC VRAD HELICOR'],
        flux_density_units=tabhead['TUNIT2'],density_jacobian_applied=True,
        quality_selection='QUAL==0, finite wavelength/flux/error, error>0; negative valid fluxes retained; no SNR cut',
        source_valid_in_zoom=int(np.sum(good&(wave>18580)&(wave<19050))),
        source_total_in_zoom=int(np.sum((wave>18580)&(wave<19050))),
        telluric_correction_applied=False,release_document=RELEASE,
        model_scale=scale,normalization_window_vacuum_um=[1.985,1.995],anchor_points=int(anchor.sum()),
        normalization='One scalar matches median observed flux to modeled continuum in the marked sideband; does not fit emission lines or remove atmospheric effects',
        model_blur='Gaussian LSF at Pa alpha with R from FITS; constant sigma over this narrow plot, not a measured full LSF',
        model_smoothing_sigma_fine_pixels=sigma,identical_pair_axes=geometry,
        limitations=['No telluric correction or validated intrinsic Pa-alpha flux/profile from this product','One archival slit extraction versus full synthetic field; no aperture/seeing/registration match','Flux standard calibration and slit losses do not establish an absolute full-galaxy flux','Expected He I/He II/Fe II markers are model transitions, not confirmed detections','Current synthetic grid convergence/inventory and assumed continuum/amplitude limitations remain','No observed samples or invalid gaps were synthesized or filled'])
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (OUT/'README.md').write_text(f'''# Corrected pair: measured NGC 1068 spectra around F187N

Use `01_measured_ngc1068_labeled` and `02_measured_ngc1068_with_synthetic` (PNG/PDF/SVG). Both use the exact same measured samples, wavelength axis, statistical errors, quality mask, line guides and display limits. The only plotted data addition in the second figure is the current synthetic model, with one recorded scale and approximate instrumental spectral broadening.

## What has changed

The earlier `f187n_spectral_reference_v2` figures showed a continuum model constrained by published flux/EW proxies, not a sampled observed spectrum. They remain preserved for provenance. These new figures replace that reference with a genuine downloaded ESO X-shooter NIR FITS spectrum of NGC 1068, programme 109.22YE.001, observed {header['DATE-OBS']}. Archive product: [ADP.2022-08-12T12:26:18.491]({ARCHIVE}). The product contains measured WAVE, FLUX, ERR and QUAL columns. No line amplitudes or spectral pixels were invented.

## Critical limitation: Earth atmosphere

The [ESO X-shooter release description]({RELEASE}) says these archived products have not been corrected for telluric absorption. Strong water-vapor absorption affects the F187N range. This is genuine observed data, but it is NOT a correct intrinsic NGC 1068 spectral profile. The tan band is a broad caution interval, not a measured atmospheric transmission curve and not an applied correction. Comparing the blue above-atmosphere synthetic curve directly to the depressed observed flux inside that band cannot validate the intrinsic emission strengths. No in-band residual score or Pa-alpha flux is inferred here.

To obtain an intrinsic spectrum suitable for fitting we still need validated telluric correction, correct extraction aperture and seeing/registration matching. Deeply absorbed or statistically unconstrained samples may need to remain masked even after correction; they must not be filled using the galaxy model and called observations. Published line-flux measurements remain separate constraints, with their own telluric caveats.

## Wavelength and flux audit

The table WAVE column is in nm, its UCD is `em.wl;obs.atmos` (air), and SPECSYS is TOPOCENT. We convert air wavelengths to vacuum using the repository's Peck & Reeder convention, then apply the stored heliocentric correction ({header['ESO QC VRAD HELICOR']:.6f} km/s). The observed axis is heliocentric vacuum, consistent with interpreting the adopted z=0.003793 marker convention heliocentrically. The change of wavelength variable includes the corresponding F_lambda/ERR density Jacobian. This is a documented wavelength convention, not a claimed correction for unknown source velocities or an independently fitted systemic redshift.

Only QUAL=0, finite samples with positive statistical error are plotted. Negative valid fluxes are retained on a symlog axis; no SNR cut is made. Bad-sample gaps are not interpolated. Error shading is the provided statistical error; it does not include telluric or slit-loss systematics. Raw FITS data, masks and transformed numerical samples are retained.

## Model overlay and labels

The overlay is the current image-model spectrum from `f187n_spectral_reference_v2/spectra.npz`, not the new converged AGN fitting pilot. Approximate Gaussian spectral blur uses the measured product's resolving power R={header['SPEC_RES']}. A single scalar normalizes its continuum to the median observed signal at 1.985–1.995 um, outside the main F187N band; this does not assert the sideband is perfectly telluric-free or fit the emission lines. The source model is the full 7.9-arcsec field; the observation is a 0.9-arcsec-wide slit extraction. No exact along-slit extraction/PSF registration was established, so the overlay is a qualitative illustration rather than a flux-calibrated or aperture-matched fit.

Pa alpha and Si VI are known NGC 1068 lines from published measurements. Asterisks explicitly distinguish modeled He I, He II and Fe II guide positions from observed detections; no Gaussian fit or detection claim has been made for them. Si VI lies outside the main F187N band. Wavelength guide values come from the current CLOUDY line inventory and adopted redshift, not from measured X-shooter centroids.

## Reproducibility

See manifest.json for product identity, checksums, units, transformations, masks, scale and display limits; measured_spectral_samples.csv and plotted_data.npz preserve the plotted numbers. line_guides.csv lists the marker provenance. The old reference figures and current galaxy images were not overwritten. Regenerate using `scripts/cloudy/plot_ngc1068_observed_f187n.py` with a fresh OUT path.
''')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
