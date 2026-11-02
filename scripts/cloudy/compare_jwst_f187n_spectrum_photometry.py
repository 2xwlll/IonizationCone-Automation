"""Compare the unchanged synthetic spectrum with real JWST photometry only.

One imaging filter supplies a band integral, never a measured spectral curve.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR','/tmp/agn-jwst-photometry-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.cloudy.plot_f187n_spectral_reference import reconstruct,line_catalog
from src.simple_utils.synthetic_photometry import photon_bandpass_weights

OUT=ROOT/'results/cloudy/ngc1068_jwst_only_comparison_v2'


def main():
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    state_path=ROOT/'results/cloudy/ngc1068_f187n_spectral_export_v1/spectral_state.npz'
    image_path=ROOT/'results/cloudy/ngc1068_f187n_soft_gas_v1/image_mask.npz'
    reference_path=ROOT/'results/cloudy/f187n_reference/nucleus_cutout.npz'
    state=dict(np.load(state_path));image=np.load(image_path);reference=np.load(reference_path)
    cfg=json.loads(str(state['configuration']));meta=json.loads(str(reference['metadata']))
    image_meta=json.loads(str(image['metadata']))
    y,x=np.indices(image['clean'].shape);cx,cy=meta['nucleus_cutout_xy'];radius=np.hypot(x-cx,y-cy)*meta['pixel_scale_arcsec']
    mask=reference['valid']&(radius>.3)&(radius<3.7)&np.isfinite(image['clean'])
    n=int(mask.sum())
    obs_mean=float(reference['science'][mask].mean())
    model_mean=float(image['clean'][mask].mean())
    baseline=float(image_meta['galaxy_plus_background_baseline_mjysr'])
    obs_stat=float(np.sqrt(np.sum(reference['error'][mask]**2))/n)
    grid=dict(np.load(ROOT/cfg['grid']));hii=dict(np.load(ROOT/cfg['star_forming']['grid']))
    fw=grid['filter_wavelength_angstrom'];throughput=grid['throughput']
    wave=np.arange(fw[0],fw[-1]+.05,.1)
    spectra,continua=reconstruct(state,cfg,grid,hii,wave,aperture=mask)
    weights=photon_bandpass_weights(wave,fw,throughput)
    coarse=photon_bandpass_weights(fw,fw,throughput)
    checks={}
    amplitude=image_meta['empirical_scale']
    for name,s in spectra.items():
        pred=float(coarse@continua[name]) if name in continua else float(weights@s)
        measured=float(image[name][mask].sum())/amplitude
        err=abs(pred/measured-1)
        if err>3e-4:raise AssertionError((name,err))
        checks[name]=dict(relative_roundtrip_error=err)
    total=sum(spectra.values());continuum=sum(spectra[k] for k in continua)
    norm=float(np.interp(18740.,wave,continuum))
    catalog=line_catalog(cfg['redshift'],fw,throughput)
    report=dict(status='JWST-only measured filter-integral comparison; no observed F187N spectrum inferred',
        observed_mean_mjysr=obs_mean,current_model_mean_mjysr=model_mean,
        observed_formal_statistical_mean_error_ignoring_covariance_mjysr=obs_stat,
        model_minus_observed_mean_mjysr=model_mean-obs_mean,model_over_observed_mean=model_mean/obs_mean,
        shared_empirical_baseline_mjysr=baseline,model_source_mean_mjysr=model_mean-baseline,
        observed_mean_minus_assumed_baseline_mjysr=obs_mean-baseline,
        valid_pixels=n,aperture='Same observed valid detector pixels; 0.3 < nuclear radius < 3.7 arcsec',
        observed_quantity='Mean JWST SCI in calibrated MJy/sr; one band-equivalent surface-brightness measurement, not F_lambda at each wavelength',
        synthetic_spectrum='Same aperture applied through adjoint PSF; source components only, no added baseline or noise; continuum normalized to one at 1.874 um',
        photometry='Both means include the existing model/observation baseline. No new scaling/fitting; model mean from unchanged saved clean image.',
        limitations=['Model scale and baseline already informed by the same F187N observation; this is descriptive agreement, not independent flux validation',
            'SCI calibration and spectral color assumptions are not independently recalibrated here','ERR mean formula ignores resampling covariance/systematics; not used for a fit significance',
            'Spectral shape of the real galaxy is unknown from this one band','No wavelength distribution assigned to the added empirical baseline',
            'Existing synthetic model convergence, line-list, geometry and continuum assumptions persist'],
        component_roundtrip_checks=checks,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (state_path,image_path,reference_path,Path(__file__))})
    (OUT/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(OUT/'synthetic_aperture_spectrum.npz',observed_vacuum_angstrom=wave,
        synthetic_relative_flambda=total,synthetic_continuum=continuum,aperture_mask=mask,
        filter_wave=fw,throughput=throughput,continuum_normalization=norm,**spectra)
    plt.rcParams.update({'font.size':12,'axes.labelsize':14})
    ylim=(0,float(np.max(total[(wave>18580)&(wave<19050)])/norm)*1.20)
    for compare in (False,True):
        fig=plt.figure(figsize=(13,10))
        gs=fig.add_gridspec(3,1,height_ratios=[.9,3.2,1.9],hspace=.50)
        label_ax,ax,band=[fig.add_subplot(gs[i]) for i in range(3)]
        fig.subplots_adjust(left=.10,right=.90,top=.84,bottom=.21)
        fig.suptitle('NGC 1068 — synthetic spectrum'+(' and real JWST measurement' if compare else ' near F187N'),fontsize=20,y=.975)
        fig.text(.5,.925,'JWST/NIRCam F187N: one measured band integral, not a wavelength-resolved observed spectrum',ha='center',fontsize=12)
        label_ax.set(xlim=(1.858,1.905),ylim=(0,3.8));label_ax.axis('off')
        positions=[(1.8665,2.0),(1.8718,3.25),(1.8766,1.55),(1.8804,3.25),(1.8864,2.0),(1.893,3.25),(1.900,2.0)]
        for row,(tx,ty) in zip(catalog[:7],positions):
            xx=row['observed_vacuum_um'];color='#b85c08' if row['line']=='H I Pa alpha' else '#675c85'
            label_ax.annotate(row['line'].replace('Pa alpha','Pa α')+'\n'+f'{xx:.5f} μm',xy=(xx,.03),xytext=(tx,ty),ha='center',va='center',fontsize=10,color=color,arrowprops=dict(arrowstyle='-',color=color,lw=.8))
            ax.axvline(xx,color=color,alpha=.2,lw=.7)
        ax.plot(wave/1e4,total/norm,color='#007eae',lw=1.8,label='Current synthetic source spectrum (same selected pixels)')
        ax.set(xlim=(1.858,1.905),ylim=ylim,xlabel='Observed-frame vacuum wavelength (μm)',ylabel='Relative model $F_\lambda$\n(continuum = 1 at 1.874 μm)')
        ax.grid(alpha=.15);ax.legend(loc='upper left',fontsize=10)
        ax.text(.02,.76,'Labels mark model transitions,\nnot observed detections.',transform=ax.transAxes,fontsize=10)
        twin=ax.twinx();twin.fill_between(fw/1e4,throughput/max(throughput),alpha=.12,color='#768293')
        twin.set(ylim=(0,1.05),ylabel='F187N throughput / peak');twin.set_zorder(ax.get_zorder()-1);ax.patch.set_visible(False)
        if compare:
            band.plot([0],[obs_mean],'o',ms=10,color='black',label='Real JWST F187N SCI')
            band.plot([1],[model_mean],'s',ms=9,color='#007eae',label='Unchanged current model')
            for xx,val in [(0,obs_mean),(1,model_mean)]:band.annotate(f'{val:.1f} MJy/sr',(xx,val),xytext=(0,12),textcoords='offset points',ha='center',fontsize=12)
            band.set(xticks=[0,1],xticklabels=['Real JWST F187N image','Synthetic F187N image'],xlim=(-.6,1.6),ylim=(0,max(obs_mean,model_mean)*1.35),ylabel='Mean surface brightness\n(MJy/sr)')
            band.grid(axis='y',alpha=.2)
            band.set_title('One measurement per image over the same valid aperture; no observed spectral curve',fontsize=12)
        else:
            band.plot(fw/1e4,throughput/max(throughput),color='#768293',lw=1.5)
            band.fill_between(fw/1e4,throughput/max(throughput),alpha=.12,color='#768293')
            si=catalog[7]['observed_vacuum_um'];band.axvline(si,color='#b52727',alpha=.6)
            band.text(si,.75,'[Si VI]\noutside main F187N band',ha='right',fontsize=10,color='#b52727')
            band.set(xlim=(1.80,2.00),ylim=(0,1.1),xlabel='Observed-frame vacuum wavelength (μm)',ylabel='Throughput / peak')
        fig.text(.10,.09,'Same selected pixels: valid SCI, 0.3–3.7 arcsec annulus; spectrum includes the current PSF and prescribed gas velocities.',fontsize=10)
        fig.text(.10,.06,'Model spectral curve excludes the added empirical baseline; both image measurements include it. No spectral shape is invented for that baseline.',fontsize=10)
        fig.text(.10,.03,'The model amplitude/baseline already used this JWST image. This is a descriptive band check, not independent spectral validation.',fontsize=10,color='#794b40')
        name='02_model_spectrum_and_jwst_photometry' if compare else '01_current_synthetic_spectrum'
        for ext in ('png','pdf','svg'):fig.savefig(OUT/f'{name}.{ext}',dpi=200)
        plt.close(fig)
    (OUT/'README.md').write_text(f'''# JWST-only comparison available with current data

The actual JWST F187N image provides one band-integrated measurement per image pixel. It does not provide a resolved spectrum. No ground-based observations, continuum-proxy reference curve or fabricated observed spectral peaks are used here.

Figure 01 shows the current synthetic source spectrum and the official F187N response. Figure 02 repeats that spectrum and compares the actual JWST F187N mean surface brightness with the unchanged synthetic image, using exactly the same observed valid pixels in the 0.3–3.7 arcsec annulus. Source spectra are re-extracted through the current PSF for that mask, not taken from the full galaxy and compared to another aperture.

Observed SCI mean = {obs_mean:.3f} MJy/sr. Model clean mean = {model_mean:.3f} MJy/sr. Model/observed = {model_mean/obs_mean:.4f}. The numerical difference is not a new fit or independent calibration: the same real image already informed the model amplitude and empirical baseline. Central invalid pixels and the innermost 0.3 arcsec are excluded; the nuclear core is not validated by this test. The chosen 3.7-arcsec outer limit avoids the cutout edges.

The upper spectrum is normalized to model continuum at observed 1.874 um. It excludes the image's added empirical baseline, for which no wavelength distribution is known. The lower image measurements include that baseline. Accordingly the top curve's y axis is relative model F_lambda, and the bottom measurement is calibrated band-equivalent MJy/sr. The lower panel must not be interpreted as measuring that spectral curve point by point. Pixel-summed, photon-weighted spectral extraction agrees with every saved component map to the tolerances in comparison.json.

## Archive result

The fresh positional MAST query is saved in `results/cloudy/ngc1068_jwst_spectral_audit_v2`: 42 nearby JWST observation records, including 36 MIRI IFU cube records, 2 MIRI imaging records and 4 NIRCam imaging records. These are archive records, not counts of independent nuclear spectra, and include background observations. None of the returned spectral records spans F187N. The query returned no NIRSpec observation. This is a documented query result, not proof that no such observation exists anywhere or will become available.

The MIRI spectra are genuine JWST spectroscopy at longer wavelengths. They could constrain the broader AGN model in a separate mid-infrared comparison, but cannot replace a 1.87-um observed spectrum. The current near-IR spectral model also does not yet constitute a validated model for the full MIRI range.

## What remains possible

- Compare real and synthetic F187N images and their band-integrated brightness now.
- Show the synthetic spectrum with the filter curve now.
- Make a measured spectral comparison in the F187N range when suitable space-based spectroscopy is obtained and its aperture, calibration and quality are checked.
- Infer a candidate real-galaxy spectrum under explicit assumptions if desired, but label it as an inferred model rather than a measured JWST spectrum. One filter cannot determine line strengths, continuum shape and velocities uniquely.

PNG, PDF and SVG copies are provided. comparison.json and synthetic_aperture_spectrum.npz contain the values and provenance. No model parameters, images or masks were changed. All original convergence, incomplete line-list, continuum and relative-amplitude caveats remain.
''')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
