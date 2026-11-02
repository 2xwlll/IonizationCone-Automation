#!/usr/bin/env python3
"""Freeze the current model and screen it against real NGC 1068 constraints.

No fitting, image renormalization, new CLOUDY runs, or invented observed spectra.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR','/tmp/agn-validation-matplotlib')
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm, TwoSlopeNorm

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.jwst_utils.nircam import convolve_kernel
from src.simple_utils.f187n import printed_emergent_lines
from src.simple_utils.wavelengths import cloudy_air_to_vacuum

OUT=ROOT/'results/cloudy/ngc1068_current_model_validation_v1'
SOURCE=ROOT/'data/spectra/ngc1068_may_steiner2017_v1'
URL='https://academic.oup.com/mnras/article/469/1/994/3586652'


def import_observations():
    if SOURCE.exists(): raise FileExistsError(SOURCE)
    SOURCE.mkdir(parents=True)
    rows=[]
    # Manual numerical transcription, Table 3. Missing values stay blank, not zero.
    data={'A':(.1,[30.75,.03,42.,2.,2.8,.3]),
          'B':(.1,[.92,.08,None,None,.05,.02]),
          'NE':(.1,[6.06,.04,11.46,.08,1.30,.03]),
          'S1':(.25,[70.,30.,106.,5.,48.,1.])}
    for region,(radius,values) in data.items():
        for j,(name,nominal) in enumerate([('pa_alpha',18751),('si_vi',19641),('br_gamma',21661)]):
            rows.append(dict(region=region,aperture_radius_arcsec=radius,line_id=name,
                published_nominal_vacuum_angstrom=nominal,flux_1e15=values[j*2],
                reported_error_1e15=values[j*2+1],status='not reported' if values[j*2] is None else 'reported detection'))
    path=SOURCE/'selected_table3_lines.csv'
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    manifest=dict(source='May & Steiner 2017, MNRAS 469, 994, Table 3 and Sections 2, 3.3',
        doi='10.1093/mnras/stx886',url=URL,import_method='Manual transcription of selected published line fluxes; not sampled observed spectra or digitized profiles',
        flux_unit='1e-15 erg s^-1 cm^-2',
        nuclear_aperture='S1 circular radius 0.25 arcsec; A/B/NE radius 0.1 arcsec',
        observations='SINFONI H+K, 1.45-2.45 um, R~2400, ESO programme 076.B-0098(A), 2005/2006; processed/deconvolved cubes, not the 2024 JWST image',
        pa_alpha_warning='Authors report strong atmospheric absorption, and spectrally Butterworth-filter Pa alpha before measuring. Nuclear flux uncertainty is 30/70; no claim that this is a clean line profile.',
        wavelength_warning='Table heading says vacuum but nominal Pa alpha identifier is 18751 A. Preserve source identifiers; use existing CLOUDY air-to-vacuum wavelengths in model calculations, not those rounded identifiers.',
        limitations=['No raw spectral pixels or calibrated spectral cube downloaded','No precise A/B/NE spatial registration undertaken','Reported errors do not establish independent calibration or telluric covariance','PSF and registration cannot be matched exactly from a table alone'],
        sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (SOURCE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return rows


def line_maps(s,cfg,g,h):
    """Integrated energy maps before filter weighting; preserve original normalizations."""
    manifest=json.loads((ROOT/'data/cloudy/ngc1068_f187n_pilot_v1/manifest.json').read_text())
    by={r['model_id']:r for r in manifest['sources']}
    br=[];raw=[]
    for mid in g['model_id']:
        path=ROOT/by[str(mid)]['source_out'];table=printed_emergent_lines(path.read_text())
        br.append(table.get(('H 1',21655.3),np.nan))
        raw.append(dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    br=np.asarray(br)
    if not np.isfinite(br).all(): raise ValueError('Missing Br gamma diagnostics must not silently become zero')
    hpath=ROOT/cfg['star_forming']['grid'];mid=str(h['model_id'][cfg['star_forming']['model_index']])
    raw_h=hpath.parent/'runs'/f'{mid}.out'
    hbr=printed_emergent_lines(raw_h.read_text())[('H 1',21655.3)]
    raw.append(dict(path=str(raw_h.relative_to(ROOT)),sha256=hashlib.sha256(raw_h.read_bytes()).hexdigest()))
    maps={};missing={}
    for name,air in [('pa_alpha',18751.),('si_vi',19624.7),('br_gamma',21655.3)]:
        j=int(np.argmin(abs(g['rest_line_angstrom']-air)))
        flux=br if name=='br_gamma' else g['line_flux'][:,j]
        rest=float(cloudy_air_to_vacuum(air))
        total=np.zeros_like(s['tau']);unavailable=np.zeros_like(total)
        for pop in ('cone','host'):
            index=s[pop+'_index'];velocity=s['velocity' if pop=='cone' else 'host_velocity']
            weight=s[pop+'_weight']
            screen=np.exp(-s['tau']*(rest*(1+velocity/299792.458)/18751.)**(-cfg['components']['dust_exponent']))
            total+=weight*np.nan_to_num(flux,nan=0)[index]*screen
            unavailable+=weight*~np.isfinite(flux[index])
        if name in ('pa_alpha','br_gamma'):
            hflux=h['line_flux'][cfg['star_forming']['model_index'],0] if name=='pa_alpha' else hbr
            screen=np.exp(-s['tau']*(rest*(1+s['hii_velocity']/299792.458)/18751.)**(-cfg['components']['dust_exponent']))
            total+=s['hii_spatial']*float(s['hii_normalization'])*hflux*screen
        # No H II Si VI line is present in the pilot list; do not invent one.
        maps[name]=total;missing[name]=unavailable
    return maps,missing,raw


def line_screen(s,cfg,g,h,obs):
    maps,missing,raw=line_maps(s,cfg,g,h)
    scale=.03074246911919934/cfg['oversample'];y,x=np.indices(s['tau'].shape)
    cx=(x.shape[1]-1)/2;cy=(x.shape[0]-1)/2
    experiments=[('existing_NIRCam_PSF',None,0.,0.),('Gaussian_0p12_arcsec',.12,0.,0.),('Gaussian_0p16_arcsec',.16,0.,0.)]
    experiments += [(f'Gaussian_0p16_center_{axis}_{offset:+.2f}',.16,offset if axis=='x' else 0.,offset if axis=='y' else 0.) for axis in ('x','y') for offset in (-.05,.05)]
    scenarios=[]
    for name,fwhm,dx,dy in experiments:
        aperture=(np.hypot((x-cx)*scale-dx,(y-cy)*scale-dy)<.25).astype(float)
        acceptance=(convolve_kernel(aperture,s['kernel'][::-1,::-1]) if fwhm is None else gaussian_filter(aperture,fwhm/2.354820045/scale,mode='constant'))/cfg['oversample']**2
        sums={k:float(np.sum(v*acceptance)) for k,v in maps.items()}
        unavailable=float(np.sum(acceptance*missing['si_vi'])/np.sum(acceptance*(s['cone_weight']+s['host_weight'])))
        scenarios.append(dict(name=name,psf_fwhm_arcsec=fwhm,center_offset_arcsec=[dx,dy],
            integrated_relative_line_energy=sums,si_vi_over_br_gamma=sums['si_vi']/sums['br_gamma'],
            pa_alpha_over_br_gamma=sums['pa_alpha']/sums['br_gamma'],
            fraction_agn_spatial_weight_with_censored_si_vi=unavailable))
    nucleus={r['line_id']:r for r in obs if r['region']=='S1'}
    ratios={}
    for line in ('si_vi','pa_alpha'):
        a=nucleus[line];b=nucleus['br_gamma'];value=a['flux_1e15']/b['flux_1e15']
        sigma=value*np.hypot(a['reported_error_1e15']/a['flux_1e15'],b['reported_error_1e15']/b['flux_1e15'])
        ratios[line+'_over_br_gamma']=dict(value=value,reported_error_if_independent=float(sigma))
    result=dict(observed_nuclear_ratios=ratios,model_scenarios=scenarios,raw_sources=raw,
        interpretation='Fixed-model screening discrepancy, not formal rejection significance. Radius is matched; actual SINFONI PSF, deconvolution and registration are not replicated. No line amplitude fitted.',
        caveats=['Pa alpha has strong atmospheric absorption and was filtered in the paper; prioritize Si VI/Br gamma for this screen',
            'Missing/censored Si VI is zero-filled only to reproduce the implemented image model, not treated as measured physical absence',
            'Br gamma is recovered from existing rounded CLOUDY printed output, not a new gas model or newly fitted component',
            'No H II Si VI line exists in current pilot inventory','Nuclear emission includes prescribed H II placement and gas amplitudes; errors cannot be assigned only to AGN physics'])
    fig,ax=plt.subplots(figsize=(9,6));xx=np.arange(2)
    vals=[ratios[k]['value'] for k in ('si_vi_over_br_gamma','pa_alpha_over_br_gamma')]
    err=[ratios[k]['reported_error_if_independent'] for k in ('si_vi_over_br_gamma','pa_alpha_over_br_gamma')]
    ax.errorbar(xx-.08,vals,yerr=err,fmt='o',ms=8,color='black',capsize=5,label='Published S1 nuclear measurements')
    for i,(scenario,color) in enumerate(zip(scenarios[:3],['#007fa3','#cb7530','#7656a7'])):
        ax.plot(xx+.06+i*.06,[scenario[k] for k in ('si_vi_over_br_gamma','pa_alpha_over_br_gamma')],'s',ms=7,color=color,label=scenario['name'].replace('_',' '))
    ax.set(yscale='log',ylim=(.2,20),xticks=xx,xticklabels=['[Si VI] / Br γ','Pa α / Br γ'],ylabel='Integrated line-energy ratio',xlim=(-.5,1.5))
    ax.grid(axis='y',alpha=.2);ax.legend(fontsize=10,loc='upper left')
    ax.set_title('Real NGC 1068 line ratios versus the unchanged current model\nCentral aperture radius 0.25 arcsec; approximate PSF checks',fontsize=13)
    ax.text(1,.27,'Pa α is affected by atmospheric absorption;\nits discrepancy needs particular caution.',ha='center',fontsize=10,color='#7b4a40')
    fig.subplots_adjust(bottom=.18,top=.85)
    fig.text(.08,.05,'May & Steiner 2017, Table 3. No spectral pixels imported. Different epoch and incompletely matched PSF/registration.\nRatios were not used to fit the current image. Model sensitivity points are checks, not a confidence interval.',fontsize=9)
    for ext in ('png','pdf'):fig.savefig(OUT/f'line_ratio_screen.{ext}',dpi=180)
    plt.close(fig)
    return result


def image_screen():
    p=np.load(ROOT/'results/cloudy/ngc1068_f187n_soft_gas_v1/image_mask.npz')
    r=np.load(ROOT/'results/cloudy/f187n_reference/nucleus_cutout.npz')
    meta=json.loads(str(r['metadata']));scale=meta['pixel_scale_arcsec'];cy,cx=meta['nucleus_cutout_xy'][1],meta['nucleus_cutout_xy'][0]
    y,x=np.indices(p['clean'].shape);radius=np.hypot(x-cx,y-cy)*scale
    obs=r['science'].astype(float);model=p['clean'];valid=r['valid']&np.isfinite(obs)&np.isfinite(model)
    use=valid&(radius>.3)&(radius<3.7);residual=model-obs
    a,b=obs[use],model[use]
    result=dict(region='0.3 < nuclear radius < 3.7 arcsec; invalid observed pixels excluded',pixels=int(use.sum()),
        median_absolute_fractional_difference=float(np.median(abs(b-a)/a)),median_model_over_observed=float(np.median(b/a)),
        residual_rms_mjysr=float(np.sqrt(np.mean((b-a)**2))),pearson_correlation=float(np.corrcoef(a,b)[0,1]),
        observed_outer_baseline_mjysr=meta['outer_median_science_mjysr'],
        limitations=['Current model amplitude and baseline already use this reference image; amplitude agreement is not independent validation',
            'Outer baseline contains galaxy light, not a measured empty sky','No registration, PSF or morphology fitting performed here',
            'ERR alone omits correlated errors and model uncertainty; no chi-square goodness probability is reported',
            'Central 0.3 arcsec excluded from metrics; invalid core values are not filled or treated as zero'],
        source_fits=meta['source_fits'])
    vmax=float(np.percentile(obs[use],99.5));extent=np.array([-128,128,-128,128])*scale
    fig,axes=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
    cmap=plt.colormaps['magma'].copy();cmap.set_bad('#bcbcbc')
    for ax,data,title in zip(axes[:2],[obs,model],['Observed JWST F187N','Current synthetic F187N (clean)']):
        im=ax.imshow(np.ma.array(data,mask=~valid),origin='lower',extent=extent,cmap=cmap,norm=AsinhNorm(linear_width=100,vmin=0,vmax=vmax))
        ax.set_title(title);ax.set_xlabel('Detector x offset (arcsec)');ax.set_ylabel('Detector y offset (arcsec)')
    fig.colorbar(im,ax=list(axes[:2]),label='MJy/sr (same display scale)',shrink=.8)
    limit=float(np.percentile(abs(residual[use]),98))
    cm=plt.colormaps['RdBu_r'].copy();cm.set_bad('#bcbcbc')
    im=axes[2].imshow(np.ma.array(residual,mask=~use),origin='lower',extent=extent,cmap=cm,norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit))
    axes[2].set(title='Residual: model − observation',xlabel='Detector x offset (arcsec)',ylabel='Detector y offset (arcsec)')
    fig.colorbar(im,ax=axes[2],label='Residual MJy/sr',shrink=.8)
    fig.suptitle('Same field and sampling; source-image amplitude and baseline already informed the model\nResiduals quantify remaining disagreement, not independent flux validation',fontsize=12)
    for ext in ('png','pdf'):fig.savefig(OUT/f'image_residuals.{ext}',dpi=180)
    plt.close(fig)
    edges=np.linspace(.3,3.7,18);rows=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=valid&(radius>=lo)&(radius<hi)
        rows.append([float((lo+hi)/2),int(mask.sum()),float(obs[mask].mean()),float(model[mask].mean())])
    with (OUT/'radial_profile.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['radius_arcsec','valid_pixels','observed_mean_mjysr','model_mean_mjysr']);w.writerows(rows)
    result['radial_profile']=rows
    return result


def main():
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    observations=import_observations()
    s=dict(np.load(ROOT/'results/cloudy/ngc1068_f187n_spectral_export_v1/spectral_state.npz'))
    cfg=json.loads(str(s['configuration']));g=dict(np.load(ROOT/cfg['grid']));h=dict(np.load(ROOT/cfg['star_forming']['grid']))
    image=image_screen();line=line_screen(s,cfg,g,h,observations)
    primary=line['model_scenarios'][2]
    obs=line['observed_nuclear_ratios']['si_vi_over_br_gamma']['value'];pred=primary['si_vi_over_br_gamma']
    report=dict(status='Current model is not observationally validated; screening detects image and line-ratio disagreement',
        no_fitting_or_model_changes=True,image=image,line_ratios=line,
        primary_si_vi_br_gamma_observed_over_model=obs/pred,
        spectroscopy='New real line fluxes imported; no clean sampled spectrum across F187N obtained. SINFONI programme 076.B-0098(A) is a concrete archive lead, not a downloaded/reduced product.',
        current_spectrum_identity='Prior spectral export reproduces the current image components; that is internal consistency, separate from the real-data checks here.')
    (OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    paths=[Path(__file__),ROOT/'results/cloudy/ngc1068_f187n_soft_gas_v1/image_mask.npz',ROOT/'results/cloudy/f187n_reference/nucleus_cutout.npz',ROOT/'results/cloudy/ngc1068_f187n_spectral_export_v1/spectral_state.npz',ROOT/cfg['grid'],ROOT/cfg['star_forming']['grid'],SOURCE/'selected_table3_lines.csv']
    provenance=dict(sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},published_source=URL,original_cloudy_outputs=line['raw_sources'])
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    text=f'''# Check of the unchanged current NGC 1068-inspired model

The model does not yet pass observational validation. No amplitudes, geometry, gas models, noise settings or training labels were changed to improve these comparisons. No new CLOUDY jobs were run.

## F187N image check

The real JWST image and the current clean synthetic image were compared at their existing 256 x 256 sampling. Finite valid observed pixels in the annulus 0.3 < r < 3.7 arcsec were used ({image['pixels']} pixels). The median absolute fractional pixel difference is {image['median_absolute_fractional_difference']:.1%}; the median model/observed ratio is {image['median_model_over_observed']:.3f}; residual RMS is {image['residual_rms_mjysr']:.1f} MJy/sr. Pearson correlation is {image['pearson_correlation']:.3f}.

These are descriptive checks, not a calibrated likelihood or proof that the cone is recovered. The model's amplitude and added baseline were already chosen using this same observation. Matching the brightness range therefore is not independent evidence. The outer baseline includes galaxy light. The model morphology and registration have not been fitted. Invalid observed pixels are masked, and the central 0.3 arcsec is excluded from the numerical metrics.

## Newly located real emission-line measurements

[May & Steiner (2017), Table 3 and Sections 2 and 3.3]({URL}) report Paschen-alpha, [Si VI] and Brackett-gamma fluxes in NGC 1068, including a central S1 aperture of radius 0.25 arcsec. A selected numerical transcription is saved in `data/spectra/ngc1068_may_steiner2017_v1`. It is line-flux data, not sampled spectral pixels. A/B/NE measurements are retained as observations but not treated as positions of our illustrative synthetic clumps.

For S1, [Si VI] = (106 +/- 5), Br gamma = (48 +/- 1), and Pa alpha = (70 +/- 30), in units of 1e-15 erg s^-1 cm^-2. The authors specifically warn that atmospheric absorption affects Pa alpha and apply spectral filtering before measuring it. Its uncertainty and systematic issues prevent treating it as a clean measured profile. The 2010 source's unusable Pa alpha does not mean all NGC 1068 observations lack Pa alpha measurements.

The nuclear [Si VI]/Br gamma ratio is {obs:.3f}. Our current model gives {pred:.3f} using a 0.25-arcsec aperture and an illustrative 0.16-arcsec Gaussian PSF: a factor {obs/pred:.2f} lower. Existing NIRCam PSF, 0.12-arcsec and 0.16-arcsec Gaussian checks and +/-0.05-arcsec centre shifts are in validation.json. They test sensitivity; they are not a posterior or exact SINFONI reduction match. No line ratios were fitted.

The model ratio uses existing AGN gas spectra and the current H II placement/normalization. Br gamma is recovered from the same preserved CLOUDY raw outputs used for these populations, not from the newer converged AGN fitting pilot. It is not an F187N contribution; Br gamma and [Si VI] are external spectral diagnostics of the gas used in the model. All 81 AGN Br gamma entries are present in the printed output. Rounding and original convergence warnings remain. [Si VI] censored entries are zero-filled to reproduce the implemented model; this is not evidence of physical zero flux. The model H II line list does not contain Si VI.

The line-ratio screen indicates that the fixed gas mixture/parameters are not an established match to NGC 1068. It does not uniquely identify a density, abundance, shock or atomic-data cause. Different epoch, PSF, deconvolution and uncertain registration remain limitations. In particular, we do not convert a ratio discrepancy into a formal rejection significance.

## What the check establishes

- Our plotted synthetic spectrum is internally consistent with the existing images.
- Actual F187N morphology still differs from the current model.
- New independent line-flux constraints reveal an excitation-ratio mismatch worth addressing before training claims.
- A clean, sampled observed F187N-range spectrum still has not been obtained. The SINFONI H+K data in ESO programme 076.B-0098(A), identified in the 2017 paper, provide a specific route to investigate; acquisition/reduction and telluric-quality assessment remain undone.

Recommended next step: use a complete converged AGN line grid, constrain its gas mixture with the external diagnostics, and obtain/assess an aperture-matched spectral product before replacing the reference continuum/line atlas with an observed spectral curve. Keep the current images labeled illustrative during that work.

Outputs: image_residuals.png/.pdf, line_ratio_screen.png/.pdf, radial_profile.csv, validation.json and provenance.json. Published line measurements are preserved in a separate source folder; prior images and spectral figures remain unchanged.
'''
    (OUT/'README.md').write_text(text)
    print(json.dumps(dict(image=image,primary_nuclear_model=primary,observed_si_vi_br_gamma=obs,observed_over_model=obs/pred),indent=2))


if __name__=='__main__':main()
