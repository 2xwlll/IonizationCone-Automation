"""Compare independent corrected measurements with registered slit-model spectra."""
import argparse,csv,hashlib,json,os
from pathlib import Path
import sys
os.environ.setdefault('MPLCONFIGDIR','/tmp/agn-slit-plots')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.simple_utils.slit_aperture import sky_offsets
from scripts.cloudy.plot_ngc1068_corrected_f187n import broaden
from scripts.cloudy.plot_f187n_spectral_reference import line_catalog


def rectangle(g):
    along=np.array([-1,1,1,-1,-1])*g['length']/2+g['along_center']
    cross=np.array([-1,-1,1,1,-1])*g['width']/2+g['cross_center']
    a=np.deg2rad(g['pa_deg'])
    return np.c_[along*np.sin(a)+cross*np.cos(a),along*np.cos(a)-cross*np.sin(a)]


def save(fig,out,name):
    for ext in ['png','pdf','svg']:fig.savefig(out/(name+'.'+ext),dpi=180)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--extraction',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    folder=a.extraction.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    audit=json.loads((folder/'aperture_audit.json').read_text());extractions=json.loads((folder/'extraction_summary.json').read_text())
    obs_path=ROOT/'results/cloudy/ngc1068_telluric_dry_v1/corrected_spectrum.npz';obs=np.load(obs_path)
    correction=json.loads((obs_path.parent/'summary.json').read_text());r=correction['fitted_resolving_power']
    w=obs['heliocentric_vacuum_angstrom'];x=w/1e4;good=obs['usable'];unit=1e-15
    anchor=good&(w>=19850)&(w<=19950);line=good&(w>=18810)&(w<=18845)
    if anchor.sum()<10 or line.sum()<5:raise ValueError('Insufficient retained measurements')
    fullpath=ROOT/'results/cloudy/f187n_spectral_reference_v2/spectra.npz'
    model_sources={**{k:folder/(k+'.npz') for k in audit['scenarios']},'previous_full_field':fullpath}
    curves={};metrics={};continuum_scaled={}
    for name,path in model_sources.items():
        d=np.load(path);sw=d['observed_vacuum_angstrom'];fw=d['filter_wavelength_angstrom'];t=d['throughput']
        continuum=broaden(sw,d['synthetic_continuum'],r);model=broaden(sw,d['synthetic_relative_flambda'],r)
        scale=float(np.median(obs['corrected_flux'][anchor])/np.median(np.interp(w[anchor],sw,continuum)))
        curves[name]=model*scale;continuum_scaled[name]=continuum*scale
        sampled=np.interp(w,sw,curves[name])
        metrics[name]=dict(continuum_scale=scale,median_model_over_observation_retained_pa_window=float(np.median(sampled[line]/obs['corrected_flux'][line])),
            median_absolute_fractional_difference_retained_pa_window=float(np.median(abs(sampled[line]/obs['corrected_flux'][line]-1))))
    envelope_names=[k for k in audit['scenarios'] if k!='header_pointing_alternative']
    envelope=np.array([curves[k] for k in envelope_names]);low=envelope.min(axis=0);high=envelope.max(axis=0)
    plt.rcParams.update({'font.size':12,'axes.labelsize':13})
    # Show verified aperture placement on the actual images and the slit trace.
    reference=np.load(ROOT/'results/cloudy/f187n_reference/nucleus_cutout.npz');meta=json.loads(str(reference['metadata']))
    rendered=np.load(ROOT/'results/cloudy/ngc1068_f187n_soft_gas_v1/image_mask.npz')
    profile=np.load(folder/'slit_audit_data.npz');g=audit['scenarios']['registered']
    fig=plt.figure(figsize=(14,11));gs=fig.add_gridspec(2,2,height_ratios=[1.4,1],hspace=.32,wspace=.18)
    panels=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1])];trace=fig.add_subplot(gs[1,:])
    fig.subplots_adjust(left=.085,right=.93,top=.88,bottom=.16)
    for ax,title,data,center in [(panels[0],'Real JWST F187N image',np.where(reference['valid'],reference['science'],np.nan),meta['nucleus_cutout_xy']),
                               (panels[1],'Current synthetic F187N image',rendered['clean'],None)]:
        e,n=sky_offsets(data.shape,meta['pixel_scale_arcsec'],meta['pixel_to_sky_matrix'],center=center)
        im=ax.pcolormesh(e,n,data,norm=LogNorm(80,50000),cmap='magma',shading='nearest',rasterized=True)
        ax.add_patch(Polygon(rectangle(g),closed=True,fill=False,edgecolor='#56e4ed',lw=2,label='Registered 0.90″ × 3.99″ extraction'))
        ax.add_patch(Polygon(rectangle(audit['scenarios']['header_pointing_alternative']),closed=True,fill=False,edgecolor='#99d071',ls=':',lw=1.7,label='Header-pointing alternative'))
        ax.plot(0,0,'+',color='white',ms=11,mew=1.7)
        ax.set(xlim=(5,-5),ylim=(-4.8,5),aspect='equal',xlabel='East offset from nucleus (arcsec)',ylabel='North offset (arcsec)',title=title)
    panels[0].legend(loc='lower left',fontsize=8.5,facecolor='#222222',labelcolor='white')
    fig.colorbar(im,ax=panels,shrink=.8,pad=.025,label='F187N surface brightness (MJy/sr)')
    s=profile['spatial_arcsec'];f=profile['continuum_profiles'][0];frac=profile['good_sample_fractions'][0]
    reliable=frac>=.5
    trace.plot(s,np.where(reliable,f,np.nan),'o-',color='#222222',ms=4,label='Continuum profile: rows with ≥50% valid samples')
    trace.plot(s[~reliable],f[~reliable],'x',color='#a6462b',ms=7,label='Mostly flagged core rows: excluded from profile fit')
    trace.plot(s,np.where((s>-3.5)&(s<-1.5),profile['fitted_profile'],np.nan),'--',color='#ad7437',lw=1.4,label='Gaussian + local background fit to usable wings')
    trace.axvspan(*audit['actual_extraction_edges_arcsec'],color='#1fb1ba',alpha=.16,label='Actual 19-row extraction')
    trace.axvline(audit['continuum_peak_fits'][0]['center_arcsec'],color='#ad7437',lw=1)
    trace.set(xlim=(-5.4,5.5),ylabel='Median reduced continuum (ADU)',xlabel='Archived spatial coordinate along slit (arcsec)')
    trace.legend(fontsize=8.8,ncol=2);trace.grid(alpha=.15)
    fig.suptitle('Which part of NGC 1068 does this spectrum measure?',fontsize=21,y=.97)
    fig.text(.5,.925,'2023-08-24 X-shooter • PA = 38° east of north • Nuclear peak lies outside the archive extraction',ha='center',fontsize=12)
    fig.text(.085,.10,'Cyan rectangle: archive extraction registered using the near-IR continuum peak and nominal cross-slit coordinate. No emission-line alignment fit.',fontsize=10)
    fig.text(.085,.069,f"Effective compact width ≈ {g['fwhm']:.2f}″; tests use 0.40–0.80″ seeing. Core saturation limits this estimate. Dotted green: alternate header coordinates.",fontsize=10)
    fig.text(.085,.038,'Images retain their JWST PSF for orientation; spectral extraction uses intrinsic 2D components with a separate ground-based seeing operator.',fontsize=10)
    save(fig,out,'01_aperture_and_extraction_audit')
    # Observed data and atmospheric correction stay exactly as in the last result.
    y=np.where(good,obs['corrected_flux']/unit,np.nan);err=np.where(good,obs['corrected_error_envelope']/unit,np.nan)
    zoom=(1.858,1.905);sm=(sw/1e4>zoom[0])&(sw/1e4<zoom[1]);lines=line_catalog(.003793,fw,t)
    fig=plt.figure(figsize=(14,10));gs=fig.add_gridspec(3,1,height_ratios=[.8,3.5,1],hspace=.12)
    strip,ax,atm=[fig.add_subplot(gs[i]) for i in range(3)];fig.subplots_adjust(left=.09,right=.97,top=.82,bottom=.23)
    strip.set(xlim=zoom,ylim=(0,3.9));strip.axis('off')
    positions=[(1.866,2),(1.872,3.35),(1.877,1.65),(1.881,3.35),(1.886,2.8),(1.892,3.35),(1.900,2)]
    for row,(tx,ty) in zip(lines[:7],positions):
        pos=row['observed_vacuum_um'];label=row['line'].replace('Pa alpha','Pa α')+('' if row['line']=='H I Pa alpha' else '*')
        color='#a65b18' if row['line']=='H I Pa alpha' else '#76518f'
        strip.annotate(label+'\n'+f'{pos:.5f} μm',xy=(pos,.05),xytext=(tx,ty),ha='center',va='center',fontsize=10.5,color=color,arrowprops=dict(arrowstyle='-',color=color,lw=.8))
        ax.axvline(pos,color=color,alpha=.16,lw=.8)
    ax.fill_between(x,y-err,y+err,color='black',alpha=.18,linewidth=0)
    ax.plot(x,y,'.-',color='#222222',lw=.9,ms=2.7,label='Corrected measured samples; estimated error shading')
    ax.plot(sw/1e4,curves['previous_full_field']/unit,'--',color='#8c8c8c',lw=1.2,label='Previous full-field synthetic spectrum')
    ax.fill_between(sw/1e4,low/unit,high/unit,color='#008eaf',alpha=.15,linewidth=0,label='Position / seeing scenarios (not a confidence interval)')
    ax.plot(sw/1e4,curves['registered']/unit,color='#007faa',lw=1.8,label='Synthetic spectrum through registered slit + seeing')
    ax.plot(sw/1e4,curves['header_pointing_alternative']/unit,':',color='#5a8b37',lw=1.2,label='Separate header-pointing alternative')
    ymax=max(float(high[sm].max()),float(curves['previous_full_field'][sm].max()),float(curves['header_pointing_alternative'][sm].max()))/unit*1.2
    ax.set(xlim=zoom,ylim=(0,ymax),ylabel=r'$F_\lambda$ ($10^{-15}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)');ax.tick_params(labelbottom=False);ax.grid(alpha=.15)
    ax.legend(loc='upper left',fontsize=9,framealpha=.97)
    atm.plot(x,obs['transmission'],color='#a15d29',lw=1,label='Effective atmospheric transmission')
    atm.plot(fw/1e4,t/t.max(),'--',color='#007faa',lw=1.3,label='F187N throughput / peak')
    atm.fill_between(x,0,1,where=~good,color='#bbbbbb',alpha=.3,step='mid',linewidth=0)
    atm.set(xlim=zoom,ylim=(-.04,1.04),xlabel='Heliocentric vacuum wavelength (μm)',ylabel='Transmission');atm.legend(loc='upper left',fontsize=9,ncol=2)
    fig.suptitle('NGC 1068 — corrected spectrum versus slit-extracted synthetic galaxy',fontsize=19,y=.975)
    fig.text(.5,.932,'Same 2023 slit observation and atmospheric correction • Galaxy emission parameters unchanged',ha='center',fontsize=12)
    fig.text(.5,.893,f"Extraction: 0.90″ × 3.99″ at PA 38° • Nominal seeing: {g['fwhm']:.2f}″ • Registration remains approximate",ha='center',fontsize=11,color='#9a4526')
    fig.text(.09,.145,'Each model continuum is scaled once at 1.985–1.995 μm. No Pa-alpha strength, velocity or line centroid was fitted to the observation.',fontsize=10)
    fig.text(.09,.112,'Blue shading varies position by ±0.21″ along / ±0.30″ across the slit and seeing over 0.40–0.80″, plus the nominal-coordinate placement.',fontsize=10)
    fig.text(.09,.079,'Grey gaps remain rejected measurements. * = expected model transition, not a detection. Atmospheric error shading is not a calibrated confidence interval.',fontsize=10)
    fig.text(.09,.046,'Model field edges, uncertain cross-slit registration, continuum assumptions and residual atmosphere prevent a definitive line-flux validation.',fontsize=10)
    save(fig,out,'02_corrected_vs_slit_synthetic')
    # Compare component contributions to the photon-weighted band integral.
    groups={'AGN cone gas':['cone_pa_alpha','cone_other_lines','cone_nebular_continuum'],
        'Host gas':['host_pa_alpha','host_other_lines','host_nebular_continuum'],
        'Star-forming gas':['starforming_pa_alpha','starforming_other_lines','starforming_nebular_continuum'],
        'Stellar continuum':['stellar_continuum'],'Nuclear continuum':['nuclear_continuum']}
    full={k:float(np.sum(rendered[k])) for names in groups.values() for k in names}
    matched=extractions['registered']['component_band_integrals'];fractions={}
    for name,bands in [('Previous full field',full),('Registered slit',matched)]:
        totals=np.array([sum(bands[k] for k in names) for names in groups.values()]);fractions[name]=(totals/totals.sum()).tolist()
    fig,ax=plt.subplots(figsize=(12,6));fig.subplots_adjust(left=.19,right=.97,top=.82,bottom=.22)
    yy=np.arange(len(groups));ax.barh(yy+.18,np.array(fractions['Previous full field'])*100,height=.34,color='#999999',label='Previous full field')
    ax.barh(yy-.18,np.array(fractions['Registered slit'])*100,height=.34,color='#0084a8',label='Registered slit + seeing')
    ax.set(yticks=yy,yticklabels=list(groups),xlabel='Fraction of modeled F187N source signal (%)',xlim=(0,100));ax.invert_yaxis();ax.legend();ax.grid(axis='x',alpha=.15)
    fig.suptitle('How the observed aperture changes the model components',fontsize=19)
    fig.text(.19,.12,'Photon-weighted filter integrals of the unchanged source components; no empirical background or detector noise included.',fontsize=10)
    fig.text(.19,.067,'These are model contributions, not an observed decomposition. They describe the selected region, not the whole galaxy.',fontsize=10)
    save(fig,out,'03_component_selection')
    np.savez_compressed(out/'comparison_data.npz',observed_wave_angstrom=w,corrected_flux=obs['corrected_flux'],corrected_error_envelope=obs['corrected_error_envelope'],
        usable=good,synthetic_wave_angstrom=sw,scenario_min=low,scenario_max=high,**curves)
    with (out/'model_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['scenario']+list(metrics['registered']));writer.writeheader()
        for k,v in metrics.items():writer.writerow({'scenario':k,**v})
    report=dict(status='Archive extraction matched; approximate registration and seeing, not a completed absolute aperture/flux calibration',
        extraction_folder=str(folder.relative_to(ROOT)),observed_spectrum_unchanged=True,atmospheric_correction_refitted=False,
        physical_galaxy_parameters_changed=False,geometric_model_stays_2d=True,
        anchor_um=[1.985,1.995],anchor_samples=int(anchor.sum()),diagnostic_pa_window_um=[1.881,1.8845],diagnostic_retained_samples=int(line.sum()),
        metric_warning='Median per-sample ratio in retained Pa-alpha-region pixels after continuum scaling; not an integrated line-flux ratio, goodness-of-fit significance, or a recovery of missing wavelengths',
        metrics=metrics,component_fractions=fractions,registered_field_coverage=extractions['registered'],
        aperture_audit=audit,hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [obs_path,fullpath,folder/'registered.npz',Path(__file__)]})
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(metrics=metrics,component_fractions=fractions),indent=2))

if __name__=='__main__':main()
