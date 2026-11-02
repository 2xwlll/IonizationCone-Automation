"""Plot masked atmospheric corrections and the unchanged current galaxy spectrum."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR','/tmp/agn-telluric-plot-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.cloudy.plot_f187n_spectral_reference import line_catalog


def broaden(wave,flux,resolution):
    logwave=np.linspace(np.log(wave[0]),np.log(wave[-1]),len(wave))
    value=np.interp(np.exp(logwave),wave,flux)
    sigma=1/(2.354820045*resolution*np.diff(logwave).mean())
    return np.interp(wave,np.exp(logwave),gaussian_filter1d(value,sigma,mode='nearest'))


def process(folder,out,synthetic_path):
    out.mkdir(parents=True)
    summary=json.loads((folder/'summary.json').read_text())
    data=np.load(folder/'corrected_spectrum.npz');source=np.load(synthetic_path)
    wave=data['heliocentric_vacuum_angstrom'];x=wave/1e4;good=data['usable']
    flux=data['corrected_flux'];error=data['corrected_error_envelope'];unit=1e-15
    sw=source['observed_vacuum_angstrom'];fw=source['filter_wavelength_angstrom'];throughput=source['throughput']
    syn=broaden(sw,source['synthetic_relative_flambda'],summary['fitted_resolving_power'])
    continuum=broaden(sw,source['synthetic_continuum'],summary['fitted_resolving_power'])
    anchor=good&(wave>=19850)&(wave<=19950)
    if anchor.sum()<10:raise ValueError('Insufficient corrected continuum anchor samples')
    scale=float(np.median(flux[anchor])/np.median(np.interp(wave[anchor],sw,continuum)))
    if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid model scale')
    syn=syn*scale/unit
    y=np.where(good,flux/unit,np.nan);err=np.where(good,error/unit,np.nan)
    zoom=(1.858,1.905);inside=(x>zoom[0])&(x<zoom[1]);sm=(sw/1e4>zoom[0])&(sw/1e4<zoom[1])
    ymax=max(float(np.nanmax((y+err)[inside])),float(syn[sm].max()))*1.18
    ymin=min(0.,float(np.nanmin((y-err)[inside]))*1.1)
    lines=line_catalog(.003793,fw,throughput)
    for row in lines:
        row['measurement_status']='Expected transition guide, not a detection or fitted centroid in this extraction'
        if row['line'] in ('He I','He II','Fe II blend'):row['source_url']='https://gitlab.nublado.org/cloudy/cloudy'
    date=summary['observation_date'][:10]
    title=f"ESO/VLT X-shooter • {date} • {summary['target']}"
    colors={'hard':'#70429a','star':'#197a56','hydrogen':'#ac5408','iron':'#985064','coronal':'#b52727'}
    geometry=[]
    plt.rcParams.update({'font.size':12,'axes.labelsize':13,'savefig.facecolor':'white'})
    for overlay in [False,True]:
        fig=plt.figure(figsize=(14,9.5))
        gs=fig.add_gridspec(3,1,height_ratios=[.85,3.5,1.15],hspace=.12)
        strip,ax,atm=[fig.add_subplot(gs[i]) for i in range(3)]
        fig.subplots_adjust(left=.095,right=.97,top=.82,bottom=.22)
        fig.suptitle('NGC 1068 — atmosphere-corrected spectral samples'+(' + synthetic model' if overlay else ''),fontsize=19,y=.975)
        fig.text(.5,.93,title,ha='center',fontsize=12)
        fig.text(.5,.894,'PRELIMINARY • Ground-based spectrum in the F187N wavelength range • Rejected wavelengths remain gaps',ha='center',fontsize=11,color='#9a4526')
        strip.set(xlim=zoom,ylim=(0,3.9));strip.axis('off')
        positions=[(1.866,2.),(1.872,3.35),(1.877,1.65),(1.881,3.35),(1.886,2.8),(1.892,3.35),(1.900,2.)]
        for row,(tx,ty) in zip(lines[:7],positions):
            line_x=row['observed_vacuum_um'];label=row['line'].replace('Pa alpha','Pa α')
            if row['line']!='H I Pa alpha':label+='*'
            strip.annotate(label+'\n'+f'{line_x:.5f} μm',xy=(line_x,.05),xytext=(tx,ty),ha='center',va='center',fontsize=10.5,color=colors[row['classification']],arrowprops=dict(arrowstyle='-',color=colors[row['classification']],lw=.8))
            ax.axvline(line_x,color=colors[row['classification']],alpha=.2,lw=.8)
        ax.fill_between(x,y-err,y+err,color='#555555',alpha=.22,linewidth=0,label='Estimated error envelope (not a formal confidence interval)')
        ax.plot(x,y,'.-',ms=2.6,lw=.9,color='#202020',label='Corrected measured samples passing quality cuts')
        if overlay:ax.plot(sw/1e4,syn,color='#007faa',lw=1.7,label='Current synthetic galaxy: continuum-scaled, aperture unmatched')
        ax.set(xlim=zoom,ylim=(ymin,ymax),ylabel=r'$F_\lambda$ ($10^{-15}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)')
        ax.tick_params(labelbottom=False);ax.grid(alpha=.15)
        ax.legend(loc='upper left',fontsize=9.2,framealpha=.97)
        count=int(np.sum(good&inside));total=int(inside.sum())
        ax.text(.99,.96,f'{count}/{total} samples retained in this plotted interval',transform=ax.transAxes,ha='right',va='top',fontsize=10)
        atm.plot(x,data['transmission'],color='#a15d29',lw=1.,label='Fitted effective Earth-atmosphere transmission')
        atm.plot(fw/1e4,throughput/throughput.max(),color='#007faa',ls='--',lw=1.4,label='F187N throughput / peak')
        atm.axhline(.25,color='#a15d29',ls=':',lw=1)
        atm.fill_between(x,0,1,where=~good,step='mid',color='#cccccc',alpha=.35,linewidth=0)
        atm.set(xlim=zoom,ylim=(-.04,1.04),ylabel='Transmission',xlabel='Heliocentric vacuum wavelength (μm)')
        atm.legend(loc='upper left',fontsize=8.5,ncol=2,framealpha=.95)
        fig.text(.095,.13,'Grey shading below marks rejected samples. Expected line guides at z = 0.003793; * denotes a model transition, not an observed detection.',fontsize=10)
        fig.text(.095,.098,'Correction fitted outside 1.858–1.905 μm, independently of CLOUDY. Error envelope includes statistical, atmosphere-fit and line-division sensitivity.',fontsize=10)
        fig.text(.095,.066,'Overlay uses the full 7.9″ synthetic field versus a 0.9″ slit; one continuum scale from 1.985–1.995 μm. No in-band line fitting or gap filling.',fontsize=10)
        fig.text(.095,.034,'This is a partial spectral comparison, not a recovered full Pa-alpha profile or a JWST spectrum. Absolute calibration and slit-loss errors remain unquantified.',fontsize=10)
        name='02_corrected_with_synthetic' if overlay else '01_corrected_labeled'
        for ext in ['png','pdf','svg']:fig.savefig(out/f'{name}.{ext}',dpi=180)
        geometry.append([list(ax.get_xlim()),list(ax.get_ylim())]);plt.close(fig)
    assert geometry[0]==geometry[1]
    # Separately show what was fitted, and the held-out data; no galaxy template.
    fig,(ax,res)=plt.subplots(2,1,figsize=(14,7),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    fig.subplots_adjust(left=.09,right=.98,top=.87,bottom=.18,hspace=.12)
    raw=np.where(data['observed_valid'],data['raw_flux']/unit,np.nan)
    pred=data['forward_flux']/unit
    ax.plot(x,raw,color='#888888',lw=.65,label='Raw measured spectrum (valid pipeline samples)')
    ax.plot(x,pred,color='#a15d29',lw=1.1,label='Fitted smooth continuum × atmosphere, blurred by instrument')
    ax.plot(x,y,color='#151515',lw=.8,label='Corrected retained samples')
    visible=(x>=1.75)&(x<=2.002)
    ax.set(ylim=(0,np.nanmax(np.r_[raw[visible],y[visible]])*1.12),ylabel=r'$F_\lambda$ ($10^{-15}$ cgs / Å)')
    ax.legend(fontsize=9,loc='upper left');ax.grid(alpha=.15)
    fractional=data['raw_flux']/data['forward_flux']-1
    for key,color,label in [('fit_training','#7c7c7c','Fitted samples'),('fit_validation','#208bad','Withheld continuum samples')]:
        mask=data[key]&(data['transmission']>=.25)
        res.plot(x[mask],fractional[mask],'.',ms=2.5,color=color,label=label)
    for lo,hi in summary['excluded_galaxy_lines_nm']:res.axvspan(lo/1000,hi/1000,color='#bbaacb',alpha=.16)
    res.axhline(0,color='black',lw=.8)
    res.set(xlim=(1.75,2.002),ylim=(-.3,.3),ylabel='(raw − fit) / fit',xlabel='Heliocentric vacuum wavelength (μm)')
    res.legend(fontsize=9,loc='upper left',ncol=2);res.grid(alpha=.15)
    fig.suptitle('Atmospheric correction audit — '+date,fontsize=18)
    fig.text(.09,.09,f"Held-out robust scatter: {100*summary['heldout_fractional_robust_scatter']:.2f}% • Fitted PWV: {summary['fitted_pwv_mm']:.3f} mm • Effective resolving power: {summary['fitted_resolving_power']:.0f}",fontsize=11)
    fig.text(.09,.045,'Purple intervals were excluded from atmospheric fitting. A smooth continuum is a calibration assumption, not a reconstruction of galaxy emission lines.',fontsize=10)
    for ext in ['png','pdf']:fig.savefig(out/f'03_correction_diagnostics.{ext}',dpi=180)
    plt.close(fig)
    names=['heliocentric_vacuum_angstrom','raw_flux','raw_error','corrected_flux','corrected_stat_error','corrected_error_envelope','transmission','transmission_sensitivity','line_injection_bias_envelope','relative_atmosphere_error','observed_quality','usable','rejection_bits','fit_training','fit_validation']
    with (out/'spectral_samples.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(names);writer.writerows(zip(*(data[k] for k in names)))
    with (out/'line_guides.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(lines[0]));writer.writeheader();writer.writerows(lines)
    np.savez_compressed(out/'comparison_data.npz',**{k:data[k] for k in data.files},synthetic_wave_angstrom=sw,synthetic_scaled_flux=syn*unit,synthetic_scaled_continuum=continuum*scale,filter_wave_angstrom=fw,filter_throughput=throughput)
    band_t=np.interp(wave,fw,throughput,left=0,right=0)
    halfmax=band_t>=.5*throughput.max()
    report=dict(correction_folder=str(folder.relative_to(ROOT)),observation=summary['observation'],date=date,target=summary['target'],
        model_scale=scale,continuum_anchor_um=[1.985,1.995],anchor_sample_count=int(anchor.sum()),
        main_band_samples=int(halfmax.sum()),main_band_retained_samples=int((halfmax&good).sum()),
        approximate_photon_throughput_weighted_coverage=float(np.sum(wave*band_t*good)/np.sum(wave*band_t)),
        coverage_note='Observed wavelength-bin coverage only, not recovered line flux or fractional signal completeness',
        identical_pair_axes=geometry[0],aperture_matched=False,galaxy_parameters_refitted=False,
        correction_summary=summary,
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [folder/'corrected_spectrum.npz',synthetic_path,Path(__file__)]})
    (out/'comparison_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results/cloudy/ngc1068_corrected_f187n_comparison_v1')
    args=p.parse_args();out=args.output.resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    syn=ROOT/'results/cloudy/f187n_spectral_reference_v2/spectra.npz'
    reports={}
    for key,folder in [('2022_nuclear_slit','ngc1068_telluric_wet_v2'),('2023_slit_position_2','ngc1068_telluric_dry_v1')]:
        reports[key]=process(ROOT/'results/cloudy'/folder,out/key,syn)
    (out/'summary.json').write_text(json.dumps(reports,indent=2)+'\n')
    print(json.dumps({k:{a:r[a] for a in ['main_band_samples','main_band_retained_samples','approximate_photon_throughput_weighted_coverage']} for k,r in reports.items()},indent=2))


if __name__=='__main__':main()
