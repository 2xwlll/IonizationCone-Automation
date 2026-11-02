"""Fit coarse reference summaries and render a small exploratory F187N population."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/ioncone-matplotlib')
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.simple_utils.f187n_calibration import GROUPS, annular_means, component_group, fit_annular_templates


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/cloudy/f187n_population_pilot_v3'))
    parser.add_argument('--render',action='store_true')
    args=parser.parse_args()
    out=args.output
    if out.exists():
        raise FileExistsError('Choose a new population output directory')
    source=Path('results/cloudy/ngc1068_f187n_pilot_v2/image_mask.npz')
    with np.load(source) as f:
        meta=json.loads(f['metadata'].item())
        observed,valid=f['reference_science'],f['reference_valid'].astype(bool)
        original=f['clean'].copy()
        groups={group:sum(f[name] for name in meta['component_names']
                          if component_group(name)==group)/meta['empirical_scale'] for group in GROUPS}
        old_mask=f['mask'].copy()
    n=observed.shape[0]
    y,x=np.indices(observed.shape)
    radius=np.hypot(x-(n-1)/2,y-(n-1)/2)*meta['pixel_scale_arcsec']
    # Include valid core pixels to constrain the nucleus; invalid pixels stay excluded.
    edges=np.r_[0.,.15,np.linspace(.3,3.7,17)]
    scales={g:max(float(np.max(groups[g]) if g=='nuclear' else np.percentile(groups[g],99)),1e-30) for g in GROUPS}
    templates=np.array([groups[g]/scales[g] for g in GROUPS])
    unconstrained,profiles,target,usable=fit_annular_templates(templates,observed,valid,radius,edges)
    # Deliberate cone-containing scenario, not a measured cone fraction. Preserve
    # the unconstrained solution to expose the single-band degeneracy.
    fixed_cone = .2*scales['cone']*meta['empirical_scale']
    # Conservative descriptive cap: the normalized point-source peak alone
    # cannot exceed the largest valid observed SCI value. This is not a flux
    # measurement of the nucleus and does not fill the missing central pixels.
    upper=np.array([np.inf,np.inf,np.inf,float(observed[valid].max()),np.inf])
    coefficients,_,_,_=fit_annular_templates(templates,observed,valid,radius,edges,
                                           fixed_first=fixed_cone,upper_bounds=upper)
    best_profile = profiles.T@unconstrained[:-1]+unconstrained[-1]
    calibrated=sum(layer*c for layer,c in zip(templates,coefficients[:-1]))+coefficients[-1]
    before=annular_means(original,valid,radius,edges)
    after=annular_means(calibrated,valid,radius,edges)
    score=lambda profile:float(np.sqrt(np.mean(((profile[usable]-target[usable])/np.maximum(abs(target[usable]),1))**2)))
    calibration={'status':'cone-containing exploratory scenario with nuisance amplitudes fitted to annular means',
                 'unconstrained_amplitudes_mjysr':dict(zip(GROUPS,map(float,unconstrained[:-1]))),
                 'unconstrained_baseline_mjysr':float(unconstrained[-1]),
                 'fixed_cone_assumption':'20 percent of the previous prototype cone amplitude; NOT an observational inference',
                 'fixed_cone_amplitude_mjysr':float(fixed_cone),
                 'nuclear_peak_upper_bound_mjysr':float(upper[3]),
                 'source_template':str(source),'template_scales_relative':scales,
                 'amplitudes_mjysr':dict(zip(GROUPS,map(float,coefficients[:-1]))),
                 'baseline_mjysr':float(coefficients[-1]),'annular_edges_arcsec':edges.tolist(),
                 'fractional_annular_rms_unconstrained':score(best_profile),'fractional_annular_rms_before':score(before),'fractional_annular_rms_after':score(after),
                 'method':'NNLS on annular mean SCI with inverse mean brightness weights; fixed nonzero cone and capped nuclear peak; no pixel morphology fitting',
                 'limitations':['Single-band template coefficients are degenerate, not physical gas fractions',
                                'All annuli are fit constraints, not held-out validation',
                                'Reference template geometry is not fitted',
                                'No empty-sky region; baseline includes galaxy light',
                                'Original CLOUDY convergence, continuum and instrument limitations remain']}
    out.mkdir(parents=True)
    (out/'calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
    np.savez_compressed(out/'reference_comparison.npz',real=observed,valid=valid,before=original,
                        after=calibrated,mask=old_mask,annular_edges_arcsec=edges,
                        reference_profile=target,before_profile=before,after_profile=after)
    base=json.loads(Path('configs/cloudy/ngc1068_f187n_pilot_v2.json').read_text())
    base['calibration']=str(out/'calibration.json')
    # These are deliberately exploratory choices, not posterior samples or measured bounds.
    variants=[('reference',1069,30,45,35,.85,100,.55,.4,1.,1.),
              ('variant_01',2101,75,25,22,.65,130,.75,.2,.6,1.3),
              ('variant_02',2102,125,60,42,.90,35,.45,.8,1.5,.8),
              ('variant_03',2103,170,40,28,.75,85,.65,.5,1.,1.5)]
    manifest={'status':'exploratory pilot; not a train/validation/test split or calibrated population prior',
              'calibration':str(out/'calibration.json'),'samples':[]}
    (out/'configs').mkdir()
    for name,seed,pa,inc,opening,extent,host_pa,q,tau,gas_factor,stars_factor in variants:
        cfg=json.loads(json.dumps(base))
        cfg['seed']=seed
        cfg['output']=str(out/name)
        cfg['status']='Exploratory population member anchored to coarse single-band reference summaries'
        cfg['geometry'].update(position_angle_east_of_north_deg=pa,inclination_deg=inc,
                               opening_half_angle_deg=opening,radius_fraction=extent)
        cfg['host_geometry'].update(position_angle_east_of_north_deg=host_pa,axis_ratio=q)
        cfg['components']['foreground_tau_at_pa_alpha']=tau
        cfg['population_amplitude_factors']={'cone':gas_factor,'host':1/gas_factor,'stellar':stars_factor,'nuclear':1.}
        path=out/'configs'/f'{name}.json'
        path.write_text(json.dumps(cfg,indent=2)+'\n')
        manifest['samples'].append({'name':name,'configuration':str(path),'output':cfg['output']})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import AsinhNorm
    fig,axes=plt.subplots(1,4,figsize=(16,4),constrained_layout=True)
    top=float(np.percentile(observed[valid],99.5))
    for ax,im,title in zip(axes[:3],(observed,original,calibrated),('Real F187N','Previous prototype','Cone-containing scenario')):
        ax.imshow(np.ma.array(im,mask=~valid),origin='lower',cmap='magma',norm=AsinhNorm(linear_width=.03*top,vmin=0,vmax=top))
        ax.set_title(title); ax.set_axis_off()
    centers=(edges[1:]+edges[:-1])/2
    for profile,name in ((target,'Real'),(before,'Previous'),(after,'Fit')):
        axes[3].plot(centers,profile,'o-',label=name)
    axes[3].set_yscale('log'); axes[3].set_xlabel('Radius (arcsec)'); axes[3].set_ylabel('Mean MJy/sr'); axes[3].legend()
    fig.suptitle('Broad radial constraints only: cloud positions and cone geometry are not fitted')
    fig.savefig(out/'reference_comparison.png',dpi=140); plt.close(fig)
    print(json.dumps(calibration,indent=2),flush=True)
    if not args.render:
        return
    for sample in manifest['samples']:
        print('Rendering '+sample['name'],flush=True)
        subprocess.run([sys.executable,'scripts/cloudy/render_f187n_pilot.py','--config',sample['configuration']],check=True)
    fig,axes=plt.subplots(2,len(variants),figsize=(14,7),constrained_layout=True)
    for column,sample in enumerate(manifest['samples']):
        with np.load(Path(sample['output'])/'image_mask.npz') as f:
            axes[0,column].imshow(f['input'],origin='lower',cmap='magma',norm=AsinhNorm(linear_width=.03*top,vmin=0,vmax=top))
            axes[1,column].imshow(f['mask'],origin='lower',cmap='gray',vmin=0,vmax=1)
        axes[0,column].set_title(sample['name'])
        axes[1,column].set_title('Known cone geometry')
    for ax in axes.flat: ax.set_axis_off()
    fig.suptitle('Exploratory F187N population: shared reference calibration, varied galaxies')
    fig.savefig(out/'population.png',dpi=140); plt.close(fig)
    print('Finished '+str(out),flush=True)


if __name__=='__main__':
    main()
