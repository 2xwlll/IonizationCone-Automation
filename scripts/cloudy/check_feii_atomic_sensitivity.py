"""Document a controlled atomic-dataset comparison, without altering defaults."""
import csv,json,os,sys
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ.setdefault('MPLCONFIGDIR','/tmp/ioncone-matplotlib')
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.cloudy.fit_ngc1068_spectral_constraints import fit_lines


def rates(dataset):
 root=Path('data/tools/c25.00/data/stout/fe/fe_2')
 energies={}
 for row in (root/f'fe_2_{dataset}.nrg').read_text().splitlines()[1:]:
  parts=row.split()
  if len(parts)>2 and parts[0].isdigit():energies[int(parts[0])]=float(parts[1])
 transitions={}
 for row in (root/f'fe_2_{dataset}.tp').read_text().splitlines():
  parts=row.split()
  if len(parts)==4 and parts[0]=='A':transitions[int(parts[1]),int(parts[2])]=float(parts[3])
 ratio=transitions[1,10]/transitions[6,10]*(energies[10]-energies[1])/(energies[10]-energies[6])
 return {'A_1_10_s_inverse':transitions[1,10],'A_6_10_s_inverse':transitions[6,10],
  'optically_thin_intrinsic_energy_ratio_1257_to_1644':ratio,
  'level_definition':'indices 1,6,10 checked in the bundled energy files: ground a6D9/2, a4F9/2, upper a4D7/2'}


def main():
 out=Path('results/cloudy/ngc1068_spectral_constraints_v1')
 with Path('data/spectra/ngc1068_martins2010_v1/nuclear_lines.csv').open() as f:
  measured={r['line_id']:{'flux_1e13':float(r['flux_1e13']),'flux_error_1e13':float(r['flux_error_1e13'])} for r in csv.DictReader(f)}
 with np.load('data/cloudy/ngc1068_agn_spectral_pilot_v1/grid.npz') as f:default={k:f[k] for k in f.files}
 with np.load('data/cloudy/ngc1068_agn_tayal18_control_v1/grid.npz') as f:control={k:f[k] for k in f.files}
 # Compare identical macroscopic conditions: log U=-1.5, log n_H=3, same SED/grains.
 default_index=int(np.flatnonzero(default['log_u']==-1.5)[0])
 one={k:(v[default_index:default_index+1] if k in ('model_id','log_u','log_nh_cm3','metallicity_solar','dust_scale','pah_scale','line_flux','continuum_rest_flambda') else v) for k,v in default.items()}
 best_default=fit_lines(one,measured)[0];best_control=fit_lines(control,measured)[0]
 ratio=lambda grid,index:float(grid['line_flux'][index,list(grid['line_names']).index('fe2_12566')]/grid['line_flux'][index,list(grid['line_names']).index('fe2_16435')])
 report={'status':'atomic-data sensitivity test, not a validated selection or a correction to observed fluxes',
  'bundled_atomic_rate_audit':{name:rates(name) for name in ('Smyth19','Tayal18')},
  'default_emergent_feii_ratio':ratio(default,default_index),'control_emergent_feii_ratio':ratio(control,0),
  'observed_nuclear_feii_ratio':measured['fe2_12566']['flux_1e13']/measured['fe2_16435']['flux_1e13'],
  'default_best_scenario':best_default,'control_best_scenario':best_control,
  'limitations':['Same-upper-level optically-thin branching ratios differ substantially between the bundled data sets',
    'Emergent line ratios also depend on radiative transfer/attenuation; measured ratio is not an intrinsic branching-ratio calibration',
    'Do not attribute iron mismatch uniquely to shocks, geometry or abundances before accounting for atomic-data sensitivity',
    'Changing atomic data may change thermal balance; the control recalculates the entire cloud',
    'Neither dataset is established as uniquely correct by this single test; both outputs are preserved',
    'Physical parameters and abundances remain restricted; no production image replacement is performed']}
 (out/'feii_atomic_sensitivity.json').write_text(json.dumps(report,indent=2)+'\n')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
 names=best_default['line_ids'];x=np.arange(len(names))
 ax.errorbar(x,[measured[n]['flux_1e13'] for n in names],yerr=[measured[n]['flux_error_1e13'] for n in names],fmt='o',label='Published nuclear flux')
 ax.plot(x,best_default['predicted_flux_1e13'],'s',label='Smyth19 default data')
 ax.plot(x,best_control['predicted_flux_1e13'],'^',label='Tayal18 control data')
 ax.set_xticks(x,names,rotation=35,ha='right');ax.set_yscale('log');ax.set_ylabel('Flux (10^-13 erg s^-1 cm^-2)')
 ax.legend(fontsize=9);ax.set_title('Same cloud conditions, different Fe II atomic dataset\nSeparate fitted scale/foreground attenuation; this is a sensitivity check')
 fig.savefig(out/'atomic_dataset_control.png',dpi=150);plt.close(fig)
 print(json.dumps(report,indent=2))


if __name__=='__main__':main()
