"""Fit published nuclear line constraints and local-continuum proxies.

This is NOT a fit to downloaded pixel spectra. Pa-alpha was unusable in the
source study. No derived prediction is substituted for a measurement.
"""
import argparse,csv,hashlib,json,os,sys
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/ioncone-matplotlib')
import numpy as np
from scipy.optimize import nnls
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.simple_utils.nuclear_sed import nuclear_flambda_shape
from src.simple_utils.wavelengths import cloudy_air_to_vacuum
from src.simple_utils.f187n import line_response
from src.simple_utils.synthetic_photometry import photon_bandpass_weights
from src.jwst_utils.nircam import load_filters

REF=18756.119850349412
FIT_IDS=('pa_beta','br_gamma','pa_gamma','he2_10123','fe2_12566','fe2_16435','si6_19624')


def fit_lines(grid,measurements):
 names=list(grid['line_names']);ids=[x for x in FIT_IDS if x in measurements]
 indices=[names.index(x) for x in ids]
 observed=np.array([measurements[x]['flux_1e13'] for x in ids])
 errors=np.array([measurements[x]['flux_error_1e13'] for x in ids])
 vacuum=cloudy_air_to_vacuum(grid['rest_line_angstrom'][indices])
 results=[]
 for i in range(len(grid['model_id'])):
  intrinsic=grid['line_flux'][i,indices]
  for tau in np.linspace(0,4,161):
   shape=intrinsic*np.exp(-tau*(vacuum/REF)**(-1.3))
   scale=max(float(np.sum(shape*observed/errors**2)/np.sum((shape/errors)**2)),0.)
   prediction=scale*shape
   score=float(np.sum(((prediction-observed)/errors)**2))
   results.append({'model_index':i,'model_id':str(grid['model_id'][i]),'log_u':float(grid['log_u'][i]),
      'log_nh_cm3':float(grid['log_nh_cm3'][i]),'tau_at_pa_alpha':float(tau),
      'scale_in_table_flux_units_per_cloudy_line_unit':scale,'weighted_residual_sum':score,
      'line_ids':ids,'predicted_flux_1e13':prediction.tolist()})
 results.sort(key=lambda row:row['weighted_residual_sum'])
 return results


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--grid',type=Path,default=Path('data/cloudy/ngc1068_agn_spectral_pilot_v1/grid.npz'))
 p.add_argument('--output',type=Path,default=Path('results/cloudy/ngc1068_spectral_constraints_v1'))
 a=p.parse_args()
 if a.output.exists():raise FileExistsError('Choose a new output directory')
 source=Path('data/spectra/ngc1068_martins2010_v1/nuclear_lines.csv')
 with source.open() as f:
  rows=[{k:(v if k in ('line_id','label') else float(v)) for k,v in r.items()} for r in csv.DictReader(f)]
 measured={r['line_id']:r for r in rows}
 # F_line/EW gives local fitted continuum, not measured spectrum pixels. The
 # source flux and EW errors are correlated; these illustrative propagated
 # uncertainties and residual scores are not likelihoods or confidence limits.
 points=[r for r in rows if r['line_id']!='h2_1956']
 wave=np.array([r['published_wavelength_um']*1e4 for r in points])
 continuum=np.array([r['flux_1e13']/r['ew_um']*1e-17 for r in points])
 proxy_sigma=continuum*np.sqrt(np.array([(r['flux_error_1e13']/r['flux_1e13'])**2+(r['ew_error_um']/r['ew_um'])**2 for r in points]))
 scale_c=1e-14;target=continuum/scale_c;sig=np.maximum(proxy_sigma,.3*continuum)/scale_c
 candidates=[]
 for temp in range(700,1601,100):
  for tau in np.linspace(0,3,31):
   base={'reference_rest_angstrom':REF,'hot_dust_temperature_k':float(temp),
    'hot_dust_fraction_at_reference':1.,'powerlaw_alpha_nu':-.5,'tau_at_reference':0.,'extinction_exponent':1.3}
   dust=nuclear_flambda_shape(wave,base)
   star=nuclear_flambda_shape(wave,{**base,'hot_dust_temperature_k':4500.})
   power=(wave/REF)**(-1.5)
   screen=np.exp(-tau*(wave/REF)**(-1.3))
   design=np.column_stack((dust,power,star))*screen[:,None]
   coeff,_=nnls(design/sig[:,None],target/sig)
   score=float(np.sum(((design@coeff-target)/sig)**2))
   candidates.append({'dust_temperature_k':temp,'tau_at_reference':float(tau),'coefficients':coeff.tolist(),'descriptive_score':score})
 candidates.sort(key=lambda r:r['descriptive_score']);cbest=candidates[0]
 def continuum_model(rest_wave):
  base={'reference_rest_angstrom':REF,'hot_dust_temperature_k':float(cbest['dust_temperature_k']),
   'hot_dust_fraction_at_reference':1.,'powerlaw_alpha_nu':-.5,'tau_at_reference':0.,'extinction_exponent':1.3}
  components=np.array([nuclear_flambda_shape(rest_wave,base),(rest_wave/REF)**(-1.5),
                       nuclear_flambda_shape(rest_wave,{**base,'hot_dust_temperature_k':4500.})])
  return np.array(cbest['coefficients'])[:,None]*components*scale_c*np.exp(-cbest['tau_at_reference']*(rest_wave/REF)**(-1.3))
 with np.load(a.grid,allow_pickle=False) as f:grid={k:f[k] for k in f.files}
 results=fit_lines(grid,measured);best=results[0]
 regional_source=source.parent/'regional_lines.csv'
 with regional_source.open() as f:regional_rows=list(csv.DictReader(f))
 regional_results={}
 for region in ('south_Ap01','north_Ap09'):
  measurements={r['line_id']:{'flux_1e13':float(r['flux_1e14'])/10.,
    'flux_error_1e13':float(r['flux_error_1e14'])/10.} for r in regional_rows
    if r['region']==region and r['kinematic_component']=='main'}
  ranked=fit_lines(grid,measurements)
  regional_results[region]={'fit_scope':'reported main kinematic component only, not total aperture flux',
    'best_scenario':ranked[0],'measurements_in_1e13_units':measurements,
    'best_per_agn_model':[next(r for r in ranked if r['model_index']==j) for j in range(len(grid['model_id']))]}
 i=best['model_index'];vac=cloudy_air_to_vacuum(grid['rest_line_angstrom'])
 fluxes=grid['line_flux'][i]*best['scale_in_table_flux_units_per_cloudy_line_unit']*1e-13*np.exp(-best['tau_at_pa_alpha']*(vac/REF)**(-1.3))
 z=.003793;fw,t=load_filters()['F187N']['curve'].T
 cont_band=continuum_model(fw/(1+z)).sum(axis=0)/(1+z)
 predictions=[]
 for v in (-600.,0.,600.):
  responses=np.array([line_response(rest,z,v,85,fw,t) for rest in vac])
  line_photons=fluxes*responses
  cont_photons=float(photon_bandpass_weights(fw,fw,t)@cont_band)
  predictions.append({'assumed_bulk_velocity_kms':v,'assumed_sigma_kms':85.,
   'selected_line_photons_cm2_s':float(line_photons.sum()),
   'continuum_proxy_model_photons_cm2_s':cont_photons,
   'selected_lines_plus_proxy_continuum_photons_cm2_s':float(line_photons.sum()+cont_photons)})
 a.output.mkdir(parents=True)
 with (a.output/'continuum_proxies.csv').open('w',newline='') as f:
  writer=csv.writer(f);writer.writerow(['line_id','nominal_rest_angstrom','derived_local_flambda_erg_s_cm2_angstrom','sigma_if_independent_flux_ew_errors'])
  writer.writerows((r['line_id'],w,c,s) for r,w,c,s in zip(points,wave,continuum,proxy_sigma))
 report={'status':'first-pass line-constrained model and continuum-proxy fit; not a fit to raw observed spectra',
  'source_measurements':str(source),'source_manifest':'data/spectra/ngc1068_martins2010_v1/manifest.json',
  'agn_grid':str(a.grid),'best_line_scenario':best,'regional_main_component_fits':regional_results,'best_continuum_scenario':cbest,
  'continuum_coefficient_order':['hot_dust_Planck','powerlaw_alpha_nu_minus_0.5','4500K_stellar_Planck_proxy'],
  'best_per_agn_model':[next(r for r in results if r['model_index']==j) for j in range(len(grid['model_id']))],
  'continuum_near_best_scenarios':candidates[:20],
  'predicted_pa_alpha_integrated_flux_erg_s_cm2':float(fluxes[list(grid['line_names']).index('pa_alpha')]),
  'predicted_f187n_scenarios':predictions,
  'largest_abs_line_residual_over_reported_error':max(abs((pred-measured[name]['flux_1e13'])/measured[name]['flux_error_1e13']) for name,pred in zip(best['line_ids'],best['predicted_flux_1e13'])),
  'continuum_fit_weighting':'illustrative independent flux/EW error propagation, with 30 percent continuum error floor; unknown covariance prevents likelihood interpretation',
  'limitations':['Pa alpha was not reliably measured in the source; its flux here is a model prediction',
    'No downloaded sampled nuclear spectrum or registered regional spectral extraction was obtained',
    'Continuum points are derived F_line/EW with unknown covariance and approximate EW frame assumption',
    'Continuum modeling uses an illustrative stellar blackbody, not an SSP spectrum or unique decomposition',
    'Three single-zone AGN models at fixed density/abundance do not span a physical posterior',
    'H II regions, shocks, multiple gas phases and stellar absorption are not fitted here',
    'He I 1.083 blend excluded: current saved model component is not established as the observed full blend',
    'H2 lines excluded from fit; S(3) near Si VI is blended in source',
    'Line residual sum ignores calibration covariance; it is not a validated chi-square likelihood',
    'F187 predictions assume the published EW frame is rest and sigma85 km/s; velocities are illustrative',
    'Regional main-component fits do not classify those apertures as pure AGN, host or star formation; seeing mixes neighboring regions',
    'No observation-matched aperture registration/PSF correction or direct F187 flux validation performed',
    'Published 2007 measurements and 2024 imaging differ in epoch; images are not automatically recalibrated'],
  'hashes':{str(path):hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in [source,a.grid]}}
 (a.output/'fit.json').write_text(json.dumps(report,indent=2)+'\n')
 rest_plot=np.linspace(9000,24000,5000);comp=continuum_model(rest_plot)
 np.savez_compressed(a.output/'modeled_continuum.npz',rest_angstrom=rest_plot,modeled_flambda=comp.sum(axis=0),components=comp,
    observed_proxy_nominal_rest_angstrom=wave,derived_continuum_proxy=continuum)
 with (a.output/'predicted_line_band_contributions.csv').open('w',newline='') as f:
  writer=csv.writer(f);writer.writerow(['line_id','rest_vacuum_um','observed_center_um','predicted_integrated_flux_erg_s_cm2','predicted_f187n_photons_cm2_s_at_v0'])
  for name,rest,flux in zip(grid['line_names'],vac,fluxes):
   writer.writerow([name,rest/1e4,rest*(1+z)/1e4,flux,flux*line_response(rest,z,0,85,fw,t)])
 with (a.output/'line_comparison.csv').open('w',newline='') as f:
  writer=csv.writer(f);writer.writerow(['line_id','measured_flux_erg_s_cm2','reported_error_erg_s_cm2','model_flux_erg_s_cm2','residual_over_reported_error'])
  for name,pred in zip(best['line_ids'],best['predicted_flux_1e13']):
   r=measured[name];writer.writerow([name,r['flux_1e13']*1e-13,r['flux_error_1e13']*1e-13,pred*1e-13,(pred-r['flux_1e13'])/r['flux_error_1e13']])
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
 axes[0].errorbar(wave/1e4,continuum/scale_c,yerr=proxy_sigma/scale_c,fmt='o',color='black',markersize=4,label='Local continuum inferred from published flux/EW')
 axes[0].plot(rest_plot/1e4,comp.sum(axis=0)/scale_c,label='Phenomenological continuum fit')
 for layer,label in zip(comp,('Hot dust','Power law','Stellar blackbody proxy')):axes[0].plot(rest_plot/1e4,layer/scale_c,'--',alpha=.6,label=label)
 axes[0].axvspan(1.863/(1+z),1.885/(1+z),color='orange',alpha=.2,label='F187N half-maximum interval')
 axes[0].set(xlabel='Nominal rest wavelength (micrometers)',ylabel='Continuum proxy (10^-14 erg s^-1 cm^-2 A^-1)',yscale='log',ylim=(.02,10.))
 axes[0].legend(fontsize=7)
 names=best['line_ids'];indices=np.arange(len(names));obs=np.array([measured[n]['flux_1e13'] for n in names]);err=np.array([measured[n]['flux_error_1e13'] for n in names])
 axes[1].errorbar(indices,obs,yerr=err,fmt='o',label='Published nuclear line flux')
 axes[1].plot(indices,best['predicted_flux_1e13'],'s',label='Best of three CLOUDY AGN scenarios')
 axes[1].set_xticks(indices,names,rotation=40,ha='right');axes[1].set(ylabel='Integrated flux (10^-13 erg s^-1 cm^-2)',yscale='log');axes[1].legend(fontsize=8)
 fig.suptitle('NGC 1068: published spectral constraints versus first-pass models\nPa-alpha unmeasured in source; continuum points are proxies, not a sampled observed spectrum',fontsize=12)
 fig.savefig(a.output/'spectral_constraints.png',dpi=150);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
 for ax,(title,data,scenario) in zip(axes,[('Nucleus',measured,best)]+[(region,result['measurements_in_1e13_units'],result['best_scenario']) for region,result in regional_results.items()]):
  ids=scenario['line_ids'];x=np.arange(len(ids))
  ax.errorbar(x,[data[n]['flux_1e13'] for n in ids],yerr=[data[n]['flux_error_1e13'] for n in ids],fmt='o',label='Published line flux')
  ax.plot(x,scenario['predicted_flux_1e13'],'s',label='Single-zone AGN scenario')
  ax.set_xticks(x,ids,rotation=50,ha='right');ax.set_yscale('log');ax.set_title(title)
  ax.set_ylabel('Flux (10^-13 erg s^-1 cm^-2)');ax.legend(fontsize=7)
 fig.suptitle('Regional spectral constraints: adjacent apertures are seeing-correlated\nNorth/south panels use main velocity components only; neither is an assumed pure excitation class',fontsize=11)
 fig.savefig(a.output/'regional_line_constraints.png',dpi=150);plt.close(fig)
 print(json.dumps({'best_line_model':best,'best_continuum':cbest,'F187N_predictions':predictions},indent=2))


if __name__=='__main__':main()
