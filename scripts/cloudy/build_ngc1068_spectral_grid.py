"""Calculate a small converged AGN grid with explicit near-IR diagnostic lines."""
import argparse,hashlib,itertools,json,os,re,subprocess,sys
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.jwst_utils.nircam import load_filters
from src.simple_utils.f187n import continuum_from_cached_file

# Air wavelengths as explicitly requested from CLOUDY. Photon calculations must
# convert lines to vacuum; continuum save output is already vacuum.
LINES=[('pa_alpha','H 1',18751.,'H  1 1.87510m air'),
 ('he1_18685','He 1',18685.4,'He 1 1.86854m air'),
 ('he1_18697','He 1',18697.2,'He 1 1.86972m air'),
 ('he2_18636','He 2',18636.5,'He 2 1.86365m air'),
 ('he2_18743','He 2',18743.1,'He 2 1.87431m air'),
 ('he2_18757','He 2',18757.,'He 2 1.87570m air'),
 ('fe2b_18890','Fe 2b',18890.,'"Fe 2b" 18890A air'),
 ('he1_19089','He 1',19089.4,'He 1 1.90894m air'),
 ('h1_18174','H 1',18174.1,'H  1 1.81741m air'),
 ('h1_19445','H 1',19445.6,'H  1 1.94456m air'),
 ('he1_19543','He 1',19543.1,'He 1 1.95431m air'),
 ('si6_19624','Si 6',19624.7,'Si 6 1.96247m air'),
 ('pa_beta','H 1',12818.1,'H  1 1.28181m air'),
 ('br_gamma','H 1',21655.3,'H  1 2.16553m air'),
 ('pa_gamma','H 1',10938.1,'H  1 1.09381m air'),
 ('he1_10830','He 1',10830.3,'He 1 1.08303m air'),
 ('he2_10123','He 2',10123.3,'He 2 1.01233m air'),
 ('fe2_12566','Fe 2',12566.8,'Fe 2 1.25668m air'),
 ('fe2_16435','Fe 2',16435.5,'Fe 2 1.64355m air'),
 ('h_beta','H 1',4861.32,'H  1 4861.32A air'),
 ('oiii_5007','O 3',5006.84,'O  3 5006.84A air')]


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--cloudy',type=Path,default=Path('data/tools/c25.00/source/sys_gcc/cloudy.exe'))
 p.add_argument('--log-u',type=float,nargs='+',default=[-3.,-2.25,-1.5])
 p.add_argument('--fe-dataset',choices=['Smyth19','Tayal18','Bautista18','Verner99'])
 p.add_argument('--output',type=Path,default=Path('data/cloudy/ngc1068_agn_spectral_pilot_v1'))
 a=p.parse_args();exe=a.cloudy.resolve()
 if not exe.is_file():raise FileNotFoundError(exe)
 if a.output.exists():raise FileExistsError('Use a fresh output directory')
 runs=a.output/'runs';runs.mkdir(parents=True)
 (runs/'nir_lines.dat').write_text('\n'.join(row[3] for row in LINES)+'\n')
 models=[]
 for u in a.log_u:
  m={'log_u':u,'log_nh_cm3':3.,'metallicity_solar':1.,'dust_scale':1.,'pah_scale':0.,'log_column_density_cm2':21.5}
  if a.fe_dataset:m['fe_dataset']=a.fe_dataset
  m['model_id']='agn_'+hashlib.sha256(json.dumps(m,sort_keys=True).encode()).hexdigest()[:12]
  stem=m['model_id'];models.append(m)
  text='\n'.join([f'title "NGC1068 converged spectral pilot {stem}"','table agn',f'ionization parameter {u}',
   'hden 3','abundances ism no grains','metals 0 log','grains ism 0 log no qheat single',
   'stop column density 21.5','iterate to convergence','print last iteration',
   f'save line list "{stem}.lin" "nir_lines.dat" last no hash emergent absolute',
   f'save continuum "{stem}.con" last units Angstrom',''])
  if a.fe_dataset:text=text.replace('table agn\n','table agn\nspecies \"Fe+\" dataset=\"'+a.fe_dataset+'\"\n')
  (runs/(stem+'.in')).write_text(text)
 manifest={'status':'prepared','models':models,'lines':LINES,'line_wavelength_medium':'air',
  'executable':str(exe),'executable_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),
  'fe_dataset_requested':a.fe_dataset or 'installation default (Smyth19 in this restored release)',
  'scope':'configured AGN models at fixed density/abundance/grain prescription; not a complete grid or fitted galaxy',
  'completed_models':[]}
 def record(): (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 record();flux=[];continua=[];fw,t=load_filters()['F187N']['curve'].T;rest=fw/(1+.003793)
 for m in models:
  stem=m['model_id'];manifest['status']='running';record()
  result=subprocess.run([str(exe),'-r',stem],cwd=runs,text=True,capture_output=True)
  (runs/(stem+'.stdout.txt')).write_text(result.stdout);(runs/(stem+'.stderr.txt')).write_text(result.stderr)
  raw=(runs/(stem+'.out')).read_text()
  try:
   if result.returncode or 'Cloudy exited OK' not in raw:raise ValueError('No successful calculation exit')
   if re.search(r'not converged|did not converge|failed to converge|convergence failure|not reliable on the first iteration',raw,re.I):raise ValueError('Convergence warning')
   rows=[line.split('\t') for line in (runs/(stem+'.lin')).read_text().splitlines() if line.lower().startswith('iteration')]
   iteration=int(rows[-1][0].split()[-1]);values=np.array([float(x) for x in rows[-1][1:]])
   if iteration<2 or len(values)!=len(LINES) or not np.isfinite(values).all() or np.any(values<0):raise ValueError('Invalid saved line list')
   continuum=continuum_from_cached_file(runs/(stem+'.con'),rest)
  except Exception as error:
   manifest.update(status='failed; raw outputs preserved',failure=f'{stem}: {error}');record();raise
  m['iterations']=iteration;m['out_sha256']=hashlib.sha256(raw.encode()).hexdigest()
  flux.append(values);continua.append(continuum);manifest['completed_models'].append(stem);record()
  print('Completed '+stem+f' ({iteration} iterations)',flush=True)
 arrays={key:np.array([m[key] for m in models]) for key in ('model_id','log_u','log_nh_cm3','metallicity_solar','dust_scale','pah_scale')}
 arrays.update(line_flux=np.array(flux),line_names=np.array([r[0] for r in LINES]),line_labels=np.array([r[1] for r in LINES]),
  rest_line_angstrom=np.array([r[2] for r in LINES]),continuum_rest_flambda=np.array(continua),
  continuum_rest_angstrom=rest,filter_wavelength_angstrom=fw,throughput=t)
 np.savez_compressed(a.output/'grid.npz',**arrays)
 manifest['status']='completed with iteration and convergence-warning checks; inspect raw cautions';record()
 print(a.output/'grid.npz',flush=True)


if __name__=='__main__':main()
