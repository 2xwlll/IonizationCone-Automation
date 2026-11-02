"""Prepare or run a separate stellar-ionized CLOUDY grid for F187N.

A hot-star blackbody is a controlled first approximation, not a fitted stellar
population. Inputs and every raw output are retained. No failed job is retried.
"""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.jwst_utils.nircam import load_filters
from src.simple_utils.wavelengths import cloudy_air_to_vacuum
from src.simple_utils.f187n import continuum_from_cached_file

# Explicit high-precision IR targets plus optical diagnostics. This is a
# deliberately incomplete faint-line inventory, never presented as all lines.
LINES=[('H 1',18751.,'H  1  1.87510m'),('He 1',18685.4,'He 1  1.86854m'),
       ('He 1',18697.2,'He 1  1.86972m'),('H 1',4861.32,'H  1  4861.32A'),
       ('O 3',5006.84,'O  3  5006.84A')]


def model_input(model,cfg):
    return '\n'.join([
        f'title "Stellar-ionized F187N pilot {model["model_id"]}"',
        f'blackbody {model["stellar_temperature_k"]:.1f} K',
        f'ionization parameter {model["log_u"]:.5f}',
        f'hden {model["log_nh_cm3"]:.5f}',
        'abundances ism no grains',
        f'metals {math.log10(model["metallicity_solar"]):.6f} log',
        f'grains ism {math.log10(model["dust_scale"]):.6f} log no qheat single',
        f'stop column density {cfg["log_column_density_cm2"]:.5f}',
        'iterate to convergence',
        'print last iteration',
        f'save line list "{model["model_id"]}.lin" "hii_lines.dat" last no hash emergent absolute',
        f'save continuum "{model["model_id"]}.con" last units Angstrom',''])


def final_saved_lines(path):
    rows=[line.split('\t') for line in path.read_text().splitlines()
          if line.lower().startswith('iteration')]
    if not rows or len(rows[-1])!=len(LINES)+1:
        raise ValueError('Unexpected saved line-list format: '+str(path))
    iteration=int(rows[-1][0].split()[-1])
    values=np.array([float(x) for x in rows[-1][1:]])
    if iteration<2 or not np.isfinite(values).all() or np.any(values<0):
        raise ValueError('Invalid or first-iteration line intensities: '+str(path))
    return iteration,values


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/cloudy/hii_f187n_pilot_v1.json'))
    parser.add_argument('--output',type=Path,default=Path('data/cloudy/hii_f187n_pilot_v1'))
    parser.add_argument('--cloudy',type=Path,help='Omit to prepare inputs without running CLOUDY')
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh output directory; no overwrite/retry')
    cfg=json.loads(args.config.read_text())
    executable=args.cloudy.resolve() if args.cloudy else None
    if executable and not executable.is_file(): raise FileNotFoundError(executable)
    runs=args.output/'runs';runs.mkdir(parents=True)
    (runs/'hii_lines.dat').write_text('\n'.join(line[2]+' air' for line in LINES)+'\n')
    keys=['stellar_temperature_k','log_u','log_nh_cm3','metallicity_solar']
    combos=list(itertools.product(*(cfg[k] for k in keys)))
    if args.limit: combos=combos[:args.limit]
    models=[]
    for values in combos:
        model=dict(zip(keys,map(float,values)))
        model.update(dust_scale=cfg['dust_scale'],pah_scale=0.)
        model['model_id']='hii_'+hashlib.sha256(json.dumps(model,sort_keys=True).encode()).hexdigest()[:12]
        (runs/(model['model_id']+'.in')).write_text(model_input(model,cfg))
        models.append(model)
    manifest={'status':'prepared only; no CLOUDY calculations run','configuration':cfg,
              'models':models,'line_list':LINES,'line_wavelength_medium':'vacuum','line_list_input_medium':'air','line_inventory':'targeted Pa alpha and helium lines only, plus optical diagnostics; incomplete weak-line inventory',
              'ionizing_source':'hot-star blackbody, not an AGN and not a population synthesis spectrum',
              'executable':str(executable) if executable else None}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if not executable:
        print(f'Prepared {len(models)} stellar-ionized inputs: {args.output}');return
    manifest['executable_sha256']=hashlib.sha256(executable.read_bytes()).hexdigest()
    fw,throughput=load_filters()['F187N']['curve'].T
    rest_wave=fw/(1+cfg['redshift'])
    fluxes=[];continua=[]
    manifest.update(status='running; no compact grid available yet',completed_model_ids=[])
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def fail(message):
        manifest.update(status='failed; raw outputs preserved; no automatic retry',failure=message)
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        raise RuntimeError(message)

    for model in models:
        stem=model['model_id']
        result=subprocess.run([str(executable),'-r',stem],cwd=runs,text=True,capture_output=True,
                              env={**os.environ,'OMP_NUM_THREADS':'1'})
        (runs/(stem+'.stdout.txt')).write_text(result.stdout)
        (runs/(stem+'.stderr.txt')).write_text(result.stderr)
        if result.returncode: fail('CLOUDY failed; raw outputs preserved: '+stem)
        text=(runs/(stem+'.out')).read_text()
        if 'Cloudy exited OK' not in text: fail('No successful exit marker: '+stem)
        if re.search(r'not converged|did not converge|failed to converge|convergence failure|not reliable on the first iteration',text,re.I):
            fail('Convergence warning requires review; not packing model: '+stem)
        try:
            iteration,values=final_saved_lines(runs/(stem+'.lin'))
            continuum=continuum_from_cached_file(runs/(stem+'.con'),rest_wave)
        except (ValueError,FileNotFoundError) as error:
            fail(f'Output validation failed for {stem}: {error}')
        model['iterations']=iteration
        model['out_sha256']=hashlib.sha256(text.encode()).hexdigest()
        fluxes.append(values)
        continua.append(continuum)
        manifest['completed_model_ids'].append(stem)
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(f'Completed {stem}, {iteration} iterations',flush=True)
    arrays={k:np.array([m[k] for m in models]) for k in ['model_id',*keys,'dust_scale','pah_scale']}
    arrays.update(line_flux=np.asarray(fluxes)[:,:3],line_labels=np.array([x[0] for x in LINES[:3]]),
                  rest_line_angstrom=cloudy_air_to_vacuum(np.array([x[1] for x in LINES[:3]])),
                  rest_line_air_angstrom=np.array([x[1] for x in LINES[:3]]),
                  continuum_rest_flambda=np.array(continua),continuum_rest_angstrom=rest_wave,
                  filter_wavelength_angstrom=fw,throughput=throughput,
                  h_beta=np.asarray(fluxes)[:,3],oiii_5007=np.asarray(fluxes)[:,4])
    np.savez_compressed(args.output/'grid.npz',**arrays)
    manifest.update(status='completed stellar-ionized pilot with iteration and convergence-warning checks; inspect raw cautions before scientific use',models=models)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Saved '+str(args.output/'grid.npz'),flush=True)


if __name__=='__main__': main()
