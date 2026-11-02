"""Recover a relative near-IR pilot grid from cached Cloudy diagnostic outputs.

This is a provenance-preserving prototype, not a new converged Cloudy run.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.jwst_utils.nircam import load_filters
from src.simple_utils.f187n import continuum_from_cached_file, printed_emergent_lines


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('data/cloudy/ngc1068_oiii_full_v1'))
    p.add_argument('--output', type=Path, default=Path('data/cloudy/ngc1068_f187n_pilot_v1'))
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new pilot-grid directory')
    manifest = json.loads((args.source/'manifest.json').read_text())
    models = manifest['models']
    fw, throughput = load_filters()['F187N']['curve'].T
    redshift = .003793
    rest_wave = fw/(1+redshift)
    all_lines, sources, errors = [], [], []
    for model in models:
        source = args.source/'runs'/f"{model['model_id']}.out"
        raw = source.read_text()
        if 'Cloudy exited OK' not in raw:
            raise ValueError(f'No successful Cloudy completion marker: {source}')
        lines = printed_emergent_lines(raw)
        if ('H 1', 18751.) not in lines:
            raise ValueError(f'Missing Pa alpha: {source}')
        for label, wave, key in [('H 1',4861.32,'H_beta_4861'),('O 3',4958.91,'OIII_4959'),('O 3',5006.84,'OIII_5007')]:
            recovered = lines[(label,wave)]
            errors.append(abs(recovered/model[key]-1))
        all_lines.append({key:value for key,value in lines.items() if rest_wave[0] <= key[1] <= rest_wave[-1]})
        sources.append({'model_id':model['model_id'],'source_out':str(source),
                        'sha256':sha256(source.read_bytes()).hexdigest(),
                        'first_iteration_warning':'emergent intensities are not reliable on the first iteration' in raw})
    if max(errors) > .0013:
        raise ValueError(f'Printed intensities do not reproduce compact optical grid: {max(errors)}')
    keys = sorted(set().union(*(set(lines) for lines in all_lines)), key=lambda key:key[1])
    # NaN distinguishes an unprinted line from a measured zero flux.
    flux = np.array([[lines.get(key,np.nan) for key in keys] for lines in all_lines])
    continua = np.array([continuum_from_cached_file(args.source/'runs'/f"{m['model_id']}.con",rest_wave) for m in models])
    args.output.mkdir(parents=True)
    arrays = {name:np.array([m[name] for m in models]) for name in ['model_id','log_u','log_nh_cm3','metallicity_solar','dust_scale','pah_scale']}
    arrays.update(line_flux=flux,line_labels=np.array([key[0] for key in keys]),
                  rest_line_angstrom=np.array([key[1] for key in keys]),
                  continuum_rest_flambda=continua,continuum_rest_angstrom=rest_wave,
                  filter_wavelength_angstrom=fw,throughput=throughput)
    np.savez_compressed(args.output/'grid.npz',**arrays)
    report = dict(status='relative prototype extracted from printed diagnostics',redshift=redshift,
                  model_count=len(models),line_count=len(keys),missing_line_entries=int(np.isnan(flux).sum()),
                  max_optical_recovery_relative_error=float(max(errors)),sources=sources,
                  first_iteration_warning_models=sum(s['first_iteration_warning'] for s in sources),
                  limitations=['Printed log fluxes have finite rounding precision',
                               'Missing lines are censored/unknown; no complete faint-line inventory',
                               'Original one-iteration Cloudy convergence warning remains unresolved',
                               'AGN-illuminated clouds only; no dedicated H II-region grid',
                               'No absolute luminosity, covering factor or observer flux calibration'])
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print({key:value for key,value in report.items() if key!='sources'})


if __name__=='__main__':
    main()
