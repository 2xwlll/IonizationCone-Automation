"""Inspect a phenomenological nuclear spectrum through measured NIRCam responses."""
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('MPLCONFIGDIR','/tmp/ioncone-matplotlib')
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.simple_utils.nuclear_sed import nuclear_flambda_shape,nuclear_filter_response
from src.simple_utils.synthetic_photometry import photon_bandpass_weights
from src.jwst_utils.nircam import load_filters


def main():
    out=Path('results/cloudy/nuclear_sed_pilot_v1')
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    cfg={'status':'illustrative obscured hot-dust plus power-law continuum; not fitted to NGC 1068',
         'reference_rest_angstrom':18751.,'hot_dust_temperature_k':1100.,
         'hot_dust_fraction_at_reference':.9,'powerlaw_alpha_nu':-.5,
         'tau_at_reference':1.,'extinction_exponent':1.3}
    z=.003793
    wave=np.linspace(6000,55000,6000)
    flambda=nuclear_flambda_shape(wave/(1+z),cfg)/(1+z)
    filters=load_filters()
    rows=[]
    for name,info in filters.items():
        fw,t=info['curve'].T
        response=nuclear_filter_response(fw,t,z,cfg)
        rows.append({'filter':name,'photon_response_per_unit_rest_reference_flambda':response,
                     'photon_weighted_mean_flambda':response/float(photon_bandpass_weights(fw,fw,t).sum())})
    (out/'model.json').write_text(json.dumps({'configuration':cfg,'redshift':z,'filter_predictions':rows,
        'normalization':'dimensionless rest F_lambda(reference)=1; no absolute luminosity or observer flux',
        'limitations':['No stellar continuum or emission lines in this nuclear SED',
                       'Not a radiative-transfer torus model; no measured dust temperature/extinction fit',
                       'No broad Pa alpha added for this obscured Seyfert 2 nucleus']},indent=2)+'\n')
    np.savez_compressed(out/'spectrum.npz',observed_angstrom=wave,relative_flambda=flambda)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    ax.plot(wave/1e4,flambda,color='black',label='Assumed obscured nuclear continuum')
    ax.set_xlabel('Observed wavelength (micrometers)');ax.set_ylabel('Relative F_lambda (arbitrary normalization)')
    ax.set_yscale('log');ax.set_ylim(.003,100)
    secondary=ax.twinx()
    for name in ('F164N','F187N','F212N','F300M','F335M','F405N'):
        fw,t=filters[name]['curve'].T
        secondary.plot(fw/1e4,t,label=name,alpha=.75,linewidth=1.2)
    secondary.set_ylabel('Total system throughput');secondary.set_ylim(0,.7)
    secondary.legend(loc='upper left',ncol=3,fontsize=9)
    ax.legend(loc='lower right');ax.set_xlim(.6,5.3)
    ax.set_title('Wavelength-dependent nuclear light: assumed spectrum, measured filter responses')
    fig.savefig(out/'spectrum.png',dpi=160)
    plt.close(fig)
    # Separate versioned example; old reference calibration is not reused.
    galaxy=json.loads(Path('configs/cloudy/ngc1068_f187n_pilot_v2.json').read_text())
    galaxy['nuclear_sed']=cfg
    galaxy['output']='results/cloudy/ngc1068_f187n_nuclear_sed_v1'
    Path('configs/cloudy/ngc1068_f187n_nuclear_sed_v1.json').write_text(json.dumps(galaxy,indent=2)+'\n')
    print(out/'spectrum.png')


if __name__=='__main__':main()
