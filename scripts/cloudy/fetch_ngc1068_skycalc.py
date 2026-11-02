"""Fetch independent ESO atmospheric models, preserving inputs and raw output."""
import argparse
from pathlib import Path
import sys
import json
import hashlib
import requests
from datetime import datetime,timezone
from astropy.io import fits

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'data/tools/skycalc-runtime'))
from skycalc_cli.skycalc import SkyModel


# The upstream client sets no HTTP timeout and exits successfully on some
# errors. Bound requests and make missing output a failed scientific run.
_post=requests.post
_get=requests.get
requests.post=lambda *a,**kw:_post(*a,**dict({'timeout':(15,90)},**kw))
requests.get=lambda *a,**kw:_get(*a,**dict({'timeout':(15,90)},**kw))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'data/spectra/ngc1068_skycalc_v1')
    p.add_argument('--base-url',default=None)
    p.add_argument('--spectrum',type=Path,default=ROOT/'data/spectra/ngc1068_eso_xshooter_v1/3273695/spectrum.fits')
    p.add_argument('--pwv',type=float,nargs='+',default=[3.5,5.,7.5])
    args=p.parse_args();out=args.output.resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    header=fits.getheader(args.spectrum)
    report=dict(retrieved_utc=datetime.now(timezone.utc).isoformat(),tool='ESO skycalc_cli 1.5',
        source='https://www.eso.org/observing/etc/doc/skycalc/helpskycalccli.html',
        observed_airmass=float(header['ESO QC AIRM MEAN']),
        observed_pwv_mm=[header['ESO TEL AMBI IWV START'],header['ESO TEL AMBI IWV END']],models=[])
    report['observation']=str(args.spectrum.resolve())
    for pwv in args.pwv:
        m=SkyModel(base_url=args.base_url)
        m.params.update(airmass=float(header['ESO QC AIRM MEAN']),pwv_mode='pwv',pwv=pwv,
            vacair='vac',wmin=1735.,wmax=2020.,wgrid_mode='fixed_wavelength_step',wdelta=.006,
            lsf_type='none',observatory='paranal',incl_moon='N',incl_starlight='N',incl_zodiacal='N',
            incl_loweratm='N',incl_upperatm='N',incl_airglow='N',incl_therm='N')
        stem=f'pwv_{pwv:.1f}'
        (out/(stem+'_input.json')).write_text(json.dumps(m.params,indent=2)+'\n')
        print('Request',pwv,'mm at',m.base_url,flush=True)
        try:
            m.call()
        except SystemExit as exc:
            raise RuntimeError('ESO SkyCalc client exited before returning a verified model') from exc
        if not m.data:raise RuntimeError('ESO SkyCalc returned no FITS bytes')
        m.write(str(out/(stem+'.fits')))
        path=out/(stem+'.fits')
        with fits.open(path) as h: h.verify('exception');print(h[1].columns.names,flush=True)
        report['models'].append(dict(pwv_mm=pwv,file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),base_url=m.base_url))
        (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
