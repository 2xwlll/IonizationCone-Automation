"""Download archived 2D slit product and ESO documentation for aperture auditing."""
from pathlib import Path
import argparse,hashlib,json
import requests
from astropy.io import fits

ROOT=Path(__file__).resolve().parents[2]
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    urls={
        'slit_2d.fits':'https://dataportal.eso.org/dataPortal/file/ADP.2023-09-07T07:09:03.717',
        'archive_readme.txt':'https://dataportal.eso.org/dataPortal/file/ADP.2023-09-07T07:09:03.718',
        'pipeline_manual.pdf':'https://ftp.eso.org/pub/dfs/pipelines/xshooter/xsh-pipeline-manual-3.0.pdf'}
    (out/'requests.json').write_text(json.dumps(urls,indent=2))
    report={}
    for name,url in urls.items():
        print('Downloading',name,flush=True)
        try:
            r=requests.get(url,timeout=(20,150));r.raise_for_status();(out/name).write_bytes(r.content)
            if name.endswith('.fits'):
                with fits.open(out/name) as h:h.verify('exception')
            report[name]={'url':url,'bytes':len(r.content),'sha256':hashlib.sha256(r.content).hexdigest()}
            (out/'manifest.json').write_text(json.dumps(report,indent=2))
        except Exception as e:
            (out/'failure.txt').write_text(repr(e));raise
if __name__=='__main__':main()
