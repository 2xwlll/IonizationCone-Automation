#!/usr/bin/env python3
"""Retrieve public ESO X-shooter spectral products and their DataLink metadata."""
from datetime import datetime, timezone
import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path

import requests
from astropy.table import Table

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/spectra/ngc1068_eso_xshooter_v1'
AUDIT=ROOT/'results/cloudy/ngc1068_eso_spectral_audit_v1'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--obs-id',action='append',default=[])
    args=parser.parse_args()
    output=args.output.resolve()
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    query=json.loads((AUDIT/'nearby_all_spectral.txt').read_text())
    names=[v['name'] for v in query['metadata']]
    rows=[dict(zip(names,v)) for v in query['data']]
    session=requests.Session()
    manifest={'retrieved_utc':datetime.now(timezone.utc).isoformat(),'query_source':str(AUDIT.relative_to(ROOT)),'products':[]}
    for row in rows:
        if row['instrument_name']!='XSHOOTER' or row['dataproduct_type']!='spectrum':continue
        if args.obs_id and row['obs_id'] not in args.obs_id:continue
        folder=output/row['obs_id'];folder.mkdir()
        response=session.get(row['access_url'],timeout=30);response.raise_for_status()
        (folder/'datalink.xml').write_bytes(response.content)
        table=Table.read(BytesIO(response.content),format='votable')
        links=[]
        for entry in table:
            item={key:str(entry[key]) for key in table.colnames}
            links.append(item)
        (folder/'links.json').write_text(json.dumps(links,indent=2)+'\n')
        primary=[item for item in links if item.get('semantics')=='#this' and item.get('access_url') not in ('','--')]
        if len(primary)!=1:raise ValueError(f'Ambiguous primary product: {primary}')
        url=primary[0]['access_url']
        product=session.get(url,timeout=120);product.raise_for_status()
        path=folder/'spectrum.fits';path.write_bytes(product.content)
        item={'query':row,'download_url':url,'local_file':str(path.relative_to(ROOT)),
              'bytes':len(product.content),'sha256':hashlib.sha256(product.content).hexdigest()}
        manifest['products'].append(item)
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(row['obs_id'],row['target_name'],len(product.content),flush=True)


if __name__=='__main__':main()
