"""Expanded JWST near-IR search, including exposure-level MAST services.

The general observation catalogue does not necessarily expose all observing
modes equally. Query instrument metadata and the known NIRISS programme too.
"""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import json
from astroquery.mast import Observations,Mast

OUT=Path('results/cloudy/ngc1068_jwst_near_ir_search_v1')
if OUT.exists():raise FileExistsError(OUT)
OUT.mkdir(parents=True)
Observations.TIMEOUT=45;Mast.TIMEOUT=45
position=[dict(paramName='targ_ra',values=[dict(min=40.6196216667,max=40.7196216667)]),
          dict(paramName='targ_dec',values=[dict(min=-.0632944444,max=.0367055556)])]
report=dict(retrieved_utc=datetime.now(timezone.utc).isoformat(),
    method='Instrument-level target-position box +/-3 arcmin plus known NIRISS programme in observation catalogue. Target coordinates do not establish slit/IFU footprint coverage.',queries={})
queries={
 'nirspec_exposures':lambda:Mast.service_request('Mast.Jwst.Filtered.Nirspec',dict(columns='*',filters=position)),
 'niriss_exposures':lambda:Mast.service_request('Mast.Jwst.Filtered.Niriss',dict(columns='*',filters=position)),
 'niriss_program1260':lambda:Observations.query_criteria(obs_collection='JWST',proposal_id='1260'),
}
for name,query in queries.items():
    try:
        table=query();table.write(OUT/f'{name}.ecsv',format='ascii.ecsv')
        selected=[k for k in ['filename','fileName','targname','target_name','targ_ra','targ_dec','exp_type','filter','pupil','grating','program','proposal_id','dataURI','dataRights','instrument_name','obs_id','dataproduct_type'] if k in table.colnames]
        counts={k:dict(Counter(str(v) for v in table[k])) for k in ['exp_type','filter','pupil','grating','instrument_name','dataproduct_type'] if k in table.colnames}
        item=dict(rows=len(table),columns=table.colnames,counts=counts,
            selected_rows=[{k:str(r[k]) for k in selected} for r in table])
        report['queries'][name]=item
        print(name,len(table),'rows',json.dumps(counts),flush=True)
    except Exception as e:
        report['queries'][name]=dict(error=repr(e));print(name,repr(e),flush=True)
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
