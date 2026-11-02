"""Audit MAST spectral candidates near NGC 1068 without assuming coverage."""
import json
from pathlib import Path
from datetime import datetime,timezone
from astroquery.mast import Observations

out=Path('results/cloudy/ngc1068_spectroscopy_audit_v1')
out.mkdir(parents=True,exist_ok=True)
Observations.cache_location='/tmp/ioncone_mast_cache'
Observations.TIMEOUT=60
queries={
 'jwst_nirspec':dict(obs_collection='JWST',instrument_name='NIRSPEC*',coordinates='40.6696216667 -0.0132944444',radius='0.05 deg'),
 'hst_nicmos':dict(obs_collection='HST',instrument_name='NICMOS*',coordinates='40.6696216667 -0.0132944444',radius='0.03 deg')}
report={'retrieved_utc':datetime.now(timezone.utc).isoformat(),'targets':{},'note':'Positional associations alone do not confirm nuclear coverage, usable spectra or public product availability.'}
for name,query in queries.items():
 try:
  obs=Observations.query_criteria(**query)
  obs.write(out/(name+'_observations.ecsv'),format='ascii.ecsv',overwrite=True)
  cols=[c for c in ['obs_id','target_name','instrument_name','filters','dataproduct_type','dataRights','t_min','t_exptime','obsid'] if c in obs.colnames]
  print(name,len(obs),flush=True)
  if len(obs):print(obs[cols],flush=True)
  report['targets'][name]={'query':query,'observation_count':len(obs)}
 except Exception as e:
  report['targets'][name]={'query':query,'error':str(e)}
  print(name,type(e).__name__,str(e),flush=True)
 (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
