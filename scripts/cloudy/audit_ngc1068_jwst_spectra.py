"""Query all JWST observations near NGC 1068, retaining native archive metadata."""
from datetime import datetime,timezone
from pathlib import Path
from collections import Counter
import json
from astroquery.mast import Observations

out=Path('results/cloudy/ngc1068_jwst_spectral_audit_v2')
if out.exists():raise FileExistsError(out)
out.mkdir(parents=True)
Observations.cache_location='/tmp/ioncone_mast_cache'
Observations.TIMEOUT=45
query=dict(obs_collection='JWST',coordinates='40.6696216667 -0.0132944444',radius='0.05 deg')
obs=Observations.query_criteria(**query)
obs.write(out/'all_nearby_jwst.ecsv',format='ascii.ecsv')
summary=Counter((str(r['instrument_name']),str(r['dataproduct_type'])) for r in obs)
spec=obs[[str(r['dataproduct_type']).lower() in ('spectrum','cube') for r in obs]]
spec.write(out/'spectral_candidates.ecsv',format='ascii.ecsv')
cols=[c for c in ['obs_id','target_name','instrument_name','filters','dataproduct_type','dataRights','em_min','em_max','proposal_id','obsid'] if c in obs.colnames]
report=dict(retrieved_utc=datetime.now(timezone.utc).isoformat(),query=query,observations=len(obs),
    instrument_type_counts=[dict(instrument=a,product_type=b,count=n) for (a,b),n in summary.items()],
    spectral_candidates=[{c:str(r[c]) for c in cols} for r in spec],
    limitation='Archive positional query, not an exhaustive literature survey or verification that every footprint covers the same aperture. No non-JWST data substituted.')
(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
