"""Save public archive metadata for two observed ionization-cone reference routes.

This queries observation/product metadata only; it does not download FITS images.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from astroquery.mast import Observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/cloudy/nircam_reference_audit'))
    args = parser.parse_args()
    Observations.cache_location = '/tmp/ioncone_mast_cache'
    Observations.TIMEOUT = 60
    queries = {
        'ngc1068': dict(obs_collection='JWST', proposal_id='3707',
                        instrument_name='NIRCAM*', target_name='NGC1068', dataproduct_type='image'),
        # JADES mosaics are HLSP products, not obs_collection='JWST'.
        'jades182930': dict(provenance_name='JADES',
                            instrument_name='NIRCAM*', filters=['F182M', 'F210M'],
                            coordinates='53.161524 -27.85607', radius='0.01 deg'),
    }
    manifest = {'retrieved_utc': datetime.now(timezone.utc).isoformat(),
                'note': 'Archive associations require subsequent SCI/ERR/DQ/WCS and footprint inspection.',
                'targets': {}}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, query in queries.items():
        observations = Observations.query_criteria(**query)
        observations.write(args.output/f'{name}_observations.ecsv', format='ascii.ecsv', overwrite=True)
        products = Observations.get_product_list(observations) if len(observations) else None
        count = 0
        if products is not None:
            products.write(args.output/f'{name}_products.ecsv', format='ascii.ecsv', overwrite=True)
            count = len(products)
        manifest['targets'][name] = {'query': query, 'observation_count': len(observations),
                                     'product_count': count}
        print(name, 'observations:', len(observations), 'products:', count)
        if len(observations):
            print(observations[['obs_id', 'filters', 'dataRights', 't_exptime', 'obsid']])
        (args.output/'archive_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
