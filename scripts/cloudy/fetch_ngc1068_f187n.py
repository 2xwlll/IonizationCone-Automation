"""Download the public NGC 1068 F187N stage-3 mosaic without loading it into RAM."""

from hashlib import sha256
import json
from pathlib import Path

from astropy.io import fits
import requests

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT/'data/2d/jwst_ngc1068'
NAME = 'jw03707-o128_t002_nircam_clear-f187n_i2d.fits'
URL = 'https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product/'+NAME


def main():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    destination = DIRECTORY/NAME
    if destination.exists():
        print('Reusing existing mosaic:', destination)
    else:
        partial = destination.with_suffix('.fits.part')
        if partial.exists():
            raise FileExistsError(f'Incomplete earlier download exists: {partial}')
        digest = sha256()
        count, last = 0, 0
        with requests.get(URL, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with partial.open('xb') as handle:
                for chunk in response.iter_content(1024*1024):
                    handle.write(chunk)
                    digest.update(chunk)
                    count += len(chunk)
                    if count-last >= 64*1024*1024:
                        print(f'Downloaded {count/1024**2:.0f} MiB', flush=True)
                        last = count
        with fits.open(partial, memmap=True) as hdul:
            hdul.verify('exception')
            if hdul[0].header.get('FILTER') != 'F187N' or 'SCI' not in hdul:
                raise ValueError('Downloaded file is not the expected F187N science mosaic')
        partial.rename(destination)
        destination.with_suffix('.download.json').write_text(json.dumps({
            'source_url': URL, 'data_uri': 'mast:JWST/product/'+NAME,
            'sha256': digest.hexdigest(), 'bytes': count,
        }, indent=2)+'\n')
    with fits.open(destination, memmap=True) as hdul:
        print('Filter:', hdul[0].header.get('FILTER'))
        print('Science shape:', hdul['SCI'].shape)
        print('Science units:', hdul['SCI'].header.get('BUNIT'))
        print('Extensions:', [h.name for h in hdul])
    print('Saved:', destination)


if __name__ == '__main__':
    main()
