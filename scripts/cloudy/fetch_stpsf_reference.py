"""Download STScI's minimal STPSF 2.2 reference data into the project."""

from hashlib import sha256
import json
from pathlib import Path
import shutil
import tarfile

import requests

ROOT = Path(__file__).resolve().parents[2]
URL = 'https://stsci.box.com/shared/static/994a5c7uxc9u9w2li9cwwh440cumbklm.gz'
DEST = ROOT/'data/tools/stpsf-reference'
ARCHIVE = ROOT/'data/tools/minimal-stpsf-data-2.2.0.tar.gz'


def main():
    if DEST.exists() and any(DEST.iterdir()):
        raise FileExistsError(f'Reference destination already exists: {DEST}')
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256()
    if ARCHIVE.exists():
        with ARCHIVE.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024*1024), b''):
                digest.update(chunk)
    else:
        with requests.get(URL, stream=True, timeout=90) as response:
            response.raise_for_status()
            with ARCHIVE.open('wb') as handle:
                for chunk in response.iter_content(1024*1024):
                    digest.update(chunk)
                    handle.write(chunk)
    # Refuse links and paths outside the destination; extract regular data only.
    with tarfile.open(ARCHIVE, 'r:gz') as archive:
        members = archive.getmembers()
        for member in members:
            target = (DEST/member.name).resolve()
            if not target.is_relative_to(DEST.resolve()):
                raise ValueError(f'Unsafe archive path: {member.name}')
            if not (member.isfile() or member.isdir()):
                raise ValueError(f'Unexpected archive entry: {member.name}')
        DEST.mkdir(parents=True, exist_ok=True)
        for member in members:
            target = DEST/member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open('wb') as output:
                    shutil.copyfileobj(source, output)
    (DEST/'download_manifest.json').write_text(json.dumps({
        'source_url': URL, 'sha256': digest.hexdigest(),
        'documentation': 'https://stpsf.readthedocs.io/en/stable/installation.html',
        'archive_bytes': ARCHIVE.stat().st_size,
    }, indent=2)+'\n')
    print('Reference data:', DEST)
    print('Version files:', list(DEST.rglob('version.txt')))


if __name__ == '__main__':
    main()
