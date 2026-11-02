#!/usr/bin/env python3
"""Back up local research outputs that a pending Git update would untrack.

Run from the repository before pulling the cleanup commit. This only reads Git
and existing outputs; it does not reset files, pull changes, or restore an archive.
Symlinks are preserved as links, not followed into the separately managed data.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', default='origin/master')
    parser.add_argument('--staged', action='store_true', help='Back up staged removals instead of a remote update')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip())
    command = ['git', 'diff', '--name-only', '--diff-filter=D', '-z']
    command += ['--cached'] if args.staged else ['HEAD', args.revision]
    command += ['--', 'results/', 'saved_runs/']
    paths = [p for p in subprocess.check_output(command, cwd=root).decode().split('\0') if p]
    if not paths:
        print('No tracked research outputs are being removed by this update.')
        return
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = args.output or root/'.local_backups'/f'removed-outputs-{stamp}'
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                'comparison': 'staged deletions' if args.staged else args.revision,
                'symlinks': 'preserved as links; target datasets are not copied', 'files': [], 'missing': []}
    archive = output/'outputs.tar.gz'
    with tarfile.open(archive, 'w:gz', dereference=False) as tar:
        for name in paths:
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe output path in Git diff')
            source = root/relative
            if source.is_symlink():
                record = {'path': name, 'link_target': str(source.readlink())}
            elif source.is_file():
                record = {'path': name, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
            elif not source.exists():
                manifest['missing'].append(name)
                continue
            else:
                raise ValueError(f'Unexpected output type: {name}')
            tar.add(source, arcname=name, recursive=False)
            manifest['files'].append(record)
    # Read the archive back, verifying bytes or link targets without extracting.
    with tarfile.open(archive, 'r:gz') as tar:
        for record in manifest['files']:
            member = tar.getmember(record['path'])
            if 'link_target' in record:
                if not member.issym() or member.linkname != record['link_target']:
                    raise ValueError('Archived link differs from local link')
            else:
                with tar.extractfile(member) as stream:
                    digest = hashlib.sha256(stream.read()).hexdigest()
                if digest != record['sha256']:
                    raise ValueError('Archived file differs from local bytes')
    manifest['verified'] = True
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Verified backup of {len(manifest["files"])} files/links: {archive}')
    if manifest['missing']:
        print(f'{len(manifest["missing"])} paths were already absent locally; see manifest.json.')
    print('Original local files have not been changed. Keep this backup before pulling.')


if __name__ == '__main__':
    main()
