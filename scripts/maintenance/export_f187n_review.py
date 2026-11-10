#!/usr/bin/env python3
"""Export a completed F187N pilot for local review without changing the run.

Standard library only; runs on ARC login-node Python 3.6. Exports reports,
plots, logs, frozen source/inputs and the evaluated test inputs/masks. An
explicit flag adds only the validation-selected checkpoint, not all epochs.
"""
import argparse
import datetime
import hashlib
import io
import json
from pathlib import Path
import tarfile


def digest_stream(stream):
    value = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1048576), b''):
        value.update(chunk)
    return value.hexdigest()


def digest(path):
    with path.open('rb') as stream:
        return digest_stream(stream)


def safe_file(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe review path: '+str(relative))
    path = root/relative
    # All exported entries are regular files within the indicated source tree.
    if path.is_symlink() or root.resolve() not in path.resolve().parents or not path.is_file():
        raise ValueError('Expected a regular file within '+str(root)+': '+str(relative))
    return path


def verify_archive(path):
    """Verify every archived byte without extracting or importing any content."""
    with tarfile.open(str(path), 'r:gz') as archive:
        members = archive.getmembers()
        if any(not member.isfile() for member in members):
            raise ValueError('Review archive contains non-regular entries')
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError('Review archive contains duplicate entries')
        for name in names:
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('Unsafe archived path')
        manifest = json.load(archive.extractfile('EXPORT_MANIFEST.json'))
        records = manifest['files']
        if set(names) != set(records) | {'EXPORT_MANIFEST.json'}:
            raise ValueError('Review archive inventory mismatch')
        for name, record in records.items():
            if archive.getmember(name).size != record['bytes']:
                raise ValueError('Review archive size mismatch: '+name)
            with archive.extractfile(name) as stream:
                if digest_stream(stream) != record['sha256']:
                    raise ValueError('Review archive hash mismatch: '+name)
    return manifest


def export_review(run_root, output, include_checkpoint=False):
    run_root, output = run_root.resolve(), output.resolve()
    if output.exists() or Path(str(output)+'.sha256').exists():
        raise ValueError('Export already exists; choose a fresh output filename')
    model_root = run_root/'training/model'
    config = json.loads((model_root/'config.json').read_text())
    report = json.loads((model_root/'test_report.json').read_text())
    selection = json.loads((model_root/'selection.json').read_text())
    dataset = Path(config['dataset']).resolve()
    manifest_path = safe_file(dataset, 'manifest.json')
    if digest(manifest_path) != config['dataset_manifest_sha256']:
        raise ValueError('Dataset manifest differs from the evaluated dataset')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('band') != 'F187N' or manifest.get('projection') != 'circular_cone_envelope_v2':
        raise ValueError('Expected the corrected F187N image pipeline')
    samples = {row['id']: row for row in manifest['samples']}
    ids = [row['id'] for row in report['samples']]
    expected = {row['id'] for row in manifest['samples']
                if row['split'] == 'test' and row['variant'] in (config['variant'], 'cone_free')}
    if len(ids) != len(set(ids)) or set(ids) != expected or report['count'] != len(ids):
        raise ValueError('Report does not cover the expected held-out test images')
    files = {}

    def add(root, relative, prefix, expected_hash=None):
        path = safe_file(root, relative)
        if expected_hash is not None and digest(path) != expected_hash:
            raise ValueError('Changed source file: '+str(path))
        files[str(Path(prefix)/relative)] = path

    add(dataset, 'manifest.json', 'dataset')
    add(dataset, 'plan.json', 'dataset', manifest['plan_sha256'])
    add(dataset, manifest['valid_file'], 'dataset', manifest['valid_sha256'])
    for identifier in ids:
        row = samples[identifier]
        for field in ('image', 'mask'):
            add(dataset, row[field], 'dataset', row[field+'_sha256'])
    add(run_root, 'snapshot.json', 'run')
    frozen = json.loads((run_root/'snapshot.json').read_text())
    for relative, expected_hash in frozen['sha256'].items():
        if not relative.startswith('code/'):
            raise ValueError('Unexpected frozen-source path: '+relative)
        add(run_root, relative, 'run', expected_hash)
    for folder in ('training', 'logs', 'submissions'):
        for path in sorted((run_root/folder).rglob('*')):
            if path.is_file() and path.suffix in ('.json', '.jsonl', '.csv', '.png', '.log'):
                add(run_root, str(path.relative_to(run_root)), 'run')
    best_epoch = selection['best_epoch']
    if not isinstance(best_epoch, int) or not 1 <= best_epoch <= config['epochs']:
        raise ValueError('Invalid selected checkpoint epoch')
    if include_checkpoint:
        add(run_root, 'training/model/epoch_{:03d}.pth'.format(best_epoch), 'run')
    readme = (
        'F187N completed-pilot review export\n\n'
        'run/ contains original reports, plots, logs and frozen code/scientific inputs.\n'
        'dataset/ contains ORIGINAL manifest/plan metadata but ONLY evaluated test\n'
        'image/mask arrays and the common valid mask. It is a review subset, not a\n'
        'complete training dataset; the full-manifest training validator will fail.\n'
        'Absolute paths in original metadata describe ARC and are preserved.\n'
        'Predictions for every case can be recomputed locally from the selected\n'
        'checkpoint if included; test_predictions.png is only the original preview.\n'
        'No environments, other epoch checkpoints or large per-galaxy component\n'
        'archives are exported. Training arrays are not exported.\n'
        'A completed shared-generator pilot is not real-galaxy validation.\n'
    ).encode('utf-8')
    records = {name: {'sha256': digest(path), 'bytes': path.stat().st_size}
               for name, path in sorted(files.items())}
    records['README_REVIEW.txt'] = {'sha256': hashlib.sha256(readme).hexdigest(), 'bytes': len(readme)}
    export_manifest = {
        'schema': 'f187n_review_export_v1',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_run': str(run_root), 'source_dataset': str(dataset),
        'test_ids': ids, 'selected_epoch': best_epoch,
        'selected_checkpoint_included': include_checkpoint,
        'dataset_is_test_subset': True, 'files': records}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves earlier exports and leaves partial evidence on failure.
    with output.open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w:gz', dereference=False) as archive:
            for name, path in sorted(files.items()):
                archive.add(str(path), arcname=name, recursive=False)
            for name, value in (('README_REVIEW.txt', readme),
                                ('EXPORT_MANIFEST.json', (json.dumps(export_manifest, indent=2)+'\n').encode('utf-8'))):
                member = tarfile.TarInfo(name)
                member.size = len(value)
                archive.addfile(member, io.BytesIO(value))
    verify_archive(output)
    with Path(str(output)+'.sha256').open('x') as stream:
        stream.write(digest(output)+'  '+output.name+'\n')
    return {'status': 'verified', 'archive': str(output), 'bytes': output.stat().st_size,
            'sha256': digest(output), 'test_images': len(ids), 'selected_epoch': best_epoch,
            'selected_checkpoint_included': include_checkpoint}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--include-selected-checkpoint', action='store_true')
    parser.add_argument('--verify', type=Path, help='Verify an exported archive without extracting')
    args = parser.parse_args()
    if args.verify:
        manifest = verify_archive(args.verify)
        print('Verified {} files and {} held-out images.'.format(len(manifest['files']), len(manifest['test_ids'])))
    else:
        if args.run_root is None or args.output is None:
            parser.error('--run-root and --output are required for export')
        print(json.dumps(export_review(args.run_root, args.output, args.include_selected_checkpoint), indent=2))


if __name__ == '__main__':
    main()
