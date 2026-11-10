#!/usr/bin/env python3
"""Freeze and submit one GPU job for small F187N fit-set diagnostics (Python 3.6)."""
import argparse
import datetime
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.maintenance.launch_f187n_pipeline import snapshot, verify_snapshot, digest, save, job_id
from scripts.maintenance.retry_f187n_training import check_inactive


def select_fit_samples(manifest, count, seed, variants):
    """Choose the same training groups for every representation; never use holdouts."""
    if count < 1 or not variants or len(variants) != len(set(variants)) or any(
            v not in ('smooth_control', 'clumpy_cone', 'full') for v in variants):
        raise ValueError('Invalid fit groups or variants')
    rows = [r for r in manifest['samples'] if r['split'] == 'train']
    lookup = {(r['group'], r['variant']): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError('Duplicate training group/variant records')
    available = sorted({r['group'] for r in rows})
    if count > len(available):
        raise ValueError('Not enough training groups for the fit diagnostic')
    groups = sorted(random.Random(seed).sample(available, count))
    selections = {}
    for variant in variants:
        selected = []
        for group in groups:
            for name in (variant, 'cone_free'):
                if (group, name) not in lookup:
                    raise ValueError('Missing training control: '+group+' '+name)
                selected.append(lookup[group, name]['id'])
        selections[variant] = selected
    return {'groups': groups, 'sample_ids_by_variant': selections}


def prepare(source, target, groups=None, epochs=None):
    original = verify_snapshot(source)
    trained = json.loads((source/'training/model/config.json').read_text())
    dataset = Path(trained['dataset']).resolve()
    if digest(dataset/'manifest.json') != trained['dataset_manifest_sha256']:
        raise ValueError('Dataset changed since the source training run')
    manifest = json.loads((dataset/'manifest.json').read_text())
    if manifest.get('band') != 'F187N' or manifest.get('projection') != 'circular_cone_envelope_v2':
        raise ValueError('Expected the corrected F187N image pipeline')
    diagnostic = json.loads((ROOT/'configs/cloudy/f187n_overfit_diagnostic_v1.json').read_text())
    if groups is not None: diagnostic['fit_groups'] = groups
    if epochs is not None: diagnostic['max_epochs'] = epochs
    if diagnostic['max_epochs'] < 1:
        raise ValueError('Require at least one diagnostic epoch')
    training = json.loads((source/'code/configs/cloudy/f187n_train_launch.json').read_text())
    selection = select_fit_samples(manifest, diagnostic['fit_groups'], training['seed'], diagnostic['variants'])
    settings = snapshot(target, Path(original['environment']), original['groups'],
        diagnostic['max_epochs'], training['variant'], False,
        scientific_root=source/'code', training_settings=training)
    name = 'code/configs/cloudy/f187n_overfit_launch.json'
    save(target/name, diagnostic)
    settings['sha256'][name] = digest(target/name)
    settings['overfit_diagnostic'] = {
        'source_run': str(source), 'source_snapshot_sha256': digest(source/'snapshot.json'),
        'dataset': str(dataset), 'dataset_manifest_sha256': trained['dataset_manifest_sha256'],
        'dataset_plan_sha256': digest(dataset/'plan.json'), 'fit_selection': selection,
        'scope': diagnostic['scope']}
    save(target/'snapshot.json', settings)
    verify_snapshot(target)
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--fit-groups', type=int)
    parser.add_argument('--max-epochs', type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    source, target = args.source_run.resolve(), args.run_root.resolve()
    if source == target: raise ValueError('Choose a fresh diagnostic output directory')
    if not args.dry_run:
        if shutil.which('sbatch') is None: raise RuntimeError('sbatch unavailable; use --dry-run locally')
        check_inactive(source)
    settings = prepare(source, target, args.fit_groups, args.max_epochs)
    command = ['sbatch', '--parsable', '--export=ALL', '--chdir='+str(target/'code'),
        '--output='+str(target/'logs/overfit-%j.log'), str(target/'code/scripts/slurm/overfit_f187n.slurm')]
    state = {'status': 'prepared_only', 'command': command, 'run_root': str(target),
             'fit_selection': settings['overfit_diagnostic']['fit_selection']}
    if not args.dry_run:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        path = target/'submissions'/(stamp+'.json')
        env = os.environ.copy()
        env.update(AGN_ENV_PREFIX=settings['environment'], AGN_SNAPSHOT_ROOT=str(target))
        state['status'] = 'submitting'; save(path, state)
        try:
            state['diagnostic_job'] = job_id(subprocess.check_output(command, env=env, universal_newlines=True))
            state['status'] = 'submitted'; save(path, state)
        except Exception as exc:
            state.update(status='submission_failed', error=str(exc)); save(path, state); raise
    print(json.dumps(state, indent=2))
    print('Logs: '+str(target/'logs'))
    print('Review archive after successful completion: '+str(target/'review.tar.gz'))


if __name__ == '__main__':
    main()
