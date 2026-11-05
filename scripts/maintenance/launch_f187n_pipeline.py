#!/usr/bin/env python3
"""Freeze and submit the ARC F187N generate -> validate -> GPU train workflow.

Uses only the standard library and parses on system Python 3.6. Scientific
packages are loaded by the batch jobs inside the selected conda environment.
"""
import argparse
import datetime
import getpass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ('data/cloudy/ngc1068_f187n_pilot_v1/grid.npz',
          'data/cloudy/ngc1068_f187n_pilot_v1/manifest.json',
          'data/cloudy/hii_f187n_calculated_v2/grid.npz',
          'data/cloudy/hii_f187n_calculated_v2/manifest.json',
          'results/cloudy/f187n_reference/nucleus_cutout.npz',
          'results/cloudy/f187n_psf_nrcb4/kernel.npz')


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            value.update(chunk)
    return value.hexdigest()


def save(path, data):
    temporary = path.with_suffix('.pending.json')
    with temporary.open('w') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')
    temporary.replace(path)


def verify_snapshot(run_root):
    manifest = json.loads((run_root/'snapshot.json').read_text())
    for relative, expected in manifest['sha256'].items():
        path = run_root/relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError('Frozen workflow changed: '+relative)
    return manifest


def snapshot(run_root, env_prefix, groups, epochs, variant, smoke_only):
    # Fail before creating the run if prerequisites are missing.
    if not (env_prefix/'bin/python').is_file():
        raise ValueError('Environment Python missing: '+str(env_prefix/'bin/python'))
    for name in INPUTS:
        if not (ROOT/name).is_file():
            raise ValueError('Missing scientific input: '+name)
    run_root.mkdir(parents=True, exist_ok=False)
    code = run_root/'code'
    for folder in ('src', 'scripts', 'configs'):
        for path in sorted((ROOT/folder).rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.json', '.slurm') and '__pycache__' not in path.parts:
                destination = code/path.relative_to(ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(path), str(destination))
    # Copy the small scientific inputs; do not depend on mutable source symlinks.
    for name in INPUTS:
        destination = code/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(ROOT/name), str(destination))
    dataset_config = json.loads((code/'configs/cloudy/f187n_dataset_pilot_v3.json').read_text())
    dataset_config['groups'] = groups
    train_config = json.loads((code/'configs/cloudy/train_f187n_pilot_v2.json').read_text())
    train_config.update(epochs=epochs, variant=variant)
    save(code/'configs/cloudy/f187n_dataset_launch.json', dataset_config)
    save(code/'configs/cloudy/f187n_train_launch.json', train_config)
    (run_root/'logs').mkdir()
    (run_root/'submissions').mkdir()
    manifest = {
        'schema': 'f187n_arc_workflow_v3', 'environment': str(env_prefix),
        'groups': groups, 'epochs': epochs, 'variant': variant, 'smoke_only': smoke_only,
        'sha256': {str(p.relative_to(run_root)): digest(p) for p in sorted(code.rglob('*')) if p.is_file()},
        'scope': 'controlled shared-generator pilot with paired cone-removal negatives'}
    save(run_root/'snapshot.json', manifest)
    return manifest


def submission_commands(run_root, generation_id=None):
    code = run_root/'code'
    common = ['sbatch', '--parsable', '--export=ALL', '--chdir='+str(code)]
    generation = common + ['--output='+str(run_root/'logs/generate-%j.log'),
                           str(code/'scripts/slurm/generate_f187n.slurm')]
    training = common + ['--dependency=afterok:'+str(generation_id or 'GENERATION_JOB_ID'),
        '--kill-on-invalid-dep=yes', '--output='+str(run_root/'logs/train-%j.log'),
        str(code/'scripts/slurm/train_f187n.slurm')]
    return generation, training


def job_id(output):
    # --parsable may return JOBID;CLUSTER on this single-cluster ARC workflow.
    value = output.strip()
    parts = value.split(';')
    if len(parts) > 2 or not parts[0].isdigit():
        raise ValueError('Unexpected sbatch response: '+value)
    if len(parts) == 2:
        # This launcher targets one ARC cluster, not federated submissions.
        return parts[0]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--env-prefix', type=Path)
    parser.add_argument('--groups', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--variant', choices=('full', 'smooth_control', 'clumpy_cone'), default='full')
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Prepare and verify a frozen run without submitting jobs')
    parser.add_argument('--resume', action='store_true', help='Reuse the frozen run and verify completed generation groups')
    parser.add_argument('--verify-snapshot', type=Path, help='Verify an existing frozen run and exit')
    args = parser.parse_args()
    if args.verify_snapshot:
        verify_snapshot(args.verify_snapshot.resolve())
        print('Frozen source, configs, and scientific inputs verified.')
        return
    if args.run_root is None:
        parser.error('--run-root is required; choose a fresh experiment directory')
    if args.groups < 3 or args.epochs < 1:
        parser.error('Require groups >= 3 and epochs >= 1')
    run_root = args.run_root.resolve()
    if args.resume:
        settings = verify_snapshot(run_root)
        env_prefix = Path(settings['environment'])
        if args.env_prefix is not None and args.env_prefix.resolve() != env_prefix:
            raise ValueError('Resume uses the original environment prefix')
    else:
        if args.env_prefix is None:
            parser.error('--env-prefix is required for a fresh run')
        env_prefix = args.env_prefix.resolve()
        settings = snapshot(run_root, env_prefix, args.groups, args.epochs, args.variant, args.smoke_only)
        verify_snapshot(run_root)
    generation, training = submission_commands(run_root)
    if args.dry_run:
        print(json.dumps({'run_root': str(run_root), 'settings': {k:v for k,v in settings.items() if k != 'sha256'},
                          'generation_command': generation, 'training_command': training}, indent=2))
        print('Prepared only. Submit this frozen workflow with --run-root PATH --resume.')
        return
    if shutil.which('sbatch') is None:
        raise RuntimeError('sbatch not found; use --dry-run locally or submit from ARC')
    lock = run_root/'.submission.lock'
    lock.mkdir()  # A second launcher must not submit the same run concurrently.
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    state_path = run_root/'submissions'/(stamp+'.json')
    state = {'status': 'preparing', 'commands': [generation, training]}
    try:
        # Refuse another submission while any previous workflow job is active.
        previous = [json.loads(p.read_text()) for p in (run_root/'submissions').glob('*.json')]
        known = {s[k] for s in previous for k in ('generation_job', 'training_job') if k in s}
        if known:
            active = subprocess.check_output(['squeue', '--noheader', '--user', getpass.getuser(), '--format=%A'],
                                             universal_newlines=True).split()
            if known.intersection(active):
                raise RuntimeError('This workflow already has active jobs; inspect squeue before resubmitting')
        if any((run_root/'training').glob('*')):
            raise RuntimeError('A training attempt already exists; preserve it and use a fresh run root')
        env = os.environ.copy()
        env.update(AGN_ENV_PREFIX=str(env_prefix), AGN_DATASET=str(run_root/'dataset'),
            AGN_RUN=str(run_root/'training/model'), AGN_VARIANT=settings['variant'],
            AGN_SMOKE_ONLY='1' if settings['smoke_only'] else '0',
            AGN_DATASET_CONFIG=str(run_root/'code/configs/cloudy/f187n_dataset_launch.json'),
            AGN_TRAIN_CONFIG=str(run_root/'code/configs/cloudy/f187n_train_launch.json'),
            AGN_SNAPSHOT_ROOT=str(run_root), AGN_RESUME_GENERATION='1' if args.resume else '0')
        save(state_path, state)
        state['generation_job'] = job_id(subprocess.check_output(generation, env=env, universal_newlines=True))
        state['status'] = 'generation_submitted'
        save(state_path, state)
        _, training = submission_commands(run_root, state['generation_job'])
        state['commands'][1] = training
        state['training_job'] = job_id(subprocess.check_output(training, env=env, universal_newlines=True))
        state['status'] = 'submitted'
        save(state_path, state)
        print(json.dumps(state, indent=2))
        print('Logs: '+str(run_root/'logs'))
    except Exception as exc:
        state.update(status='submission_failed', error=str(exc))
        save(state_path, state)
        raise
    finally:
        lock.rmdir()


if __name__ == '__main__':
    main()
