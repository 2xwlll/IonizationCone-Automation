#!/usr/bin/env python3
"""Submit a fresh training-only attempt against an existing frozen F187N dataset.

Standard library only; compatible with the ARC login node's Python 3.6.
The original run, checkpoints, component archives and logs are read-only.
"""
import argparse
import datetime
import getpass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.maintenance.launch_f187n_pipeline import snapshot,verify_snapshot,digest,save,job_id


def check_inactive(source):
    identifiers=set()
    for path in (source/'submissions').glob('*.json'):
        record=json.loads(path.read_text())
        identifiers.update(str(record[k]) for k in ('generation_job','training_job') if k in record)
    if identifiers:
        active=subprocess.check_output(['squeue','--noheader','--user',getpass.getuser(),'--format=%A'],
                                       universal_newlines=True).split()
        if identifiers.intersection(active):
            raise RuntimeError('The source run still has active jobs; wait for them before retrying')


def prepare(source,target):
    original=verify_snapshot(source)
    dataset=source/'dataset'
    manifest_path=dataset/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('band')!='F187N' or manifest.get('projection')!='circular_cone_envelope_v2':
        raise ValueError('Retry supports the explicit corrected F187N image pipeline')
    dataset_digest=digest(manifest_path)
    previous_config=source/'training/model/config.json'
    if previous_config.is_file():
        previous=json.loads(previous_config.read_text())
        if previous['dataset_manifest_sha256']!=dataset_digest:
            raise ValueError('The dataset no longer matches the failed training attempt')
    config=json.loads((source/'code/configs/cloudy/f187n_train_launch.json').read_text())
    settings=snapshot(target,Path(original['environment']),original['groups'],
        config['epochs'],config['variant'],False,scientific_root=source/'code',training_settings=config)
    settings.update(retry_of=str(source),source_snapshot_sha256=digest(source/'snapshot.json'),
        dataset_path=str(dataset),dataset_manifest_sha256=dataset_digest,
        dataset_plan_sha256=digest(dataset/'plan.json'),
        retry_policy='fresh initialization with original settings and seed; reuse dataset, no checkpoint resume or generation')
    save(target/'snapshot.json',settings)
    verify_snapshot(target)
    return settings


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--submit-prepared',action='store_true')
    args=parser.parse_args()
    source=args.source_run.resolve();target=args.run_root.resolve()
    if source==target:raise ValueError('Use a fresh run directory; the failed attempt must be preserved')
    if not args.dry_run and shutil.which('sbatch') is None:
        raise RuntimeError('sbatch unavailable; use --dry-run locally')
    if not args.dry_run:check_inactive(source)
    if args.submit_prepared:
        settings=verify_snapshot(target)
        if settings.get('retry_of')!=str(source):raise ValueError('Prepared retry belongs to another source run')
        if list((target/'submissions').glob('*.json')):
            raise RuntimeError('This retry has already had a submission attempt; preserve it and prepare a fresh run')
    else:settings=prepare(source,target)
    if digest(source/'snapshot.json')!=settings['source_snapshot_sha256']:
        raise ValueError('Original frozen snapshot metadata changed after retry preparation')
    dataset=Path(settings['dataset_path'])
    if digest(dataset/'manifest.json')!=settings['dataset_manifest_sha256']:
        raise ValueError('Original dataset manifest changed after retry preparation')
    if digest(dataset/'plan.json')!=settings['dataset_plan_sha256']:
        raise ValueError('Original generation plan changed after retry preparation')
    command=['sbatch','--parsable','--export=ALL','--chdir='+str(target/'code'),
        '--output='+str(target/'logs/train-%j.log'),str(target/'code/scripts/slurm/train_f187n.slurm')]
    if args.dry_run:
        print(json.dumps({'status':'prepared_only','dataset':str(dataset),
            'dataset_manifest_sha256':settings['dataset_manifest_sha256'],
            'command':command,'policy':settings['retry_policy']},indent=2))
        return
    lock=target/'.submission.lock';lock.mkdir()
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    state_path=target/'submissions'/(stamp+'.json')
    state={'status':'preparing','command':command,'source_run':str(source),'dataset':str(dataset)}
    try:
        env=os.environ.copy()
        env.update(AGN_ENV_PREFIX=settings['environment'],AGN_DATASET=str(dataset),
            AGN_RUN=str(target/'training/model'),AGN_VARIANT=settings['variant'],AGN_SMOKE_ONLY='0',
            AGN_TRAIN_CONFIG=str(target/'code/configs/cloudy/f187n_train_launch.json'),
            AGN_SNAPSHOT_ROOT=str(target),AGN_DATASET_MANIFEST_SHA256=settings['dataset_manifest_sha256'])
        save(state_path,state)
        state['training_job']=job_id(subprocess.check_output(command,env=env,universal_newlines=True))
        state['status']='submitted';save(state_path,state)
        print(json.dumps(state,indent=2))
        print('Logs: '+str(target/'logs'))
    except Exception as exc:
        state.update(status='submission_failed',error=str(exc));save(state_path,state);raise
    finally:lock.rmdir()


if __name__=='__main__':main()
