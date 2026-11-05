#!/usr/bin/env python3
"""Plan, render, then finalize a reproducible grouped F187N diagnostic dataset."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR','/tmp/agn-f187n-dataset-mpl')
import numpy as np
from src.simple_utils.cone_geometry import resolve_cone_geometry, CIRCULAR
from src.simple_utils.f187n_pilot_dataset import (SCHEMA, ALL_VARIANTS, sha256, write_json,
    grouped_splits, normalize_image, validate_manifest, cone_free_image)
from src.simple_utils.f187n import rebin_surface_brightness, resample_psf
from src.simple_utils.gas_structure import soft_agn_emission_weight
from src.simple_utils.image_complexity import component_metrics, psf_axis_fwhm
from src.jwst_utils.nircam import convolve_kernel
from scripts.cloudy.measure_f187n_complexity import load_components


def code_hashes():
    paths = list((ROOT/'src/simple_utils').glob('*.py'))
    paths += [ROOT/'src/jwst_utils/nircam.py',ROOT/'scripts/cloudy/render_f187n_pilot.py',
              ROOT/'scripts/cloudy/generate_oiii_cube.py',ROOT/'scripts/cloudy/measure_f187n_complexity.py',Path(__file__)]
    return {str(p.relative_to(ROOT)):sha256(p) for p in sorted(set(paths))}


def plan_dataset(config_path, output, count=None):
    settings = json.loads(config_path.read_text())
    if settings.get('schema') != SCHEMA:
        raise ValueError('Use the versioned F187N v2 dataset config with paired negatives')
    base = json.loads((ROOT/settings['base_config']).read_text())
    if resolve_cone_geometry(base['geometry'])['projection'] != CIRCULAR or base['spatial_pixels'] != 256:
        raise ValueError('Only corrected 256-pixel F187N base configs are supported')
    count = settings['groups'] if count is None else count
    split = grouped_splits(count, settings['split_seed'])
    paths = [base[k] for k in ('grid','reference','psf')]+[base['star_forming']['grid']]
    paths += [str(Path(base['grid']).parent/'manifest.json'),
              str(Path(base['star_forming']['grid']).parent/'manifest.json')]
    inputs = {p:sha256(ROOT/p) for p in paths}  # Fail before creating plan if an input is missing.
    hii_manifest = json.loads((ROOT/Path(base['star_forming']['grid']).parent/'manifest.json').read_text())
    if not hii_manifest['status'].startswith('completed stellar-ionized'):
        raise ValueError('H II grid completion checks are missing')
    with np.load(ROOT/base['reference'],allow_pickle=False) as f:
        reference = json.loads(str(f['metadata']))
        valid = f['valid'].astype(bool)
    output.mkdir(parents=True,exist_ok=False)
    np.save(output/'valid.npy',valid)
    rng = np.random.default_rng(settings['parameter_seed'])
    groups = []
    for i in range(count):
        cfg = json.loads(json.dumps(base))
        cfg['seed'] = settings['first_realization_seed']+i
        cfg.pop('output',None)
        for key,bounds in settings['ranges'].items():
            value = float(rng.uniform(*bounds))
            target = cfg['geometry'] if key in cfg['geometry'] else cfg['components']
            target[key] = value
        resolved = resolve_cone_geometry(cfg['geometry'])
        groups.append({'id':f'g{i:04d}','split':split[i],'seed':cfg['seed'],
                       'configuration':cfg,'geometry':resolved})
    plan = {'schema':SCHEMA,'settings':settings,'groups':groups,'input_sha256':inputs,
            'code_sha256':code_hashes(),'pixel_scale_arcsec':reference['pixel_scale_arcsec'],
            'matrix_arcsec':(np.array(reference['pixel_to_sky_matrix'])*3600).tolist(),
            'valid_sha256':sha256(output/'valid.npy')}
    write_json(output/'plan.json',plan)
    print(f'Planned {count} groups / {4*count} images at {output}; no images rendered.')


def completed_record(output, group):
    """Verify a completed group before skipping it; never repair it in place."""
    path = output/'groups'/f'{group["id"]}.json'
    record = json.loads(path.read_text())
    if record['plan_sha256'] != sha256(output/'plan.json'):
        raise ValueError('Mixed plan versions')
    rows = record['samples']
    if len(rows) != len(ALL_VARIANTS) or {r['variant'] for r in rows} != set(ALL_VARIANTS):
        raise ValueError('Incomplete group record')
    for row in rows:
        if (row['group'], row['seed'], row['split']) != (group['id'], group['seed'], group['split']):
            raise ValueError('Group record disagrees with plan')
        present = row['variant'] != 'cone_free'
        if (row['id'] != group['id']+'_'+row['variant'] or row.get('cone_present') is not present or
                row['geometry'] != (group['geometry'] if present else None)):
            raise ValueError('Completed group geometry/presence disagrees with plan')
        for key in ('image', 'mask', 'source_archive'):
            relative = Path(row[key])
            if relative.is_absolute() or '..' in relative.parts or not (output/relative).resolve().is_relative_to(output.resolve()):
                raise ValueError('Group paths must stay within the dataset')
            digest = row['source_sha256'] if key == 'source_archive' else row[key+'_sha256']
            if sha256(output/relative) != digest:
                raise ValueError(f'Changed completed output: {relative}; preserve it and use a fresh dataset')
    return record


def smooth_control(cfg, matrix, scale, target_flux):
    """Analytic smooth illumination control; no spectra, dust or noise."""
    n,over = cfg['spatial_pixels'],cfg['oversample']
    high = n*over
    y,x = np.indices((high,high),dtype=float)
    x=(x-(high-1)/2)/(high/2);y=(y-(high-1)/2)/(high/2)
    unit = np.asarray(matrix)/scale
    east,north=unit[0,0]*x+unit[0,1]*y,unit[1,0]*x+unit[1,1]*y
    pa=np.deg2rad(cfg['geometry']['position_angle_east_of_north_deg'])
    axial,cross=east*np.sin(pa)+north*np.cos(pa),east*np.cos(pa)-north*np.sin(pa)
    geo=resolve_cone_geometry(cfg['geometry'])
    weight=soft_agn_emission_weight(axial,cross,np.hypot(x,y),geo['projected_half_opening_rad'],
        cfg['geometry']['radius_fraction'],np.ones_like(x),np.zeros_like(x),n/2*scale,cfg['gas_structure'])
    with np.load(ROOT/cfg['psf'],allow_pickle=False) as f:
        kernel=resample_psf(f['kernel'],float(f['pixel_scale_arcsec']),scale/over)
    image=np.maximum(rebin_surface_brightness(convolve_kernel(weight,kernel),over),0)
    return image*(target_flux/image.sum())


def render_group(output, index):
    plan=json.loads((output/'plan.json').read_text())
    if plan['schema'] != SCHEMA:
        raise ValueError('Legacy plan: create a fresh v2 dataset')
    if plan['code_sha256'] != code_hashes():
        raise ValueError('Generation code changed since planning; use a fresh versioned plan')
    for name,digest in plan['input_sha256'].items():
        if sha256(ROOT/name)!=digest:raise ValueError(f'Input changed: {name}')
    if sha256(output/'valid.npy') != plan['valid_sha256']:
        raise ValueError('Reference validity map changed since planning')
    if index<0 or index>=len(plan['groups']):raise ValueError('Group index out of range')
    group=plan['groups'][index];gid=group['id']
    records=output/'groups';records.mkdir(exist_ok=True)
    if (records/f'{gid}.json').exists():
        completed_record(output, group)
        print(f'Verified completed {gid}; skipping.', flush=True)
        return
    locks=output/'locks';locks.mkdir(exist_ok=True)
    lock=locks/gid;lock.mkdir()  # Prevent two concurrent jobs from writing the same group.
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    attempt=output/'attempts'/gid/stamp
    try:
        attempt.mkdir(parents=True)
        cfg=json.loads(json.dumps(group['configuration']));cfg['output']=str(attempt/'render')
        write_json(attempt/'config.json',cfg)
        with (attempt/'render.log').open('x') as log:
            subprocess.run([sys.executable,str(ROOT/'scripts/cloudy/render_f187n_pilot.py'),
                            '--config',str(attempt/'config.json')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        source=attempt/'render/image_mask.npz'
        meta,cone,other,mask,valid,sigma,total,checks=load_components(source)
        if meta['resolved_geometry']!=group['geometry']:raise ValueError('Rendered geometry mismatch')
        scale=plan['pixel_scale_arcsec'];y,x=np.indices(mask.shape)
        radius=np.hypot(x-127.5,y-127.5)*scale
        aperture=valid & (radius>.3) & (radius<3.3)
        with np.load(ROOT/cfg['psf'],allow_pickle=False) as f:
            widths=psf_axis_fwhm(f['kernel'],float(f['pixel_scale_arcsec']))
        controls={'smooth_control':smooth_control(cfg,plan['matrix_arcsec'],scale,float(cone.sum())),
                  'clumpy_cone':cone,'full':total,'cone_free':cone_free_image(total,cone)}
        rows=[]
        for variant,image in controls.items():
            present = variant != 'cone_free'
            target = mask if present else np.zeros_like(mask)
            tensor,softening=normalize_image(image)
            name=f'{gid}_{variant}'
            imagepath=attempt/f'{name}.npy';maskpath=attempt/f'{name}_mask.npy'
            np.save(imagepath,tensor);np.save(maskpath,target.astype(np.uint8))
            c=controls['smooth_control'] if variant=='smooth_control' else cone if present else np.zeros_like(cone)
            b=other if variant in ('full','cone_free') else np.zeros_like(c)
            metrics=component_metrics(c,b,aperture,pixel_scale_arcsec=scale,
                noise_sigma=sigma if variant in ('full','cone_free') else None)
            extent = float(radius[target].max()) if present else None
            rows.append({'id':name,'group':gid,'seed':group['seed'],'split':group['split'],
                'variant':variant,'image':str(imagepath.relative_to(output)),'mask':str(maskpath.relative_to(output)),
                'image_sha256':sha256(imagepath),'mask_sha256':sha256(maskpath),'softening':softening,
                'cone_present':present,'geometry':group['geometry'] if present else None,'complexity':metrics,
                'projected_extent_arcsec':extent,
                'extent_over_optical_psf_fwhm':extent/widths['mean_arcsec'] if present else None,
                'optical_psf_core_fwhm_arcsec':widths['mean_arcsec'],
                'baseline_in_input':float(meta['galaxy_plus_background_baseline_mjysr']) if variant in ('full','cone_free') else 0.,
                'component_membership':{'cone':checks['cone_components'] if present else [],
                    'other':checks['other_components'] if variant in ('full','cone_free') else []},
                'negative_definition':plan['settings']['negative_definition'] if not present else None,
                'component_ratio_aperture':'reference-valid pixels, 0.3 < r < 3.3 arcsec; baseline/noise excluded',
                'source_archive':str(source.relative_to(output)),'source_sha256':sha256(source),
                'component_checks':checks})
        # A killed writer leaves its attempt, never a half-written completion marker.
        completion=attempt/'completion.pending.json'
        write_json(completion,{'plan_sha256':sha256(output/'plan.json'),'samples':rows})
        completion.rename(records/f'{gid}.json')
        print(f'Completed {gid}; raw components, configs and log retained at {attempt}',flush=True)
    finally:
        lock.rmdir()


def finalize(output):
    plan=json.loads((output/'plan.json').read_text());rows=[]
    if plan['schema'] != SCHEMA:
        raise ValueError('Finalize requires a v2 plan')
    for group in plan['groups']:
        record=completed_record(output, group)
        rows.extend(record['samples'])
    manifest={'schema':SCHEMA,'band':'F187N','shape':[1,256,256],'projection':CIRCULAR,
        'normalization':'asinh(image/(0.03*p99(abs(image))))','samples':rows,
        'pixel_scale_arcsec':plan['pixel_scale_arcsec'],'matrix_arcsec':plan['matrix_arcsec'],
        'valid_file':'valid.npy','valid_sha256':plan['valid_sha256'],'plan_sha256':sha256(output/'plan.json'),
        'label':'nominal projected cone geometry, not detectable emission',
        'control_definition':'smooth_control is analytic illumination with matching PSF and total cone flux; no CLOUDY spectra, extinction, contaminants or noise. clumpy_cone is saved post-PSF cone gas with its extinction/excitation. full includes all components, empirical baseline and noise. cone_free removes all post-PSF cone light at fixed other components, empirical scale and noise; paired counterfactual, not an inactive-galaxy model.',
        'negative_definition':plan['settings']['negative_definition'],
        'validation_scope':'held-out realizations within one generator and shared spectral/reference inputs; not independent real-galaxy validation',
        'split_group_counts':{s:sum(g['split']==s for g in plan['groups']) for s in ('train','val','test')}}
    validate_manifest(output, manifest=manifest)
    if (output/'manifest.json').exists():
        if json.loads((output/'manifest.json').read_text()) != manifest:
            raise ValueError('Existing manifest differs; preserve it and use a fresh dataset')
    else:
        temporary = output/'manifest.pending.json'
        write_json(temporary,manifest)
        temporary.rename(output/'manifest.json')
    print(f'Finalized {len(rows)} images: {manifest["split_group_counts"]}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','render','finalize','validate'])
    parser.add_argument('--dataset',required=True,type=Path)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/cloudy/f187n_dataset_pilot_v2.json')
    parser.add_argument('--groups',type=int,help='Override group count for a small pipeline check')
    choose=parser.add_mutually_exclusive_group()
    choose.add_argument('--group-index',type=int);choose.add_argument('--all',action='store_true')
    args=parser.parse_args();output=args.dataset.resolve()
    if args.action=='plan':plan_dataset(args.config,output,args.groups)
    elif args.action=='render':
        if args.group_index is None and not args.all:parser.error('Specify --group-index or --all')
        count=len(json.loads((output/'plan.json').read_text())['groups'])
        for i in range(count) if args.all else [args.group_index]:render_group(output,i)
    elif args.action=='finalize':finalize(output)
    else:validate_manifest(output);print('Dataset manifest, grouping and input hashes verified.')


if __name__=='__main__':main()
