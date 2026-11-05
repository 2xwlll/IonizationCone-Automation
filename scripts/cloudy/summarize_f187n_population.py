#!/usr/bin/env python3
"""Summarize measured image descriptors; never assign a universal difficulty score."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR','/tmp/agn-f187n-summary-mpl')
import numpy as np
from src.simple_utils.f187n_pilot_dataset import validate_manifest,sha256,write_json
from src.simple_utils.image_descriptors import CATEGORY_DEFINITION


def tabulate(sample):
    regions=sample.get('complexity_regions') or {}
    whole=regions.get('whole_annulus') or sample['complexity']
    cone=regions.get('cone_annulus')
    dust=sample.get('dust_diagnostics') or {}
    dust_regions=dust.get('regions',{})
    dust_cone=dust_regions.get('cone_annulus') or {}
    dust_whole=dust_regions.get('whole_annulus') or {}
    def patch(metrics,component):
        return next((s[component] for s in (metrics or {}).get('structure',[]) if np.isclose(s['gaussian_sigma_arcsec'],.2)),None)
    categories=sample.get('categories') or {}
    return {'id':sample['id'],'group':sample['group'],'split':sample.get('split','test'),
        'variant':sample['variant'],'cone_present':sample.get('cone_present',True),
        'M_cone_annulus':cone['other_over_cone'] if cone else None,
        'M_whole_annulus':whole['other_over_cone'],
        'cone_fraction_inside_mask':cone['cone_fraction'] if cone else None,
        'other_dominant_area_inside_mask':cone['other_dominant_area_fraction'] if cone else None,
        'cone_structure_sigma_0p2_arcsec':patch(cone,'cone'),
        'other_structure_sigma_0p2_arcsec':patch(cone,'other'),
        'other_structure_whole_sigma_0p2_arcsec':patch(whole,'other'),
        'cone_median_signal_over_pixel_sigma':cone['median_cone_signal_over_pixel_sigma'] if cone else None,
        'extent_over_optical_psf_fwhm':sample['extent_over_optical_psf_fwhm'],
        'foreground_dust_applied':dust.get('applied'),
        'cone_area_mean_transmission':dust_cone.get('mean_transmission_pa_alpha'),
        'cone_dust_covered_area_fraction':dust_cone.get('covered_area_fraction'),
        'cone_area_mean_tau':dust_cone.get('mean_tau_pa_alpha'),
        'whole_area_mean_transmission':dust_whole.get('mean_transmission_pa_alpha'),
        'whole_dust_covered_area_fraction':dust_whole.get('covered_area_fraction'),
        'contrast_category':categories.get('contrast'),'obscuration_category':categories.get('obscuration')}


def population_figure(rows,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=[r for r in rows if r['variant']=='full']
    fig,axes=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
    ratios=[r['M_cone_annulus'] for r in selected if r['M_cone_annulus'] is not None]
    if ratios:
        axes[0,0].hist(ratios,bins=min(12,max(1,len(ratios))),color='steelblue')
    axes[0,0].set(xlabel='Other / cone light M inside mask',ylabel='Independent groups')
    for split,color in [('train','tab:blue'),('val','tab:orange'),('test','tab:green')]:
        subset=[r for r in selected if r['split']==split and r['M_cone_annulus'] is not None]
        axes[0,1].scatter([r['M_cone_annulus'] for r in subset],
            [r['other_structure_sigma_0p2_arcsec'] for r in subset],label=split,color=color)
        measured=[r for r in subset if r['cone_area_mean_transmission'] is not None]
        axes[1,0].scatter([r['M_cone_annulus'] for r in measured],
            [r['cone_area_mean_transmission'] for r in measured],label=split,color=color)
        axes[1,1].scatter([r['extent_over_optical_psf_fwhm'] for r in subset],
            [r['cone_median_signal_over_pixel_sigma'] for r in subset],label=split,color=color)
    axes[0,1].set(xlabel='M inside mask',ylabel='Other-light structure, sigma=0.2 arcsec')
    axes[0,1].legend()
    axes[1,0].set(xlabel='M inside mask',ylabel='Area-mean foreground transmission at Pa alpha',ylim=(0,1.05))
    axes[1,1].set(xlabel='Projected extent / optical PSF core FWHM',ylabel='Median cone signal / pixel sigma inside mask')
    fig.suptitle(f'Measured full-image descriptors: {len(selected)} groups\nNo combined difficulty score; dust transmission describes pre-PSF screen')
    fig.savefig(path,dpi=130);plt.close(fig)


def performance_figure(report,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    samples=[s for s in report['samples'] if s['cone_present']]
    negatives=[s for s in report['samples'] if not s['cone_present']]
    fig,axes=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
    specs=[('M_cone_annulus','model_common_annulus','iou','M inside mask','Test IoU'),
           ('cone_area_mean_transmission','model_common_annulus','iou','Foreground transmission','Test IoU'),
           ('M_cone_annulus','model_angle_errors','full_opening_error_deg','M inside mask','Full-opening error (deg)')]
    for ax,(xkey,section,ykey,xlabel,ylabel) in zip(axes[0],specs):
        points=[(tabulate(s)[xkey],s[section][ykey]) for s in samples]
        points=[(x,y) for x,y in points if x is not None and y is not None]
        if points:ax.scatter(*zip(*points))
        ax.set(xlabel=xlabel,ylabel=ylabel,title=f'{len(points)}/{len(samples)} defined fits/cases')
    for ax,key,label in zip(axes[1,:2],
            ('other_structure_whole_sigma_0p2_arcsec','whole_area_mean_transmission'),
            ('Negative other-light structure, sigma=0.2 arcsec','Negative whole-annulus transmission')):
        for section,name in [('model_common_annulus','U-Net'),('baseline_common_annulus','Baseline')]:
            points=[(tabulate(s)[key],s[section]['predicted_area_fraction']) for s in negatives]
            points=[(x,y) for x,y in points if x is not None and y is not None]
            if points:ax.scatter(*zip(*points),label=name)
        ax.set(xlabel=label,ylabel='Negative false-positive area fraction',ylim=(-.02,1.02))
        if ax.collections:ax.legend()
    for section,name in [('model_common_annulus','U-Net'),('baseline_common_annulus','Baseline')]:
        values=[s[section]['predicted_area_fraction'] for s in negatives]
        if values:axes[1,2].hist(values,bins=np.linspace(0,1,11),histtype='step',label=name)
    axes[1,2].set(xlabel='Negative false-positive area fraction',ylabel='Test negatives',title=f'{len(negatives)} negative cases')
    if negatives:axes[1,2].legend()
    fig.suptitle('Held-out positives (top) and negatives (bottom); failed angles remain undefined.')
    fig.savefig(path,dpi=130);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--training-report',type=Path)
    args=parser.parse_args()
    manifest=validate_manifest(args.dataset)
    plan=json.loads((args.dataset/'plan.json').read_text())
    configs={g['id']:g['configuration'] for g in plan['groups']}
    rows=[]
    for sample in manifest['samples']:
        row=tabulate(sample);cfg=configs[sample['group']]
        for name in plan['settings']['ranges']:
            keys=name.split('.') if '.' in name else [('geometry' if name in cfg['geometry'] else 'components'),name]
            value=cfg
            for key in keys:value=value[key]
            row['input_'+name]=value
        rows.append(row)
    report=json.loads(args.training_report.read_text()) if args.training_report else None
    if report:
        expected={r['id'] for r in manifest['samples'] if r['split']=='test'}
        if any(s['id'] not in expected for s in report['samples']):
            raise ValueError('Training report contains cases outside this test split')
        config=json.loads((args.training_report.parent/'config.json').read_text())
        if config['dataset_manifest_sha256']!=sha256(args.dataset/'manifest.json'):
            raise ValueError('Training report refers to another manifest')
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/'descriptors.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    counts={}
    for variant in sorted({r['variant'] for r in rows}):
        selected=[r for r in rows if r['variant']==variant]
        counts[variant]={field:{str(value):sum(r[field]==value for r in selected)
            for value in sorted({r[field] for r in selected},key=str)}
            for field in ('contrast_category','obscuration_category')}
    population_figure(rows,args.output/'population.png')
    if report:performance_figure(report,args.output/'performance_vs_descriptors.png')
    provenance={'schema':'f187n_measured_descriptors_v1','manifest_sha256':sha256(args.dataset/'manifest.json'),
        'rows':len(rows),'groups':len({r['group'] for r in rows}),'category_definition':CATEGORY_DEFINITION,
        'category_counts_by_variant':counts,
        'apertures':'reference-valid 0.3<r<3.3 arcsec whole_annulus; cone_annulus intersects nominal mask; baseline/noise excluded from component ratios',
        'control_ranges':plan['settings']['ranges'],
        'training_report_sha256':sha256(args.training_report) if report else None,
        'code_sha256':sha256(Path(__file__)),
        'limitations':'Measured descriptors and provisional bins, not a validated predictor of segmentation difficulty. Plot uses full positives once per group. No observational component decomposition.'}
    provenance['artifact_sha256']={p.name:sha256(p) for p in args.output.iterdir() if p.is_file()}
    write_json(args.output/'summary.json',provenance)
    print('Saved measured descriptors and plots:',args.output)


if __name__=='__main__':main()
