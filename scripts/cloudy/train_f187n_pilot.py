#!/usr/bin/env python3
"""Train explicitly on one variant of a validated, grouped F187N dataset."""
import argparse
import json
import os
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR','/tmp/agn-f187n-training-mpl')
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from src.machine_learning.models.model_2d import UNet
from src.machine_learning.losses.combined_BCE_Dice import BCEDiceLoss
from src.simple_utils.f187n_pilot_dataset import (VARIANTS, validate_manifest, sha256,
                                                write_json, fit_mask_angles)
from scripts.cloudy.compare_real_synthetic_cones import (sky_offsets, smooth_image, excess_profile,
    measure_lobes, measured_mask, mask_scores, angle_error)


class PilotDataset(Dataset):
    def __init__(self,root,rows):self.root,self.rows=Path(root),rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        x=np.load(self.root/row['image'],allow_pickle=False)
        y=np.load(self.root/row['mask'],allow_pickle=False)
        if x.shape!=(256,256) or y.shape!=x.shape or not np.isfinite(x).all() or not np.isin(y,[0,1]).all():
            raise ValueError(f'Invalid F187N tensor or mask: {row["id"]}')
        return torch.from_numpy(x.astype(np.float32)[None]),torch.from_numpy(y.astype(np.float32)[None]),index


def test_report(model,loader,dataset,manifest,device):
    valid=np.load(dataset/'valid.npy',allow_pickle=False).astype(bool)
    if sha256(dataset/'valid.npy')!=manifest['valid_sha256']:raise ValueError('Changed reference validity map')
    east,north=sky_offsets(valid.shape,[127.5,127.5],np.array(manifest['matrix_arcsec']))
    radius=np.hypot(east,north)
    aperture=valid & (radius>.7) & (radius<3.3)
    model.eval();results=[]
    with torch.no_grad():
        for x,y,indices in loader:
            predicted=(torch.sigmoid(model(x.to(device)))>.5).cpu().numpy()[:,0]
            for k,index in enumerate(indices.tolist()):
                row=loader.dataset.rows[index];target=y[k,0].numpy().astype(bool)
                fitted=fit_mask_angles(predicted[k],east,north,valid)
                expected=row['geometry'];linear=np.sinh(x[k,0].numpy().astype(float))*row['softening']
                profile,_,_=excess_profile(smooth_image(linear,valid,.1/manifest['pixel_scale_arcsec']),valid,east,north)
                baseline=measure_lobes(profile,.2)
                baseline_mask=measured_mask(baseline,east,north,.7,3.3)&valid
                def errors(pa,width):
                    return {'axis_error_deg':angle_error(pa,expected['axis_pa_east_of_north_deg']) if pa is not None else None,
                            'full_opening_error_deg':abs(width-expected['projected_full_opening_deg']) if width is not None else None}
                primary=baseline['primary']
                results.append({'id':row['id'],'group':row['group'],'variant':row['variant'],
                    'complexity':row['complexity'],'extent_over_optical_psf_fwhm':row['extent_over_optical_psf_fwhm'],
                    'target_geometry':expected,
                    'model_full_stamp':mask_scores(predicted[k],target,np.ones_like(valid)),
                    'model_common_annulus':mask_scores(predicted[k],target,aperture),
                    'model_fitted_geometry':fitted,
                    'model_angle_errors':errors(fitted['axis_pa_mod180_deg'],fitted['projected_full_opening_deg']) if fitted else errors(None,None),
                    'baseline_common_annulus':mask_scores(baseline_mask,target,aperture),
                    'baseline_angle_errors':errors(primary['pa_east_of_north_deg'],primary['opening_full_deg']) if primary else errors(None,None)})
    def mean(section,key):
        values=[r[section][key] for r in results if r[section][key] is not None]
        return float(np.mean(values)) if values else None
    return {'samples':results,'count':len(results),
        'mean_model_annulus_iou':mean('model_common_annulus','iou'),
        'mean_baseline_annulus_iou':mean('baseline_common_annulus','iou'),
        'model_angle_fit_count':sum(r['model_fitted_geometry'] is not None for r in results),
        'mean_model_axis_error_deg':mean('model_angle_errors','axis_error_deg'),
        'mean_model_full_opening_error_deg':mean('model_angle_errors','full_opening_error_deg'),
        'mean_baseline_axis_error_deg':mean('baseline_angle_errors','axis_error_deg'),
        'baseline_angle_fit_count':sum(r['baseline_angle_errors']['axis_error_deg'] is not None for r in results),
        'method':'Model mask fit uses fixed 0.3--1.5 arcsec annulus and 3-deg axis/full-width grid, no true geometry prior. Photometric baseline uses 0.7--3.3 arcsec angular excess. Both overlap scores use the latter common valid annulus. Missing fits stay undefined; report counts.',
        'scope':manifest['validation_scope']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/cloudy/train_f187n_pilot_v1.json')
    parser.add_argument('--variant',choices=VARIANTS)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--smoke-test',action='store_true')
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text());variant=args.variant or cfg['variant']
    dataset=args.dataset.resolve();manifest=validate_manifest(dataset)
    epochs=args.epochs if args.epochs is not None else cfg['epochs']
    if epochs<1:raise ValueError('Epoch count must be positive')
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable: run within an allocated GPU job; no silent CPU fallback')
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    torch.set_num_threads(cfg['cpu_threads'])
    loaders={}
    for split in ('train','val','test'):
        rows=[r for r in manifest['samples'] if r['split']==split and r['variant']==variant]
        ds=PilotDataset(dataset,rows)
        loaders[split]=DataLoader(ds,batch_size=cfg['batch_size'],shuffle=split=='train',
            num_workers=cfg['num_workers'],generator=torch.Generator().manual_seed(cfg['seed']))
    model=UNet(in_channels=1,out_channels=1,features=cfg['features']).to(args.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
    loss_fn=BCEDiceLoss(pos_weight=cfg['pos_weight']).to(args.device)
    amp=bool(cfg['mixed_precision'] and args.device=='cuda')
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    args.output.mkdir(parents=True,exist_ok=False)
    config=dict(cfg,variant=variant,epochs=epochs,device=args.device,dataset=str(dataset),
        smoke_test=args.smoke_test,dataset_manifest_sha256=sha256(dataset/'manifest.json'),
        python=sys.version,torch_version=str(torch.__version__),numpy_version=np.__version__,
        code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'src/simple_utils/f187n_pilot_dataset.py',ROOT/'src/machine_learning/models/model_2d.py']})
    write_json(args.output/'config.json',config)
    write_json(args.output/'dataset_manifest.json',manifest)
    if args.smoke_test:
        x,y,_=next(iter(loaders['train']));x,y=x[:1].to(args.device),y[:1].to(args.device)
        before=model.final_conv.weight.detach().clone()
        optimizer.zero_grad(set_to_none=True);logits=model(x);loss=loss_fn(logits,y);loss.backward()
        if logits.shape!=y.shape or not torch.isfinite(loss):raise RuntimeError('Invalid output/loss')
        grads=[p.grad for p in model.parameters() if p.requires_grad]
        if any(g is None or not torch.isfinite(g).all() for g in grads):raise RuntimeError('Invalid gradients')
        optimizer.step()
        if torch.equal(before,model.final_conv.weight.detach()):raise RuntimeError('Optimizer did not update')
        write_json(args.output/'smoke.json',{'input_shape':list(x.shape),'output_shape':list(logits.shape),
            'loss':float(loss.detach()),'optimizer_updated':True,'meaning':'one-batch pipeline check; no accuracy evidence'})
        print('F187N forward/backward/optimizer smoke check passed.');return
    history=[];best=float('inf')
    for epoch in range(epochs):
        stats={'epoch':epoch+1}
        for split in ('train','val'):
            model.train(split=='train');total=count=0;dice_sum=0.
            for x,y,_ in loaders[split]:
                x,y=x.to(args.device),y.to(args.device);optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(split=='train'):
                    with torch.autocast(device_type=args.device,dtype=torch.float16,enabled=amp):logits=model(x)
                    loss=loss_fn(logits.float(),y)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                    if split=='train':scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
                pred=(torch.sigmoid(logits.detach())>.5).float();axes=(1,2,3)
                dice=(2*(pred*y).sum(axes)+1e-6)/(pred.sum(axes)+y.sum(axes)+1e-6)
                total+=float(loss.detach())*len(x);count+=len(x);dice_sum+=float(dice.sum())
            stats[split+'_loss']=total/count;stats[split+'_dice']=dice_sum/count
        history.append(stats)
        checkpoint={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                    'epoch':epoch+1,'config':config,'history':history}
        # Each epoch is new; retain checkpoints/log history if a later run fails.
        torch.save(checkpoint,args.output/f'epoch_{epoch+1:03d}.pth')
        write_json(args.output/f'epoch_{epoch+1:03d}.json',stats)
        if stats['val_loss']<best:best=stats['val_loss'];best_epoch=epoch+1
        print(json.dumps(stats),flush=True)
    saved=torch.load(args.output/f'epoch_{best_epoch:03d}.pth',map_location=args.device,weights_only=True)
    model.load_state_dict(saved['model'])
    write_json(args.output/'selection.json',{'best_epoch':best_epoch,'criterion':'minimum validation loss; test evaluated once afterward'})
    report=test_report(model,loaders['test'],dataset,manifest,args.device)
    write_json(args.output/'test_report.json',report)
    print(f'Completed diagnostic training; held-out report: {args.output/"test_report.json"}')


if __name__=='__main__':main()
