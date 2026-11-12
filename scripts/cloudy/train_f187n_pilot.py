#!/usr/bin/env python3
"""Train on an explicit grouped F187N manifest, including paired empty targets."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/agn-f187n-training-mpl')
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from src.machine_learning.models.model_2d import UNet
from src.machine_learning.losses.f187n import F187NLoss
from src.machine_learning.optimization import scaled_optimizer_step
from src.machine_learning.validation_stopping import ValidationStopping, balanced_mask_score
from src.simple_utils.f187n_pilot_dataset import (
    VARIANTS, validate_manifest, sha256, write_json, fit_mask_angles, select_samples)
from scripts.cloudy.compare_real_synthetic_cones import (
    sky_offsets, smooth_image, excess_profile, measure_lobes, measured_mask, angle_error)


class PilotDataset(Dataset):
    def __init__(self, root, rows):
        self.root, self.rows = Path(root), rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        x = np.load(self.root/row['image'], allow_pickle=False)
        y = np.load(self.root/row['mask'], allow_pickle=False)
        if (x.shape != (256, 256) or y.shape != x.shape or
                not np.isfinite(x).all() or not np.isin(y, [0, 1]).all()):
            raise ValueError(f'Invalid F187N tensor or mask: {row["id"]}')
        return (torch.from_numpy(x.astype(np.float32)[None]),
                torch.from_numpy(y.astype(np.float32)[None]), index)


def binary_scores(predicted, target, aperture):
    a, b = predicted & aperture, target & aperture
    intersection, union = int((a & b).sum()), int((a | b).sum())
    return {'iou': intersection/union if union else 1.,
            'dice': 2*intersection/(int(a.sum())+int(b.sum())) if union else 1.,
            'predicted_area_fraction': float(a.sum()/aperture.sum())}


def test_report(model, loader, dataset, manifest, device, cfg):
    valid = np.load(dataset/'valid.npy', allow_pickle=False).astype(bool)
    east, north = sky_offsets(valid.shape, [127.5, 127.5], np.array(manifest['matrix_arcsec']))
    radius = np.hypot(east, north)
    aperture = valid & (radius > .7) & (radius < 3.3)
    if not aperture.any():
        raise ValueError('Empty evaluation aperture')
    model.eval()
    results, previews = [], []
    with torch.no_grad():
        for x, y, indices in loader:
            logits = model(x.to(device))
            if not torch.isfinite(logits).all():
                raise RuntimeError('Nonfinite test logits')
            predicted = (logits.sigmoid() > cfg['prediction_threshold']).cpu().numpy()[:, 0]
            for k, index in enumerate(indices.tolist()):
                row = loader.dataset.rows[index]
                target = y[k, 0].numpy().astype(bool)
                present = bool(target.any())
                fitted = fit_mask_angles(predicted[k], east, north, valid)
                expected = row['geometry'] if present else None
                linear = np.sinh(x[k, 0].numpy().astype(float))*row['softening']
                profile, _, _ = excess_profile(
                    smooth_image(linear, valid, .1/manifest['pixel_scale_arcsec']), valid, east, north)
                baseline = measure_lobes(profile, .2)
                baseline_mask = measured_mask(baseline, east, north, .7, 3.3) & valid

                def errors(pa, width):
                    return {
                        'axis_error_deg': angle_error(pa, expected['axis_pa_east_of_north_deg'])
                        if expected is not None and pa is not None else None,
                        'full_opening_error_deg': abs(width-expected['projected_full_opening_deg'])
                        if expected is not None and width is not None else None}

                primary = baseline['primary']
                results.append({
                    'id': row['id'], 'group': row['group'], 'variant': row['variant'], 'split':row['split'],
                    'cone_present': present, 'complexity': row['complexity'],
                    'complexity_regions':row.get('complexity_regions'),
                    'dust_diagnostics':row.get('dust_diagnostics'), 'categories':row.get('categories'),
                    'extent_over_optical_psf_fwhm': row['extent_over_optical_psf_fwhm'],
                    'target_geometry': expected,
                    'model_full_stamp': binary_scores(predicted[k], target, np.ones_like(valid)),
                    'model_common_annulus': binary_scores(predicted[k], target, aperture),
                    'model_fitted_geometry': fitted,
                    'model_angle_errors': errors(fitted['axis_pa_mod180_deg'], fitted['projected_full_opening_deg'])
                    if fitted else errors(None, None),
                    'baseline_common_annulus': binary_scores(baseline_mask, target, aperture),
                    'baseline_angle_errors': errors(primary['pa_east_of_north_deg'], primary['opening_full_deg'])
                    if primary else errors(None, None)})
                if len(previews) < 6:
                    previews.append((row['id'], x[k, 0].numpy(), target, predicted[k]))
    positives = [r for r in results if r['cone_present']]
    negatives = [r for r in results if not r['cone_present']]

    def mean(rows, section, key):
        values = [r[section][key] for r in rows if r[section][key] is not None]
        return float(np.mean(values)) if values else None

    report = {
        'samples': results, 'count': len(results), 'positive_count': len(positives),
        'negative_count': len(negatives), 'prediction_threshold': cfg['prediction_threshold'],
        'positive_mean_model_annulus_iou': mean(positives, 'model_common_annulus', 'iou'),
        'positive_mean_baseline_annulus_iou': mean(positives, 'baseline_common_annulus', 'iou'),
        'model_angle_fit_count': sum(r['model_fitted_geometry'] is not None for r in positives),
        'positive_mean_model_axis_error_deg': mean(positives, 'model_angle_errors', 'axis_error_deg'),
        'positive_mean_model_full_opening_error_deg': mean(positives, 'model_angle_errors', 'full_opening_error_deg'),
        'positive_mean_baseline_axis_error_deg': mean(positives, 'baseline_angle_errors', 'axis_error_deg'),
        'positive_mean_baseline_full_opening_error_deg': mean(positives, 'baseline_angle_errors', 'full_opening_error_deg'),
        'baseline_angle_fit_count': sum(r['baseline_angle_errors']['axis_error_deg'] is not None for r in positives),
        'negative_mean_model_false_positive_area_fraction': mean(negatives, 'model_common_annulus', 'predicted_area_fraction'),
        'negative_mean_baseline_false_positive_area_fraction': mean(negatives, 'baseline_common_annulus', 'predicted_area_fraction'),
        'negative_area_flag_fraction': cfg['negative_area_flag_fraction'],
        'negative_model_flagged_count': sum(r['model_common_annulus']['predicted_area_fraction'] >= cfg['negative_area_flag_fraction'] for r in negatives),
        'negative_baseline_flagged_count': sum(r['baseline_common_annulus']['predicted_area_fraction'] >= cfg['negative_area_flag_fraction'] for r in negatives),
        'method': 'Angle errors apply only to positive targets, with undefined failed fits and explicit counts. Model geometry uses fixed 0.3--1.5 arcsec annulus, 3-deg grid, no truth geometry. Baseline uses 0.7--3.3 arcsec angular excess; overlap and negative false-positive area use that common valid annulus. Empty/empty masks score overlap 1, but negative cases are excluded from positive overlap means. The 1% area flag is an experimental reporting rule, not a calibrated detection threshold.',
        'scope': manifest['validation_scope']}
    return report, previews


def save_previews(path, previews):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(previews), 3, figsize=(9, 3*len(previews)), squeeze=False)
    for row_axes, (name, image, target, predicted) in zip(axes, previews):
        for ax, layer, title in zip(row_axes, (image, target, predicted), ('Input (asinh)', 'Target', 'Prediction')):
            if title.startswith('Input'):
                ax.imshow(layer, origin='lower', cmap='magma')
            else:
                ax.imshow(layer, origin='lower', cmap='gray', vmin=0, vmax=1)
            ax.set_title(name+' / '+title, fontsize=8)
            ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def smoke_check(model, optimizer, loss_fn, dataset, device, cfg, amp, scaler):
    # Deliberately exercise a positive and a negative, not a random first image.
    indices = [next(i for i, r in enumerate(dataset.rows) if r['variant'] != 'cone_free')]
    negatives = [i for i, r in enumerate(dataset.rows) if r['variant'] == 'cone_free']
    if negatives:
        indices.append(negatives[0])
    indices += [i for i in range(len(dataset)) if i not in indices][:max(0,cfg['batch_size']-len(indices))]
    items = [dataset[i] for i in indices]
    x = torch.stack([item[0] for item in items]).to(device)
    y = torch.stack([item[1] for item in items]).to(device)
    before = model.final_conv.weight.detach().clone()
    model.train()
    overflows=0
    for attempt in range(9):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device,dtype=torch.float16,enabled=amp):
            logits=model(x)
        loss=loss_fn(logits,y)
        if logits.shape!=y.shape or not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise RuntimeError('Invalid model output or loss')
        scaler.scale(loss).backward()
        update=scaled_optimizer_step(model,optimizer,scaler,cfg['max_grad_norm'])
        if update['updated']:break
        overflows+=1
    else:
        raise RuntimeError('Smoke check could not obtain finite scaled gradients after 9 attempts')
    if torch.equal(before, model.final_conv.weight.detach()):
        raise RuntimeError('Optimizer did not update')
    return {'input_shape': list(x.shape), 'output_shape': list(logits.shape),
            'loss': float(loss.detach()), 'positive_images': int((y.sum((1, 2, 3)) > 0).sum()),
            'negative_images': int((y.sum((1, 2, 3)) == 0).sum()), 'optimizer_updated': True,
            'mixed_precision_used':amp,'overflow_attempts':overflows,'loss_scale':float(scaler.get_scale()),
            'meaning': 'mixed-target computation check; no accuracy evidence'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/cloudy/train_f187n_pilot_v2.json')
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--expected-manifest-sha256')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    stopping = ValidationStopping(cfg)
    if cfg.get('loss') != 'f187n_bce_positive_dice_v1':
        raise ValueError('Use the explicit v2 loss/training config')
    variant = args.variant or cfg['variant']
    if variant not in VARIANTS:
        raise ValueError('Invalid positive variant')
    epochs = args.epochs if args.epochs is not None else cfg['epochs']
    if (epochs < 1 or cfg['batch_size'] < 1 or not 0 < cfg['prediction_threshold'] < 1 or
            not 0 < cfg['negative_area_flag_fraction'] <= 1 or cfg['max_grad_norm'] <= 0):
        raise ValueError('Invalid training settings')
    dataset = args.dataset.resolve()
    if args.expected_manifest_sha256 and sha256(dataset/'manifest.json')!=args.expected_manifest_sha256:
        raise ValueError('Dataset manifest changed since retry preparation')
    manifest = validate_manifest(dataset)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: use an allocated GPU job; no CPU fallback')
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.set_num_threads(cfg['cpu_threads'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    loaders = {}
    for split in ('train', 'val', 'test'):
        ds = PilotDataset(dataset, select_samples(manifest, split, variant))
        loaders[split] = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=split == 'train',
            num_workers=cfg['num_workers'], generator=torch.Generator().manual_seed(cfg['seed']))
    model = UNet(in_channels=1, out_channels=1, features=cfg['features']).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    loss_fn = F187NLoss(cfg['bce_weight'], cfg['dice_weight'], cfg['pos_weight']).to(args.device)
    amp = bool(cfg['mixed_precision'] and args.device == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    args.output.mkdir(parents=True, exist_ok=False)
    code_paths = [Path(__file__), ROOT/'src/simple_utils/f187n_pilot_dataset.py',
        ROOT/'src/machine_learning/models/model_2d.py', ROOT/'src/machine_learning/losses/f187n.py',
        ROOT/'src/machine_learning/optimization.py',
        ROOT/'src/machine_learning/validation_stopping.py',
        ROOT/'scripts/cloudy/compare_real_synthetic_cones.py']
    config = dict(cfg, variant=variant, epochs=epochs, device=args.device, dataset=str(dataset),
        smoke_test=args.smoke_test, dataset_manifest_sha256=sha256(dataset/'manifest.json'),
        python=sys.version, packages={name: version(name) for name in ('torch', 'numpy', 'scipy', 'astropy', 'matplotlib')},
        mixed_precision_used=amp,overflow_policy='skip overflowed FP16 optimizer update, reduce GradScaler scale, log event; fail on persistent overflow or unscaled nonfinite gradients',
        split_counts={s: {'images': len(loader.dataset),
            'cone_free': sum(r['variant'] == 'cone_free' for r in loader.dataset.rows)} for s, loader in loaders.items()},
        code_sha256={str(p.relative_to(ROOT)): sha256(p) for p in code_paths})
    write_json(args.output/'config.json', config)
    write_json(args.output/'dataset_manifest.json', manifest)
    if args.smoke_test:
        write_json(args.output/'smoke.json', smoke_check(model, optimizer, loss_fn, loaders['train'].dataset, args.device,cfg,amp,scaler))
        print('F187N positive/negative forward/backward/optimizer smoke check passed.')
        return
    history = []
    total_updates=total_overflows=consecutive_overflows=0
    for epoch in range(epochs):
        stats = {'epoch': epoch+1}
        epoch_updates=epoch_overflows=0
        for split in ('train', 'val'):
            model.train(split == 'train')
            total = count = positive_count = negative_count = 0
            positive_dice = negative_area = 0.
            for batch_index,(x, y, indices) in enumerate(loaders[split]):
                x, y = x.to(args.device), y.to(args.device)
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(split == 'train'):
                    with torch.autocast(device_type=args.device, dtype=torch.float16, enabled=amp):
                        logits = model(x)
                    loss = loss_fn(logits, y)
                    if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite logits/loss')
                    if split == 'train':
                        scaler.scale(loss).backward()
                        update=scaled_optimizer_step(model,optimizer,scaler,cfg['max_grad_norm'])
                        if update['updated']:
                            epoch_updates+=1;total_updates+=1;consecutive_overflows=0
                        else:
                            epoch_overflows+=1;total_overflows+=1;consecutive_overflows+=1
                            event=dict(update,epoch=epoch+1,batch=batch_index+1,
                                samples=[loaders[split].dataset.rows[i]['id'] for i in indices.tolist()])
                            with (args.output/'overflow_events.jsonl').open('a') as stream:
                                stream.write(json.dumps(event)+'\n')
                            print(json.dumps(dict(event,event='fp16_overflow_skipped')),flush=True)
                            if consecutive_overflows>=8:
                                raise RuntimeError('8 consecutive gradient overflows; stopping preserved run for diagnosis')
                pred = (logits.detach().sigmoid() > cfg['prediction_threshold']).float()
                axes = (1, 2, 3)
                positive = y.sum(axes) > 0
                dice = (2*(pred*y).sum(axes)+1e-6)/(pred.sum(axes)+y.sum(axes)+1e-6)
                positive_dice += float(dice[positive].sum())
                positive_count += int(positive.sum())
                negative_area += float(pred.mean(axes)[~positive].sum())
                negative_count += int((~positive).sum())
                total += float(loss.detach())*len(x)
                count += len(x)
            stats[split+'_loss'] = total/count
            stats[split+'_positive_dice'] = positive_dice/positive_count if positive_count else None
            stats[split+'_negative_fp_area_fraction'] = negative_area/negative_count if negative_count else None
        if not epoch_updates:
            raise RuntimeError('Epoch completed without any optimizer updates')
        stats.update(optimizer_updates=epoch_updates,overflow_skipped_updates=epoch_overflows,
            cumulative_optimizer_updates=total_updates,cumulative_overflow_skips=total_overflows,
            loss_scale=float(scaler.get_scale()))
        stats['val_balanced_mask_score'] = (balanced_mask_score(
            stats['val_positive_dice'], stats['val_negative_fp_area_fraction'])
            if stats['val_positive_dice'] is not None and stats['val_negative_fp_area_fraction'] is not None else None)
        stopping.update(stats, epoch+1)
        stats['selection_value'] = stats[stopping.metric]
        stats['stale_validation_epochs'] = stopping.stale_epochs
        history.append(stats)
        checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                      'scaler': scaler.state_dict(), 'epoch': epoch+1, 'config': config, 'history': history}
        path = args.output/f'epoch_{epoch+1:03d}.pth'
        temporary = path.with_suffix('.pending.pth')
        torch.save(checkpoint, temporary)
        temporary.rename(path)
        write_json(args.output/f'epoch_{epoch+1:03d}.json', stats)
        print(json.dumps(stats), flush=True)
        if stopping.should_stop:
            print(json.dumps(dict(event='validation_early_stopping', **stopping.state())), flush=True)
            break
    best_epoch = stopping.best_epoch
    saved = torch.load(args.output/f'epoch_{best_epoch:03d}.pth', map_location=args.device, weights_only=True)
    model.load_state_dict(saved['model'])
    write_json(args.output/'selection.json', {'best_epoch': best_epoch,
        'criterion': stopping.mode+' '+stopping.metric+'; test evaluated once after selection',
        'validation_stopping': dict(stopping.state(), stopped_early=len(history)<epochs, epoch_cap=epochs),
        'score_definition': 'val_balanced_mask_score = 0.5 * (positive full-stamp Dice + 1 - negative full-stamp predicted area); selection objective, not detection accuracy'})
    report, previews = test_report(model, loaders['test'], dataset, manifest, args.device, cfg)
    write_json(args.output/'test_report.json', report)
    save_previews(args.output/'test_predictions.png', previews)
    print(f'Completed diagnostic training; held-out report: {args.output/"test_report.json"}')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        # The caller captures stdout/stderr in a persistent job log.
        traceback.print_exc()
        sys.exit(1)
