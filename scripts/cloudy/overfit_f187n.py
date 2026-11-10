#!/usr/bin/env python3
"""Fit a few TRAINING examples to diagnose F187N mask/boundary optimization.

Every run is freshly initialized. This is a memorization diagnostic, never a
held-out performance estimate; validation and test image arrays are not loaded.
"""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/agn-f187n-overfit-mpl')
import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt
from torch.utils.data import DataLoader
from src.machine_learning.models.model_2d import UNet
from src.machine_learning.losses.f187n import F187NLoss
from src.machine_learning.optimization import scaled_optimizer_step
from src.simple_utils.f187n_pilot_dataset import validate_manifest, sha256, write_json
from scripts.cloudy.train_f187n_pilot import PilotDataset
from scripts.maintenance.launch_f187n_pipeline import verify_snapshot
from scripts.maintenance.launch_f187n_overfit import select_fit_samples


def boundary_f1(predicted, target, tolerance_pixels):
    """Symmetric boundary matching within a Euclidean pixel tolerance, full stamp.

    Inner boundaries use 8-neighbor erosion including the stamp edge. This
    diagnostic is called only for positive targets; an empty prediction scores 0.
    """
    p = predicted & ~binary_erosion(predicted, structure=np.ones((3, 3)), border_value=0)
    t = target & ~binary_erosion(target, structure=np.ones((3, 3)), border_value=0)
    if not p.any() or not t.any(): return 0.
    precision = float((distance_transform_edt(~t)[p] <= tolerance_pixels).mean())
    recall = float((distance_transform_edt(~p)[t] <= tolerance_pixels).mean())
    return 2*precision*recall/(precision+recall) if precision+recall else 0.


def evaluate(model, loader, device, loss_fn, threshold, tolerance):
    model.eval()
    loss_sum = count = 0
    samples, previews, probabilities = [], [], {}
    with torch.inference_mode():
        for x, y, indices in loader:
            logits = model(x.to(device))  # FP32 eval, as in the original held-out report.
            loss = loss_fn(logits, y.to(device))
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise RuntimeError('Nonfinite diagnostic evaluation')
            probability = logits.float().sigmoid().cpu().numpy()[:, 0]
            loss_sum += float(loss)*len(x); count += len(x)
            for k, index in enumerate(indices.tolist()):
                row = loader.dataset.rows[index]
                target = y[k, 0].numpy().astype(bool)
                pred = probability[k] > threshold
                positive = bool(target.any())
                dice = 2*int((pred & target).sum())/(int(pred.sum())+int(target.sum())) if positive else None
                samples.append({'id': row['id'], 'group': row['group'], 'variant': row['variant'],
                    'cone_present': positive, 'full_stamp_dice': dice,
                    'full_stamp_boundary_f1': boundary_f1(pred, target, tolerance) if positive else None,
                    'full_stamp_predicted_area_fraction': float(pred.mean())})
                previews.append((row['id'], x[k, 0].numpy(), target, pred))
                probabilities[row['id']] = probability[k]
    positive = [s for s in samples if s['cone_present']]
    negative = [s for s in samples if not s['cone_present']]
    metrics = {'fit_loss': loss_sum/count,
        'mean_positive_dice': float(np.mean([s['full_stamp_dice'] for s in positive])),
        'minimum_positive_dice': min(s['full_stamp_dice'] for s in positive),
        'mean_positive_boundary_f1': float(np.mean([s['full_stamp_boundary_f1'] for s in positive])),
        'maximum_negative_area': max(s['full_stamp_predicted_area_fraction'] for s in negative)}
    return metrics, samples, previews, probabilities


def recovery_checks(metrics, criteria):
    return (metrics['mean_positive_dice'] >= criteria['mean_positive_dice_at_least'] and
            metrics['minimum_positive_dice'] >= criteria['minimum_positive_dice_at_least'] and
            metrics['mean_positive_boundary_f1'] >= criteria['mean_positive_boundary_f1_at_least'] and
            metrics['maximum_negative_area'] <= criteria['maximum_negative_area_at_most'])


def plot_result(output, history, previews):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    for key, label in [('mean_positive_dice', 'Positive Dice'), ('mean_positive_boundary_f1', 'Boundary F1')]:
        axes[0].plot([s['epoch'] for s in history], [s[key] for s in history], label=label)
    axes[0].set(xlabel='Epoch', ylabel='Fit-set score', ylim=(0, 1.01)); axes[0].legend()
    axes[1].plot([s['epoch'] for s in history], [100*s['maximum_negative_area'] for s in history])
    axes[1].set(xlabel='Epoch', ylabel='Maximum negative predicted area (%)')
    fig.suptitle('Training examples only: memorization, not held-out accuracy')
    fig.savefig(output/'fit_curves.png', dpi=130); plt.close(fig)
    fig, axes = plt.subplots(len(previews), 4, figsize=(10, len(previews)*2.5), squeeze=False,
                             constrained_layout=True)
    for axs, (name, image, target, pred) in zip(axes, previews):
        rgb = np.zeros(target.shape+(3,))
        rgb[target & pred] = (.2, .8, .3)
        rgb[target & ~pred] = (.1, .7, 1.)
        rgb[~target & pred] = (.95, .2, .65)
        for ax, layer, title in zip(axs, (image, target, pred, rgb),
                                    ('Input (asinh)', 'Target', 'Prediction', 'Green correct / blue missed / pink extra')):
            display = ({'cmap': 'magma'} if title.startswith('Input') else
                       {'cmap': 'gray', 'vmin': 0, 'vmax': 1} if layer.ndim == 2 else {})
            ax.imshow(layer, origin='lower', interpolation='nearest', **display)
            ax.set_title(title, fontsize=8); ax.set_axis_off()
        axs[0].set_title(name+'\nInput (asinh)', fontsize=8)
    fig.savefig(output/'fit_predictions.png', dpi=115); plt.close(fig)


def fit_variant(dataset, rows, output, cfg, diagnostic, device):
    output.mkdir(parents=True, exist_ok=False)
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.set_num_threads(cfg['cpu_threads'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    ds = PilotDataset(dataset, rows)
    train = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=0,
                       generator=torch.Generator().manual_seed(cfg['seed']))
    evaluation = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=0)
    model = UNet(in_channels=1, out_channels=1, features=cfg['features']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    criterion = F187NLoss(cfg['bce_weight'], cfg['dice_weight'], cfg['pos_weight']).to(device)
    amp = bool(cfg['mixed_precision'] and device == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    write_json(output/'config.json', dict(cfg, diagnostic=diagnostic, device=device,
        mixed_precision_used=amp, fit_sample_ids=[r['id'] for r in rows],
        parameters=sum(p.numel() for p in model.parameters()),
        packages={p: version(p) for p in ('torch', 'numpy', 'scipy', 'matplotlib')},
        scope='Same training images fitted and scored. No held-out performance claim.'))
    best = float('inf'); history = []; total_updates = skips = consecutive = 0
    for epoch in range(1, diagnostic['max_epochs']+1):
        model.train(); epoch_updates = 0
        for batch, (x, y, indices) in enumerate(train, start=1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
                logits = model(x.to(device))
            loss = criterion(logits, y.to(device))
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise RuntimeError('Nonfinite diagnostic logits/loss')
            scaler.scale(loss).backward()
            update = scaled_optimizer_step(model, optimizer, scaler, cfg['max_grad_norm'])
            if update['updated']:
                epoch_updates += 1; total_updates += 1; consecutive = 0
            else:
                skips += 1; consecutive += 1
                event = dict(update, epoch=epoch, batch=batch, samples=[rows[i]['id'] for i in indices.tolist()])
                with (output/'overflow_events.jsonl').open('a') as stream: stream.write(json.dumps(event)+'\n')
                print(json.dumps(dict(event, variant=rows[0]['variant'], event='fp16_overflow_skipped')), flush=True)
                if consecutive >= 8: raise RuntimeError('8 consecutive diagnostic overflows')
        if not epoch_updates: raise RuntimeError('Diagnostic epoch had no successful optimizer updates')
        if epoch != 1 and epoch % diagnostic['evaluate_every'] and epoch != diagnostic['max_epochs']:
            continue
        metrics, _, _, _ = evaluate(model, evaluation, device, criterion, cfg['prediction_threshold'],
                                    diagnostic['boundary_tolerance_pixels'])
        reached = recovery_checks(metrics, diagnostic['recovery_checks'])
        record = dict(metrics, epoch=epoch, recovery_checks_met=reached,
                      optimizer_updates=total_updates, overflow_skips=skips, loss_scale=float(scaler.get_scale()))
        history.append(record)
        with (output/'history.jsonl').open('a') as stream: stream.write(json.dumps(record)+'\n')
        print(json.dumps(dict(record, variant=rows[0]['variant'])), flush=True)
        stop = reached and epoch >= diagnostic['minimum_epochs']
        if metrics['fit_loss'] < best or stop:
            best = metrics['fit_loss']
            saved = {'model': model.state_dict(), 'epoch': epoch, 'config': cfg}
            temporary = output/'selected.pending.pth'
            torch.save(saved, temporary); temporary.replace(output/'selected.pth')
        if stop: break
    selected = torch.load(output/'selected.pth', map_location=device, weights_only=True)
    model.load_state_dict(selected['model'])
    metrics, samples, previews, probabilities = evaluate(model, evaluation, device, criterion,
        cfg['prediction_threshold'], diagnostic['boundary_tolerance_pixels'])
    result = {'variant': rows[0]['variant'], 'selected_epoch': selected['epoch'], 'epochs_completed': epoch,
        'recovery_checks_met': recovery_checks(metrics, diagnostic['recovery_checks']),
        'status': 'recovery_checks_met' if recovery_checks(metrics, diagnostic['recovery_checks']) else 'budget_exhausted',
        'selection': 'First checked epoch meeting recovery checks after minimum epochs, otherwise minimum fit-set loss',
        'optimizer_updates': total_updates, 'overflow_skips': skips,
        'metrics': metrics, 'samples': samples, 'scope': diagnostic['scope']}
    write_json(output/'report.json', result)
    np.savez_compressed(output/'fit_probabilities.npz', **probabilities)
    plot_result(output, history, previews)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-run', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    root = args.snapshot_run.resolve()
    settings = verify_snapshot(root)
    info = settings['overfit_diagnostic']
    dataset = Path(info['dataset'])
    if sha256(dataset/'manifest.json') != info['dataset_manifest_sha256'] or sha256(dataset/'plan.json') != info['dataset_plan_sha256']:
        raise ValueError('Diagnostic source dataset changed')
    # Metadata integrity only for unselected cases: never load holdout arrays.
    manifest = validate_manifest(dataset, verify_hashes=False)
    cfg = json.loads((root/'code/configs/cloudy/f187n_train_launch.json').read_text())
    diagnostic = json.loads((root/'code/configs/cloudy/f187n_overfit_launch.json').read_text())
    if (diagnostic['max_epochs'] < 1 or diagnostic['evaluate_every'] < 1 or
            diagnostic['minimum_epochs'] < 1 or diagnostic['boundary_tolerance_pixels'] < 0 or
            cfg['loss'] != 'f187n_bce_positive_dice_v1'):
        raise ValueError('Invalid fit-set diagnostic configuration')
    actual = select_fit_samples(manifest, diagnostic['fit_groups'], cfg['seed'], diagnostic['variants'])
    if actual != info['fit_selection']: raise ValueError('Frozen fit selection changed')
    lookup = {s['id']: s for s in manifest['samples']}
    for ids in actual['sample_ids_by_variant'].values():
        for identifier in ids:
            row = lookup[identifier]
            if row['split'] != 'train': raise ValueError('Holdout in fit diagnostic')
            for field in ('image', 'mask'):
                if sha256(dataset/row[field]) != row[field+'_sha256']:
                    raise ValueError('Changed diagnostic input: '+identifier)
    if args.device == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('Allocated CUDA GPU required')
    output = root/'diagnostics'
    output.mkdir(exist_ok=False)
    write_json(output/'provenance.json', dict(info, diagnostic=diagnostic, full_stamp_metrics=True,
        boundary_tolerance_arcsec=diagnostic['boundary_tolerance_pixels']*manifest['pixel_scale_arcsec']))
    results = {}
    for variant in diagnostic['variants']:
        rows = [lookup[i] for i in actual['sample_ids_by_variant'][variant]]
        results[variant] = fit_variant(dataset, rows, output/variant, dict(cfg, variant=variant), diagnostic, args.device)
    write_json(output/'summary.json', {'schema': diagnostic['schema'], 'fit_selection': actual,
        'dataset': str(dataset), 'dataset_manifest_sha256': info['dataset_manifest_sha256'],
        'variants': results, 'scope': diagnostic['scope']})
    print('Completed fit-set diagnostic; held-out accuracy was not evaluated.', flush=True)


if __name__ == '__main__':
    main()
