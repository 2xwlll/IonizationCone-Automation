#!/usr/bin/env python3
"""Train on all Cloudy wavelength channels and predict one projected cone mask."""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.machine_learning.datasets.ionization_dataset import IonizationConeDataset2D
from src.machine_learning.losses.combined_BCE_Dice import BCEDiceLoss
from src.machine_learning.models.model_2d import UNet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/cloudy/train_oiii_cube_v1.json"))
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Use a small U-Net and check one real forward/backward batch")
    args = parser.parse_args()
    os.chdir(ROOT)
    cfg = json.loads(args.config.read_text())
    dataset = args.dataset or Path(cfg["dataset"])
    manifest = json.loads((dataset/"manifest.json").read_text())
    wave = np.load(dataset/manifest["wavelength_file"])
    expected_shape = [cfg["input_channels"], *cfg["spatial_shape"]]
    if manifest["shape"] != expected_shape or len(wave) != cfg["input_channels"]:
        raise ValueError(f"Expected {expected_shape} input; manifest says {manifest['shape']}")
    if manifest["axis_order"] != ["wavelength", "y", "x"]:
        raise ValueError("Training requires channel-first wavelength cubes")
    if not np.all(np.diff(wave) > 0):
        raise ValueError("Invalid wavelength order")
    for split in ("train", "val", "test"):
        entries = [r for r in manifest["samples"] if r["split"] == split]
        stems = {r["id"] for r in entries}
        for folder in ("images", "masks"):
            actual = {p.stem for p in (dataset/split/folder).glob("*.npy")}
            if actual != stems:
                raise ValueError(f"{split}/{folder} differs from the fixed manifest")
    seeds = [r["seed"] for r in manifest["samples"]]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Generation seeds must be disjoint across splits")

    seed = cfg["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cpu":
        torch.set_num_threads(cfg.get("cpu_threads", 4))
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    loaders = {}
    for split in ("train", "val"):
        ds = IonizationConeDataset2D(
            dataset/split/"images", dataset/split/"masks", normalize=False,
            expected_channels=cfg["input_channels"])
        loaders[split] = DataLoader(
            ds, batch_size=cfg["batch_size"], shuffle=(split == "train"),
            num_workers=cfg.get("num_workers", 0), pin_memory=(device == "cuda"),
            generator=torch.Generator().manual_seed(seed))
    features = [8, 16, 32] if args.smoke_test else cfg["features"]
    model = UNet(in_channels=cfg["input_channels"], out_channels=1,
                 features=features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    loss_fn = BCEDiceLoss(pos_weight=cfg.get("pos_weight", 1)).to(device)
    amp = bool(cfg.get("mixed_precision", False) and device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    inputs, masks = next(iter(loaders["train"]))
    print(f"Input: {tuple(inputs.shape)}; target: {tuple(masks.shape)}")
    print(f"Wavelengths: {wave[0]:.2f}–{wave[-1]:.2f} Å; device={device}; AMP={amp}")

    if args.smoke_test:
        inputs, masks = inputs[:1].to(device), masks[:1].to(device)
        optimizer.zero_grad(set_to_none=True)
        before = model.downs[0].double_conv[0].weight.detach().clone()
        predictions = model(inputs)
        if predictions.shape != masks.shape:
            raise ValueError(f"Prediction {predictions.shape} != target {masks.shape}")
        loss = loss_fn(predictions, masks)
        loss.backward()
        grad = model.downs[0].double_conv[0].weight.grad
        if not torch.isfinite(loss) or grad is None or not torch.isfinite(grad).all():
            raise RuntimeError("Nonfinite loss or first-layer gradient")
        # Each spectral channel must reach the first convolution's gradients.
        active = (grad.abs().sum(dim=(0, 2, 3)) > 0).sum().item()
        if active != cfg["input_channels"]:
            raise RuntimeError(f"Only {active} wavelength channels affect the first-layer gradients")
        optimizer.step()
        if torch.equal(before, model.downs[0].double_conv[0].weight.detach()):
            raise RuntimeError("Optimizer did not update the model")
        report = {
            "input_shape": list(inputs.shape), "target_shape": list(masks.shape),
            "prediction_shape": list(predictions.shape), "loss": loss.item(),
            "wavelength_channels_with_gradients": active, "device": device,
            "optimizer_updated": True,
            "interpretation": "pipeline check only; not evidence of physics learning or accuracy",
        }
        out = Path(cfg["results"])/"smoke_test.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return

    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    if epochs < 1:
        raise ValueError("Epoch count must be positive")
    run = Path(cfg["results"])/datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run.mkdir(parents=True)
    run_cfg = dict(cfg, dataset=str(dataset), epochs=epochs, device=device)
    (run/"config.json").write_text(json.dumps(run_cfg, indent=2))
    (run/"dataset_manifest.json").write_text(json.dumps(manifest, indent=2))
    np.save(run/"wavelength_angstrom.npy", wave)
    history, best = [], float("inf")
    for epoch in range(epochs):
        stats = {}
        for split, loader in loaders.items():
            model.train(split == "train")
            total_loss = total_dice = n = 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                if tuple(x.shape[1:]) != tuple(expected_shape):
                    raise ValueError(f"Unexpected input batch {tuple(x.shape)}")
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(split == "train"):
                    with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp):
                        logits = model(x)
                        loss = loss_fn(logits.float(), y)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Nonfinite {split} loss")
                    if split == "train":
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                pred = (torch.sigmoid(logits.detach()) > 0.5).float()
                intersection = (pred*y).sum(dim=(1, 2, 3))
                dice = (2*intersection+1e-6)/(pred.sum(dim=(1, 2, 3))+y.sum(dim=(1, 2, 3))+1e-6)
                n += x.shape[0]
                total_loss += loss.item()*x.shape[0]
                total_dice += dice.sum().item()
            stats[f"{split}_loss"] = total_loss/n
            stats[f"{split}_dice"] = total_dice/n
        history.append(dict(epoch=epoch+1, **stats))
        improved = stats["val_loss"] < best
        if improved:
            best = stats["val_loss"]
        checkpoint = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "epoch": epoch+1, "best_val_loss": best,
            "config": run_cfg, "wavelength_angstrom": wave.tolist(),
            "normalization": manifest["normalization"],
        }
        torch.save(checkpoint, run/"checkpoint.pth")
        if improved:
            torch.save(checkpoint, run/"best.pth")
        (run/"history.json").write_text(json.dumps(history, indent=2))
        print(f"Epoch {epoch+1}/{epochs}: train={stats['train_loss']:.4f}, "
              f"val={stats['val_loss']:.4f}, val Dice={stats['val_dice']:.4f}", flush=True)
    print(f"Saved spectral model and wavelength metadata to {run}")


if __name__ == "__main__":
    main()
