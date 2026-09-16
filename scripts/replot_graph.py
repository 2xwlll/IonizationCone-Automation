#!/usr/bin/env python3
"""
replot_curves.py — recreate training curves at presentation quality.

Read the approximate values off your existing plot and paste them
into the lists below, then run this script to produce a clean figure.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse
import json
import os

# --------------------------
# EITHER load from saved json OR hardcode from your plot
# --------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--history", type=str, default=None,
                    help="Path to history.json if saved")
parser.add_argument("--output",  type=str,
                    default="training_curves_final.png")
args = parser.parse_args()

if args.history and os.path.exists(args.history):
    with open(args.history) as f:
        history = json.load(f)
    train_loss = history["train"]
    val_loss   = history["val"]
    val_dice   = history["dice"]
    print(f"Loaded history from {args.history}")

else:
    # ── PASTE YOUR VALUES HERE ─────────────────────────────────────
    # Read these off your existing training_curves.png
    # Each number = one epoch, left to right
    train_loss = [
        3.16, 2.91, 2.75, 2.62, 2.51,
        2.37, 2.25, 2.15, 2.04, 1.91,
        1.83, 1.76, 1.68, 1.68, 1.60,
        1.53, 1.49, 1.44, 1.41, 1.38,
        1.34, 1.30, 1.27, 1.24, 1.21,
        1.18, 1.15, 1.12, 1.08, 1.05,
    ]
    val_loss = [
        3.17, 2.87, 2.73, 2.60, 2.47,
        2.58, 2.31, 2.16, 2.10, 1.93,
        1.94, 1.87, 1.88, 1.94, 1.88,
        1.88, 1.93, 1.87, 1.85, 1.88,
        1.86, 1.84, 1.87, 1.82, 1.82,
        1.83, 1.80, 1.81, 1.81, 1.80,
    ]
    val_dice = [
        0.32, 0.46, 0.40, 0.41, 0.45,
        0.38, 0.45, 0.66, 0.58, 0.65,
        0.65, 0.69, 0.55, 0.60, 0.62,
        0.76, 0.72, 0.72, 0.75, 0.72,
        0.74, 0.76, 0.76, 0.75, 0.78,
        0.76, 0.77, 0.76, 0.73, 0.74,
    ]
    print("Using hardcoded values — edit lists to match your plot exactly")

epochs = list(range(1, len(train_loss) + 1))

# --------------------------
# PLOT
# --------------------------
matplotlib.rcParams.update({
    "font.size":        16,
    "axes.titlesize":   20,
    "axes.labelsize":   18,
    "xtick.labelsize":  15,
    "ytick.labelsize":  15,
    "legend.fontsize":  15,
    "lines.linewidth":  2.5,
})

fig, ax1 = plt.subplots(figsize=(14, 7))

# loss curves on left axis
ax1.plot(epochs, train_loss, color="steelblue",  label="Train Loss", lw=2.5)
ax1.plot(epochs, val_loss,   color="orange",     label="Val Loss",   lw=2.5)
ax1.set_xlabel("Epoch",      fontsize=18)
ax1.set_ylabel("Loss",       fontsize=18)
ax1.tick_params(axis="both", labelsize=15)
ax1.set_xlim(1, len(epochs))

# dice on right axis
ax2 = ax1.twinx()
ax2.plot(epochs, val_dice, color="green", linestyle="--",
         label="Val Dice", lw=2.5)
ax2.set_ylabel("Dice Score", fontsize=18)
ax2.set_ylim(0, 1)
ax2.tick_params(axis="y", labelsize=15)

# combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(
    lines1 + lines2, labels1 + labels2,
    loc="center right", fontsize=15,
    framealpha=0.9
)

# peak dice annotation
best_dice  = max(val_dice)
best_epoch = val_dice.index(best_dice) + 1
ax2.annotate(
    f"Peak Dice = {best_dice:.2f}\n(epoch {best_epoch})",
    xy=(best_epoch, best_dice),
    xytext=(best_epoch + 2, best_dice - 0.08),
    fontsize=13,
    color="green",
    arrowprops=dict(arrowstyle="->", color="green", lw=1.5)
)

plt.title("UNet Training — Synthetic Ionization Cone Data", fontsize=20, pad=15)
plt.tight_layout()
plt.savefig(args.output, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved → {args.output}")
