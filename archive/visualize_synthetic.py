#!/usr/bin/env python3
"""
visualize_synthetic.py

Loads and displays a random synthetic 2D image + its mask side by side.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

# -----------------------
# CONFIG
# -----------------------
IMAGE_FOLDER = Path("data/2d/raw/synthetic")
MASK_FOLDER  = Path("data/2d/masks/synthetic")

# -----------------------
# Pick a random image
# -----------------------
npy_files = list(IMAGE_FOLDER.glob("*.npy"))
if not npy_files:
    raise FileNotFoundError(f"No .npy files found in {IMAGE_FOLDER}")

image_path = random.choice(npy_files)
print("Loading random image:", image_path)

# -----------------------
# Match mask by filename
# -----------------------
mask_path = MASK_FOLDER / image_path.name
if not mask_path.exists():
    raise FileNotFoundError(f"Mask not found for {image_path.name} in {MASK_FOLDER}")

# -----------------------
# Load arrays
# -----------------------
image = np.load(image_path)
mask  = np.load(mask_path)

# -----------------------
# Plot side by side
# -----------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(image, origin="lower", cmap="inferno")
axes[0].set_title(f"Image: {image_path.name}")

axes[1].imshow(mask, origin="lower", cmap="gray")
axes[1].set_title(f"Mask: {mask_path.name}")
axes[1].axis("off")

plt.tight_layout()
plt.show()

