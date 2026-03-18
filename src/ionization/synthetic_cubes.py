import numpy as np
import random
from pathlib import Path
from skimage.draw import polygon

# Output directories
out_data = Path("data/processed_3d")
out_masks = Path("data/masks")

# Ensure directories exist
out_data.mkdir(parents=True, exist_ok=True)
out_masks.mkdir(parents=True, exist_ok=True)
# Image shape
C, H, W = 16, 128, 128

# Clear existing data
for f in out_data.glob("input_*.npy"):
    f.unlink()
for f in out_masks.glob("mask_*.npy"):
    f.unlink()

# Mask
def generate_cone_mask(h, w, angle, spread, length, offset):
    cx, cy = w // 2 + offset[0], h // 2 + offset[1]
    theta = np.radians(angle)
    spread = np.radians(spread)

    x0, y0 = cx, cy
    x1 = cx + length * np.cos(theta - spread / 2)
    y1 = cy + length * np.sin(theta - spread / 2)
    x2 = cx + length * np.cos(theta + spread / 2)
    y2 = cy + length * np.sin(theta + spread / 2)

    rr, cc = polygon([y0, y1, y2], [x0, x1, x2], shape=(h, w))
    rr = np.clip(rr, 0, h - 1)
    cc = np.clip(cc, 0, w - 1)

    mask = np.zeros((h, w), dtype=np.float32)
    mask[rr, cc] = 1.0
    return mask

# Synthetic Loop
n_samples = 200
for i in range(n_samples):
    # Initialize blank mask
    mask = np.zeros((H, W), dtype=np.float32)

    # ✅ 1–2 cones per image
    for _ in range(random.randint(1, 2)):
        angle = random.uniform(0, 360)
        spread = random.uniform(15, 60)
        length = random.uniform(30, 60)
        offset = (random.randint(-20, 20), random.randint(-20, 20))
        mask += generate_cone_mask(H, W, angle, spread, length, offset)

    mask = np.clip(mask, 0, 1)  # Keep it binary

    # ✅ Add occasional "no-cone" cubes (~10% of data)
    if random.random() < 0.1:
        mask = np.zeros((H, W), dtype=np.float32)

    # ✅ Generate emission cube with:
    # - cone modulation
    # - random intensity per cone
    # - minor background clutter

    cube = np.zeros((C, H, W), dtype=np.float32)
    x = np.linspace(-1, 1, W)
    y = np.linspace(-1, 1, H)
    xx, yy = np.meshgrid(x, y)
    d = np.sqrt(xx**2 + yy**2)

    # Optional: Spectral shift effect (Doppler)
    channel_shift = random.randint(-2, 2)

    for c in range(C):
        sigma = 0.4 + 0.1 * np.random.rand()
        intensity = np.exp(-(d**2) / (2.0 * sigma**2))
        noise = 0.1 * np.random.randn(H, W)

        # Vary how much mask contributes
        mask_weight = random.uniform(0.3, 0.7)

        plane = intensity + noise + mask_weight * mask

        # Optional background blobs (~30% chance)
        if random.random() < 0.3:
            for _ in range(random.randint(1, 3)):
                bx = random.randint(0, W - 1)
                by = random.randint(0, H - 1)
                blob_sigma = random.uniform(5, 15)
                cx = 2 * (bx / W) - 1
                cy = 2 * (by / H) - 1
                blob = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2.0 * (blob_sigma / W)**2))
                plane += 0.2 * blob

        cube_idx = np.clip(c + channel_shift, 0, C - 1)
        cube[c] = plane

    cube -= cube.min()
    cube /= cube.max()

    # Save files
    np.save(out_data / f"input_{i:03}.npy", cube)
    np.save(out_masks / f"mask_{i:03}.npy", mask)


print("✅ Overwritten synthetic dataset with 200 samples generated.")

