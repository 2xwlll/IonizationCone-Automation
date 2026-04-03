import numpy as np
import os
import re

def make_cone_mask(shape, center, angle_deg, length, cone_width_deg):
    H, W = shape
    Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    dx = X - center[0]
    dy = Y - center[1]
    theta = np.degrees(np.arctan2(dy, dx)) % 360
    r = np.sqrt(dx**2 + dy**2)

    angle = angle_deg % 360
    angle_min = (angle - cone_width_deg/2) % 360
    angle_max = (angle + cone_width_deg/2) % 360

    if angle_min < angle_max:
        angle_mask = (theta >= angle_min) & (theta <= angle_max)
    else:
        angle_mask = (theta >= angle_min) | (theta <= angle_max)

    dist_mask = r <= length
    return (angle_mask & dist_mask).astype(np.uint8)

def generate_masks(input_dirs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    mask_id = 0

    for input_dir in input_dirs:
        for fname in sorted(os.listdir(input_dir)):
            if not fname.endswith('.npy'):
                continue

            fpath = os.path.join(input_dir, fname)
            img = np.load(fpath)

            # Try to reduce singleton dims (like (1, H, W) or (H, W, 1))
            img = np.squeeze(img)

            # Only accept clean 2D arrays
            if img.ndim != 2:
                print(f"Skipping {fname} — unexpected shape {img.shape}")
                continue

            h, w = img.shape
           

            h, w = img.shape
            center = (w // 2, h // 2)
            angle = np.random.uniform(0, 360)
            length = np.random.uniform(10, min(h, w)//2)
            width = np.random.uniform(20, 60)

            mask = make_cone_mask((h, w), center, angle, length, width)
            mask_name = f"mask_{mask_id}.npy"
            np.save(os.path.join(output_dir, mask_name), mask)
            print(f"Saved {mask_name}")
            mask_id += 1

if __name__ == '__main__':
    input_dirs = ['data/synthetic', 'data/raw_sliced']
    output_dir = 'data/masks'
    generate_masks(input_dirs, output_dir)

