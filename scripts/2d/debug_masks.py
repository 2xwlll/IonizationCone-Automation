import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.morphology import remove_small_objects
from pathlib import Path
import random

def generate_mask_dynamic(image: np.ndarray, threshold_percentile: float = 95.0, min_size: int = 10) -> np.ndarray:
    smoothed = gaussian_filter(image, sigma=1)
    threshold_value = np.percentile(smoothed, threshold_percentile)
    raw_mask = smoothed > threshold_value
    cleaned_mask = remove_small_objects(raw_mask, min_size=min_size)
    return cleaned_mask.astype(np.uint8)

def normalize(image: np.ndarray) -> np.ndarray:
    return (image - image.min()) / (image.max() - image.min() + 1e-8)

def is_blank(arr: np.ndarray, tolerance=1e-5) -> bool:
    return np.allclose(arr, 0, atol=tolerance)

def visualize_examples(image_paths, mask_dir, n=5):
    print(f"\n🔍 Visualizing {n} random examples...")
    sample_paths = random.sample(image_paths, min(n, len(image_paths)))

    for path in sample_paths:
        image = np.load(path)
        if image.ndim == 3:
            image = image[0]  # assume 3D input: take the first slice

        normed = normalize(image)
        mask_path = mask_dir / (path.stem + "_mask.npy")
        mask = np.load(mask_path) if mask_path.exists() else np.zeros_like(normed)

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].imshow(normed, cmap="viridis")
        axs[0].set_title(f"Image: {path.name}")
        axs[1].imshow(mask, cmap="gray")
        axs[1].set_title("Generated Mask")
        plt.tight_layout()
        plt.show()

def main():
    input_dir = Path("data/2d/processed/synthetic")
    output_dir = Path("data/2d/processed/synthetic_masks")
    output_dir.mkdir(parents=True, exist_ok=True)

    npy_files = list(input_dir.glob("*.npy"))
    print(f"📁 Found {len(npy_files)} .npy files in {input_dir}")

    for path in npy_files:
        image = np.load(path)
        if image.ndim == 3:
            image = image[0]

        if is_blank(image):
            print(f"⚠️ Blank input: {path.name}")
            continue

        normed = normalize(image)
        mask = generate_mask_dynamic(normed)

        if is_blank(mask):
            print(f"⚠️ Blank mask generated for: {path.name}")

        out_path = output_dir / f"{path.stem}_mask.npy"
        np.save(out_path, mask)

    # Visual check
    visualize_examples(npy_files, output_dir, n=5)

if __name__ == "__main__":
    main()

