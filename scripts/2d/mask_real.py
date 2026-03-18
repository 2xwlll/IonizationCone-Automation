import os
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter, binary_opening, binary_closing, label
from skimage.morphology import remove_small_objects
from skimage.measure import label as skimage_label
from tqdm import tqdm

def generate_mask_real(img, threshold_percentile=85.0, gaussian_sigma=1.5, min_size=100):
    # Apply Gaussian filter to denoise
    smoothed = gaussian_filter(img, sigma=gaussian_sigma)

    # Compute threshold from percentile
    threshold = np.percentile(smoothed, threshold_percentile)
    binary_mask = smoothed > threshold

    # Morphological cleanup
    binary_mask = binary_opening(binary_mask)
    binary_mask = binary_closing(binary_mask)

    # Remove tiny noise blobs
    labeled_mask = skimage_label(binary_mask)
    cleaned_mask = remove_small_objects(labeled_mask, min_size=min_size)

    return (cleaned_mask > 0).astype(np.uint8)

def process_real_directory(
    input_dir="data/2d/processed/real",
    output_dir="data/2d/masks/real",
    threshold_percentile=85.0,
    gaussian_sigma=1.5,
    min_size=100
):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith(".fits")]

    for fname in tqdm(files, desc="Generating real masks"):
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname.replace(".fits", ".npy"))

        # Load FITS
        with fits.open(input_path) as hdul:
            image_data = hdul[0].data.astype(np.float32)

        # Generate mask
        mask = generate_mask_real(
            image_data,
            threshold_percentile=threshold_percentile,
            gaussian_sigma=gaussian_sigma,
            min_size=min_size,
        )

        # Save
        np.save(output_path, mask)

if __name__ == "__main__":
    process_real_directory()

