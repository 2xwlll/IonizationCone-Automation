import os
import numpy as np
from astropy.io import fits
from astropy.convolution import convolve, Gaussian2DKernel
from skimage.transform import rotate
from skimage.util import random_noise
import argparse
import random

def add_realism(image, psf_sigma=1.0, noise_mode='poisson', max_rotation=360):
    """Applies noise, blur, and random rotation."""
    # 1. PSF Blur (Gaussian)
    kernel = Gaussian2DKernel(x_stddev=psf_sigma)
    blurred = convolve(image, kernel)

    # 2. Noise (Poisson or Gaussian)
    noisy = random_noise(blurred, mode=noise_mode)

    # 3. Rotation (0–360 degrees)
    angle = random.uniform(0, max_rotation)
    rotated = rotate(noisy, angle, mode='reflect')

    # 4. Random flipping
    if random.choice([True, False]):
        rotated = np.fliplr(rotated)
    if random.choice([True, False]):
        rotated = np.flipud(rotated)

    return rotated.astype(np.float32)

def maybe_remove_cone(image, probability=0.02):
    """Randomly removes cone structure to simulate absence."""
    if random.random() < probability:
        return np.zeros_like(image)
    return image

def normalize_image(image):
    """Normalize to [0, 1] robustly."""
    p_min, p_max = np.percentile(image, [1, 99])
    return np.clip((image - p_min) / (p_max - p_min + 1e-8), 0, 1)

def process_fits(input_path, output_path, realism=True):
    with fits.open(input_path) as hdul:
        data = hdul[0].data.astype(np.float32)

        # Safety check: ensure 2D input
        if data.ndim != 2:
            print(f"Skipping non-2D file: {input_path}")
            return

        # Add noise, blur, rotation
        if realism:
            data = maybe_remove_cone(data)
            data = add_realism(data)

        data = normalize_image(data)

        hdu = fits.PrimaryHDU(data)
        hdu.header['PROCESSED'] = True
        hdu.header['REALISTIC'] = realism
        hdu.writeto(output_path, overwrite=True)

        print(f"Processed: {input_path} → {output_path}")

def process_all(input_dir, output_dir, realism=True):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if not file.endswith(".fits"):
            continue
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file)
        process_fits(input_path, output_path, realism)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process synthetic FITS files with realism for UNet training.")
    parser.add_argument("--input_dir", required=True, help="Directory of raw synthetic FITS images")
    parser.add_argument("--output_dir", required=True, help="Directory to save processed FITS images")
    parser.add_argument("--no_realism", action="store_true", help="Disable realism (noise, blur, etc.)")

    args = parser.parse_args()
    process_all(args.input_dir, args.output_dir, realism=not args.no_realism)

