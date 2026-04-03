import os
import numpy as np
from astropy.io import fits
from astropy.convolution import convolve, Gaussian2DKernel
from tqdm import trange
import random

# === PARAMETERS ===
IMAGE_SIZE = 128
NUM_IMAGES = 100  # You can change this
OUT_IMAGE_DIR = "data/synthetic/images"
OUT_MASK_DIR = "data/synthetic/masks"
NOISE_STD = 0.05  # Realistic background noise

os.makedirs(OUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)

# === HELPERS ===
def generate_random_cone(img_size):
    """Returns (image, mask) tuple with one or two random cones"""
    img = np.zeros((img_size, img_size), dtype=np.float32)
    mask = np.zeros((img_size, img_size), dtype=np.uint8)

    num_cones = random.choice([0, 1, 2])  # Sometimes no cone = realistic!
    
    for _ in range(num_cones):
        x0 = random.randint(int(img_size * 0.2), int(img_size * 0.8))
        y0 = random.randint(int(img_size * 0.2), int(img_size * 0.8))
        theta = random.uniform(0, 2 * np.pi)
        width = random.uniform(5, 15)
        length = random.uniform(20, 40)

        for y in range(img_size):
            for x in range(img_size):
                dx = x - x0
                dy = y - y0
                r = np.hypot(dx, dy)
                angle = np.arctan2(dy, dx) - theta
                angle = (angle + np.pi) % (2 * np.pi) - np.pi

                in_cone = (abs(angle) < np.pi / 12) and (r < length)
                if in_cone:
                    intensity = np.exp(-r**2 / (2 * width**2))
                    img[y, x] += intensity
                    mask[y, x] = 1

    return img, mask

def add_noise_and_blur(image, std=NOISE_STD):
    noisy = image + np.random.normal(0, std, image.shape).astype(np.float32)
    kernel = Gaussian2DKernel(x_stddev=1)
    return convolve(noisy, kernel)

# === MAIN LOOP ===
for i in trange(NUM_IMAGES, desc="Generating synthetic FITS"):
    img, mask = generate_random_cone(IMAGE_SIZE)
    img = add_noise_and_blur(img)

    # Normalize
    img -= img.min()
    img /= (img.max() + 1e-8)

    # Save FITS files
    fits.writeto(f"{OUT_IMAGE_DIR}/synthetic_{i:04d}.fits", img, overwrite=True)
    fits.writeto(f"{OUT_MASK_DIR}/synthetic_{i:04d}_mask.fits", mask.astype(np.uint8), overwrite=True)

