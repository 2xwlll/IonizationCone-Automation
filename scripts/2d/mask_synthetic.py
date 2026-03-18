
import numpy as np
from pathlib import Path
from scipy.ndimage import rotate, affine_transform, gaussian_filter
import shutil
import random

# ===== CONFIG =====
N_IMAGES = 1000
IMAGE_SIZE = 128
PARSECS_PER_PIXEL = 10.0  # 1 pixel = 10 parsecs

DATA_ROOT = Path("data/2d")
IMG_OUTDIR = DATA_ROOT / "processed" / "synthetic"
MASK_OUTDIR = DATA_ROOT / "masks" / "synthetic"

# Create directories
for d in [IMG_OUTDIR, MASK_OUTDIR]:
    d.mkdir(parents=True, exist_ok=True)
    # Clear old files
    for f in d.iterdir():
        try:
            f.unlink()
        except IsADirectoryError:
            shutil.rmtree(f)

# ===== CONE GENERATOR =====
def generate_gaussian_cone(size, center, angle_deg, sigma_parsec,
                           amplitude=1.0, inclination_deg=0.0,
                           pixel_scale_parsec=PARSECS_PER_PIXEL):
    angle_rad = np.radians(angle_deg)
    inclination_rad = np.radians(inclination_deg)
    sigma_pix = sigma_parsec / pixel_scale_parsec

    y, x = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    x_shifted = x - center[0]
    y_shifted = y - center[1]

    # Rotate coordinates
    x_rot = x_shifted * np.cos(angle_rad) + y_shifted * np.sin(angle_rad)
    y_rot = -x_shifted * np.sin(angle_rad) + y_shifted * np.cos(angle_rad)

    # Foreshortening
    y_rot *= np.cos(inclination_rad)

    # One-sided cone mask
    mask = (y_rot > 0).astype(float)
    cone = amplitude * np.exp(-(x_rot**2 + y_rot**2) / (2 * sigma_pix**2)) * mask
    return cone

# ===== GALAXY BACKGROUND =====
def generate_galaxy_background(size, center=None, bulge_sigma=15, disk_scale=40):
    y, x = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    if center is None:
        center = (size // 2, size // 2)
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    bulge = np.exp(-r**2 / (2 * bulge_sigma**2))
    disk = np.exp(-r / disk_scale)
    return (0.5 * bulge + 0.5 * disk) / (0.5 * bulge + 0.5 * disk).max() * 0.1

# ===== DISTORTIONS & NOISE =====
def distort_image(image):
    theta = np.random.uniform(-10, 10)
    image = rotate(image, angle=theta, reshape=False, order=1, mode='reflect')
    if np.random.rand() < 0.5:
        sx, sy = np.random.uniform(0.8, 1.2, 2)
        transform = np.array([[sx, 0], [0, sy]])
        image = affine_transform(image, transform, offset=[0, 0], order=1, mode='reflect')
    return image

def add_realistic_noise(image):
    # Gaussian background noise
    image += np.random.normal(0, 0.02, image.shape)
    # JWST-style cosmic ray hits / blobs
    for _ in range(np.random.randint(1, 5)):
        x, y = np.random.randint(0, IMAGE_SIZE, 2)
        intensity = np.random.uniform(0.2, 1.0)
        image[max(0, y-1):y+2, max(0, x-1):x+2] += intensity
    return image

# ===== FULL IMAGE + MASK GENERATOR =====
def generate_synthetic_image(i):
    base = np.random.normal(0.0, 0.01, (IMAGE_SIZE, IMAGE_SIZE))
    center = (np.random.randint(40, 88), np.random.randint(40, 88))
    angle = np.random.uniform(0, 360)
    sigma = np.random.uniform(10, 25)
    amplitude = np.random.uniform(0.5, 1.5)
    inclination = np.random.uniform(0, 60)

    # Galaxy + cones
    background = generate_galaxy_background(IMAGE_SIZE, center=center)
    cone1 = generate_gaussian_cone(IMAGE_SIZE, center, angle, sigma, amplitude, inclination)
    cone2 = generate_gaussian_cone(IMAGE_SIZE, center, angle + 180, sigma, amplitude, inclination)
    cones_clean = cone1 + cone2  # This will be the perfect mask

    # Distort image for realism
    image = base + background + distort_image(cones_clean)
    image = add_realistic_noise(image)

    # Optional: Gaussian smoothing of mask to simulate PSF (soft edges)
    mask = gaussian_filter(cones_clean > 0, sigma=1.0).astype(np.float32)
    mask = (mask > 0.1).astype(np.float32)  # Threshold to binary

    # Save
    img_path = IMG_OUTDIR / f"synthetic_{i:04d}.npy"
    mask_path = MASK_OUTDIR / f"synthetic_{i:04d}.npy"
    np.save(img_path, image.astype(np.float32))
    np.save(mask_path, mask.astype(np.float32))

    return img_path.name, mask_path.name

# ===== MAIN LOOP =====
if __name__ == "__main__":
    print(f"Generating {N_IMAGES} perfect synthetic images + masks...")
    for i in range(N_IMAGES):
        img_name, mask_name = generate_synthetic_image(i)
        print(f"[{i+1:04d}/{N_IMAGES}] Image: {img_name}, Mask: {mask_name}")
    print("All synthetic images and masks generated successfully.")

