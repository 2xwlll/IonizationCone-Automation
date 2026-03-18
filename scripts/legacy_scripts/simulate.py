import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os

def generate_cone(size=128, peak_intensity=10, noise_level=0.5, visualize=False):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    
    # 2D Gaussian cone
    sigma_x, sigma_y = 0.3, 0.1
    cone = peak_intensity * np.exp(-(xx**2 / (2 * sigma_x**2) + yy**2 / (2 * sigma_y**2)))
    
    # Add noise
    noise = np.random.normal(0, noise_level, size=(size, size))
    img = cone + noise
    img = np.clip(img, 0, None)  # No negative intensities

    if visualize:
        plt.imshow(img, cmap='viridis')
        plt.title("Synthetic Ionization Cone")
        plt.colorbar()
        plt.show()

    return img

def save_as_fits(img, filepath):
    hdu = fits.PrimaryHDU(img)
    hdu.writeto(filepath, overwrite=True)

def save_as_npy(img, filepath):
    np.save(filepath, img)

if __name__ == "__main__":
    output_dir = "data/synthetic"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(5):
        img = generate_cone(visualize=True)
        save_as_fits(img, f"{output_dir}/cone_{i}.fits")
        save_as_npy(img, f"{output_dir}/cone_{i}.npy")

    print("✅ Synthetic cones saved in 'data/synthetic'")

