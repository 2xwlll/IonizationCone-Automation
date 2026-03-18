# src/data_pipeline.py

import numpy as np
from astropy.io import fits
from pathlib import Path
from skimage import filters, transform

# Set up paths
RAW_DIR = Path("data/raw")
PROCESSED_DIR_2D = Path("data/processed_2d")
PROCESSED_DIR_3D = Path("data/processed_3d")

# Ensure output dirs exist
PROCESSED_DIR_2D.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR_3D.mkdir(parents=True, exist_ok=True)

def load_raw_data(file_path):
    """
    Loads a .npy or .fits file and returns a numpy array.
    """
    print(f"Loading data from {file_path}")
    if file_path.suffix == ".npy":
        return np.load(file_path)
    elif file_path.suffix == ".fits":
        with fits.open(file_path) as hdul:
            for hdu in hdul:
                if hdu.data is not None and hdu.data.ndim == 3:
                    return hdu.data.astype(np.float32)
            raise ValueError("No valid 3D data found in FITS file.")
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def normalize(data):
    data = data - np.min(data)
    data = data / (np.max(data) + 1e-8)
    return data

def resize_spatial(data, target_shape):
    return transform.resize(data, target_shape, mode="reflect", anti_aliasing=True)

def preprocess_2d(data, target_shape=(128, 128)):
    """
    Preprocess input by collapsing to 2D, normalizing, denoising, and resizing.
    """
    if data.ndim == 3:
        print("Collapsing cube to 2D image...")
        data = np.sum(data, axis=0)
    data = normalize(data)
    data = filters.gaussian(data, sigma=0.5)
    data = resize_spatial(data, target_shape)
    return data

def preprocess_3d(data, target_shape=(16, 128, 128)):
    """
    Preprocess full 3D cube: trim spectral axis, normalize, resize.
    """
    data = normalize(data)
    # Spectral slice trimming or interpolation
    if data.shape[0] != target_shape[0]:
        if data.shape[0] > target_shape[0]:
            mid = data.shape[0] // 2
            half = target_shape[0] // 2
            data = data[mid-half:mid+half]  # center crop
        else:
            pad = target_shape[0] - data.shape[0]
            data = np.pad(data, ((0, pad), (0, 0), (0, 0)), mode='constant')
    resized_cube = np.stack([resize_spatial(slice, target_shape[1:]) for slice in data], axis=0)
    return resized_cube

def process_all():
    for path in RAW_DIR.glob("*.fits"):
        try:
            data = load_raw_data(path)

            # Save 3D
            cube_3d = preprocess_3d(data)
            out3d_path = PROCESSED_DIR_3D / f"{path.stem}.npy"
            np.save(out3d_path, cube_3d)
            print(f"✅ Saved 3D cube: {out3d_path}")

            # Save 2D
            img_2d = preprocess_2d(data)
            out2d_path = PROCESSED_DIR_2D / f"{path.stem}.npy"
            np.save(out2d_path, img_2d)
            print(f"✅ Saved 2D projection: {out2d_path}")

        except Exception as e:
            print(f"❌ Failed on {path.name}: {e}")

if __name__ == "__main__":
    process_all()

