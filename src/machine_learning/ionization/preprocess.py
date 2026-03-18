import numpy as np
import pandas as pd
from astropy.io import fits

def load_fits_image(file_path):
    """
    Loads a 2D FITS image and returns the data array.
    """
    with fits.open(file_path) as hdul:
        data = hdul[0].data
    return data

def normalize_image(img, method='zscore'):
    """
    Normalizes a 2D image using z-score or min-max.
    """
    if method == 'zscore':
        mean = np.mean(img)
        std = np.std(img)
        return (img - mean) / std
    elif method == 'minmax':
        min_val = np.min(img)
        max_val = np.max(img)
        return (img - min_val) / (max_val - min_val)
    else:
        raise ValueError("Unknown normalization method")

def threshold_image(img, threshold):
    """
    Applies a basic intensity threshold to an image.
    """
    return (img > threshold).astype(int)

