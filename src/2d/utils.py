import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List

# -------------------------
# Single image visualization
# -------------------------
def show_image(image: np.ndarray, 
               mask: Optional[np.ndarray] = None,
               title: str = "Image",
               cmap: str = "inferno",
               mask_alpha: float = 0.5):
    """
    Display a 2D image with optional boolean mask overlay.
    """
    plt.figure(figsize=(6,6))
    plt.imshow(image, origin='lower', cmap=cmap)
    if mask is not None:
        plt.imshow(mask, origin='lower', cmap='viridis', alpha=mask_alpha)
    plt.title(title)
    plt.colorbar()
    plt.show()


# -------------------------
# Save image and mask
# -------------------------
def save_image(image: np.ndarray, path: str, mask: Optional[np.ndarray] = None):
    """
    Save image and optionally mask.
    - Saves image as .npy
    - Saves mask as .npy if provided
    """
    path = Path(path)
    np.save(path.with_suffix(".npy"), image)
    if mask is not None:
        np.save(path.with_name(path.stem + "_mask.npy"), mask)


# -------------------------
# Batch visualization
# -------------------------
def show_batch(images: List[np.ndarray], 
               masks: Optional[List[np.ndarray]] = None,
               titles: Optional[List[str]] = None,
               ncols: int = 3, mask_alpha: float = 0.5):
    """
    Display a batch of images in a grid.
    """
    n_images = len(images)
    nrows = int(np.ceil(n_images / ncols))
    
    plt.figure(figsize=(ncols*4, nrows*4))
    for i, img in enumerate(images):
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(img, origin='lower', cmap='inferno')
        if masks is not None:
            plt.imshow(masks[i], origin='lower', cmap='viridis', alpha=mask_alpha)
        if titles is not None:
            plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# -------------------------
# Simple batch loader
# -------------------------
def load_npy(path: str) -> np.ndarray:
    """Load a .npy file."""
    return np.load(path)

def load_batch(paths: List[str]) -> List[np.ndarray]:
    """Load multiple .npy files."""
    return [np.load(p) for p in paths]

