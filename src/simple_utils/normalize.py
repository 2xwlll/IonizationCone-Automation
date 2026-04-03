import numpy as np

def normalize_image(img, clip_min=0.0, clip_max=99.5):
    """
    Normalize image by clipping and scaling to [0, 1].
    """
    img = np.nan_to_num(img)  # Clean up NaNs/Infs
    vmin, vmax = np.percentile(img, [clip_min, clip_max])
    img = np.clip(img, vmin, vmax)
    if vmax - vmin == 0:
        return img * 0.0
    return (img - vmin) / (vmax - vmin)

