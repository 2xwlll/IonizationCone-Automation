import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.draw import polygon

def fit_cone_from_image(image: np.ndarray,
                        smooth_sigma: float = 2.0,
                        threshold_sigma: float = 3.0,
                        n_angle_bins: int = 360,
                        min_bins: int = 5) -> np.ndarray:
    """
    Identify ionization cone regions from a 2D [O III] broadband image.

    Parameters
    ----------
    image : np.ndarray
        2D broadband image highlighting cone emission.
    smooth_sigma : float
        Gaussian smoothing to reduce noise.
    threshold_sigma : float
        Threshold in units of background standard deviation.
    n_angle_bins : int
        Number of angular bins for polar analysis.
    min_bins : int
        Minimum consecutive angular bins to consider a cone.

    Returns
    -------
    cone_mask : np.ndarray (bool)
        Boolean mask of detected cone regions.
    """

    # Step 1: Smooth the image
    smoothed = gaussian_filter(image, sigma=smooth_sigma)

    # Step 2: Estimate background statistics
    background = np.median(smoothed)
    noise = np.std(smoothed)

    # Step 3: Threshold for significant emission
    signal = smoothed > (background + threshold_sigma * noise)

    # Step 4: Find AGN centroid (brightest pixel)
    y0, x0 = np.unravel_index(np.argmax(smoothed), smoothed.shape)

    # Step 5: Convert to polar coordinates around centroid
    yy, xx = np.indices(smoothed.shape)
    r = np.sqrt((xx - x0)**2 + (yy - y0)**2)
    theta = np.arctan2(yy - y0, xx - x0)  # radians, -pi..pi

    # Step 6: Bin by angle
    theta_bins = np.linspace(-np.pi, np.pi, n_angle_bins+1)
    flux_by_angle = np.zeros(n_angle_bins)
    for i in range(n_angle_bins):
        mask_bin = (theta >= theta_bins[i]) & (theta < theta_bins[i+1])
        flux_by_angle[i] = np.sum(smoothed[mask_bin])

    # Step 7: Identify bright angular sectors
    mean_flux = np.mean(flux_by_angle)
    cone_angles = np.where(flux_by_angle > mean_flux)[0]

    # Step 8: Build cone mask from angular sectors
    cone_mask = np.zeros_like(smoothed, dtype=bool)
    for idx in np.split(cone_angles, np.where(np.diff(cone_angles) > 1)[0] + 1):
        if len(idx) < min_bins:
            continue
        theta_min, theta_max = theta_bins[idx[0]], theta_bins[idx[-1]+1]
        r_max = np.max(r)
        sector_x = [x0] + list((x0 + r_max * np.cos([theta_min, theta_max])).ravel())
        sector_y = [y0] + list((y0 + r_max * np.sin([theta_min, theta_max])).ravel())
        rr, cc = polygon(sector_y, sector_x, smoothed.shape)
        cone_mask[rr, cc] = True

    return cone_mask


# --- Optional visualization ---
def show_mask(mask: np.ndarray, image: np.ndarray = None, title: str = "Cone Mask"):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,6))
    if image is not None:
        plt.imshow(image, origin='lower', cmap='inferno', alpha=0.5)
    plt.imshow(mask, origin='lower', cmap='viridis', alpha=0.5)
    plt.title(title)
    plt.colorbar()
    plt.show()

