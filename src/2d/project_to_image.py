# src/broadband_cones/project_to_image.py

import numpy as np
from astropy.io import fits
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import exposure

def load_cube(fits_path):
    """Load the first HDU with image data as a float32 array."""
    with fits.open(fits_path) as hdul:
        for hdu in hdul:
            if hdu.data is not None:
                data = hdu.data.astype(np.float32)
                header = hdu.header
                print(f"[Info] Loaded cube shape: {data.shape}")
                return data, header
    raise ValueError(f"No image data found in {fits_path}")

def get_wavelength_axis(header, n_pix):
    """Return wavelength axis, fallback to simple range if missing."""
    try:
        crval3 = header.get("CRVAL3")
        cdelt3 = header.get("CDELT3")
        crpix3 = header.get("CRPIX3")
        if None in (crval3, cdelt3, crpix3):
            return np.arange(n_pix)
        wav = crval3 + (np.arange(n_pix) + 1 - crpix3) * cdelt3
        # Convert to microns if in Angstroms (>1000 Å)
        if np.max(wav) > 1000:
            wav /= 1e4
            print("[Info] Wavelength axis converted to microns.")
        return wav
    except Exception:
        return np.arange(n_pix)

def collapse_band(cube: np.ndarray, wav_axis: np.ndarray,
                  line_center: float, width: float) -> np.ndarray:
    """Collapse a band around line_center ± width/2 into a 2D image."""
    mask = (wav_axis >= line_center - width/2) & (wav_axis <= line_center + width/2)
    if not np.any(mask):
        print(f"[Warning] No slices in range {line_center:.3e} ± {width/2:.3e}")
        return np.zeros(cube.shape[1:], dtype=np.float32)
    return np.sum(cube[mask, :, :], axis=0)

def extract_cone_map(cube: np.ndarray, wav_axis: np.ndarray,
                     line_center: float = 0.5007, line_width: float = 0.01,
                     cont_width: float = 0.02, cont_offset: float = 0.05) -> np.ndarray:
    """
    Build a cone-highlighted image:
    - Collapse around observed [O III] line.
    - Collapse a continuum band offset from line.
    - Subtract continuum from line.
    """
    line_image = collapse_band(cube, wav_axis, line_center, line_width)
    cont_image = collapse_band(cube, wav_axis, line_center + cont_offset, cont_width)
    cone_map = line_image - cont_image
    return cone_map

def enhance_image(image: np.ndarray, method: str = "log") -> np.ndarray:
    """Normalize + optionally enhance faint details."""
    # Clip negatives
    image = image - np.nanmin(image)
    
    finite_pixels = np.isfinite(image)
    if np.any(finite_pixels):
        high = np.nanpercentile(image[finite_pixels], 99.9)
        if high > 0:
            image /= high
        else:
            print("[Warning] Non-positive percentile (image nearly flat).")
            image /= np.nanmax(image[finite_pixels]) + 1e-8
    else:
        print("[Warning] No finite pixels in image.")
        image[:] = 0

    # Apply enhancement method
    if method == "log":
        image = np.log1p(image)
        image /= np.nanmax(image) + 1e-8
    elif method == "equalize":
        image = exposure.equalize_adapthist(image, clip_limit=0.03)

    return image

def save_image(image: np.ndarray, output_path: str, header=None, as_fits: bool = True):
    """Save image as FITS (default) or NPY."""
    output_path = Path(output_path)
    if as_fits:
        hdu = fits.PrimaryHDU(image, header=header)
        hdu.writeto(output_path.with_suffix(".fits"), overwrite=True)
    else:
        np.save(output_path.with_suffix(".npy"), image)
    print(f"[Done] Saved enhanced map → {output_path.with_suffix('.fits')}")

def plot_image(image: np.ndarray, title="Cone Map"):
    plt.imshow(image, origin="lower", cmap="inferno")
    plt.title(title)
    plt.colorbar()
    plt.show()

# --- CLI Entrypoint ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert cube → broadband cone-highlighted image.")
    parser.add_argument("--input", required=True, help="Path to input FITS cube")
    parser.add_argument("--output", required=True, help="Base path for output image")
    parser.add_argument("--method", choices=["log", "equalize", "none"], default="log",
                        help="Enhancement method")
    parser.add_argument("--line_center", type=float, default=0.5007,
                        help="Observed [O III] line center in microns (default 0.5007)")
    parser.add_argument("--line_width", type=float, default=0.01,
                        help="Width of line band in microns")
    parser.add_argument("--cont_offset", type=float, default=0.05,
                        help="Offset of continuum band from line in microns")
    parser.add_argument("--cont_width", type=float, default=0.02,
                        help="Width of continuum band in microns")

    args = parser.parse_args()

    cube, header = load_cube(args.input)
    wav_axis = get_wavelength_axis(header, cube.shape[0])

    cone_map = extract_cone_map(
        cube, wav_axis,
        line_center=args.line_center,
        line_width=args.line_width,
        cont_width=args.cont_width,
        cont_offset=args.cont_offset
    )
    enhanced = enhance_image(cone_map, method=args.method)
    save_image(enhanced, args.output, header=header, as_fits=True)
    plot_image(enhanced)

