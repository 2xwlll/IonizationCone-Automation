"""
broadband_cones: Tools for detecting ionization cones from real FITS cubes.

Modules:
- project_to_image: cube → 2D broadband image
- fit_cone: extract cone mask from 2D image
- utils: visualization and saving helpers
- pipeline: orchestrate full processing workflow
"""

# --- Project to image functions ---
from .project_to_image import (
    load_cube,
    get_wavelength_axis,
    collapse_band,
    extract_cone_map,
    enhance_image,
    save_image as save_broadband_image,
    plot_image
)

# --- Fit cone functions ---
from .fit_cone import fit_cone_from_image, show_mask as show_cone_mask

# --- Utilities ---
from .utils import (
    show_image,
    save_image,
    show_batch,
    load_npy,
    load_batch
)

# --- Pipeline ---
from .pipeline import run_pipeline, process_cube

# --- Package version ---
__version__ = "1.0.0"

# --- Optional convenience aliases ---
# For super clean imports:
# e.g., `from broadband_cones import project_cube, fit_cone, show`
project_cube = extract_cone_map
fit_cone = fit_cone_from_image
show = show_image

