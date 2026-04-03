# broadband_cones/pipeline.py

import os
from pathlib import Path
from broadband_cones.project_to_image import load_cube, get_wavelength_axis, extract_cone_map, enhance_image, save_image
from broadband_cones.fit_cone import fit_cone_from_image
from broadband_cones import utils

def process_cube(fits_path: str, output_dir: str, enhance_method: str = "log"):
    """
    Process a single FITS cube:
    1. Collapse cube → broadband cone-highlighted image
    2. Extract cone mask
    3. Save both image and mask
    """
    cube, header = load_cube(fits_path)
    wav_axis = get_wavelength_axis(header, cube.shape[0])

    # Step 1: create cone-highlighted broadband image
    cone_image = extract_cone_map(cube, wav_axis)
    enhanced_image = enhance_image(cone_image, method=enhance_method)

    # Step 2: extract cone mask
    cone_mask = fit_cone_from_image(enhanced_image)

    # Step 3: save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(fits_path).stem
    image_path = output_dir / f"{base_name}_broadband"
    mask_path = output_dir / f"{base_name}_mask"

    utils.save_image(enhanced_image, image_path, mask=cone_mask)

    # Step 4: optional visualization
    utils.show_image(enhanced_image, mask=cone_mask, title=f"{base_name} Cone Map")


def run_pipeline(input_dir: str, output_dir: str, enhance_method: str = "log"):
    """Process all FITS cubes in a directory."""
    input_dir = Path(input_dir)
    fits_files = list(input_dir.glob("*.fits"))
    if not fits_files:
        print(f"No FITS files found in {input_dir}")
        return

    for fits_file in fits_files:
        print(f"Processing {fits_file.name}...")
        process_cube(fits_file, output_dir, enhance_method)

    print("Pipeline finished!")


# --- CLI Entrypoint ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Full broadband cone detection pipeline.")
    parser.add_argument("--input", required=True, help="Directory with FITS cubes")
    parser.add_argument("--output", required=True, help="Directory for processed images/masks")
    parser.add_argument("--method", choices=["log", "equalize", "none"], default="log",
                        help="Enhancement method for broadband image")

    args = parser.parse_args()
    run_pipeline(args.input, args.output, enhance_method=args.method)

