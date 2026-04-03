# test_pipeline.py
from pathlib import Path
import broadband_cones as bc

# --- Configuration ---
fits_file = "data/raw/NGC1234_cube.fits"  # replace with your test FITS cube
output_dir = "data/processed_test"
enhance_method = "log"  # options: log, equalize, none

# --- Step 1: Load cube ---
cube, header = bc.load_cube(fits_file)
wav_axis = bc.get_wavelength_axis(header, cube.shape[0])

# --- Step 2: Create broadband image ---
broadband_image = bc.project_cube(cube, wav_axis)
enhanced_image = bc.enhance_image(broadband_image, method=enhance_method)

# --- Step 3: Extract cone mask ---
cone_mask = bc.fit_cone(enhanced_image)

# --- Step 4: Save and visualize ---
output_path = Path(output_dir)
output_path.mkdir(parents=True, exist_ok=True)

# Save image and mask
bc.save_image(enhanced_image, output_path / "NGC1234_broadband", mask=cone_mask)

# Visualize results
bc.show(enhanced_image, mask=cone_mask, title="NGC1234 Cone Detection")

print("Test pipeline finished successfully!")

