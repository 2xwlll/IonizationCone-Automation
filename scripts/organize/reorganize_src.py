import os
import shutil

# Define the source root
SRC_DIR = "src"

# Define new structure
NEW_STRUCTURE = {
    "datasets": ["ionization_dataset.py", "log_which_hdu.py", "resize_real.py"],
    "models": ["model_2d.py", "model_cube.py"],
    "losses": ["losses/combined_BCE_Dice.py", "losses/dice_loss.py"],
    "utils": ["utils/normalize.py", "utils/plot.py"],
    "scripts": ["data_pipeline.py", "import_mast_data.py"],
    "legacy": ["ionization.egg-info"]
}

# Create new folders and move files
for folder, files in NEW_STRUCTURE.items():
    dest_folder = os.path.join(SRC_DIR, folder)
    os.makedirs(dest_folder, exist_ok=True)
    for file in files:
        src_path = os.path.join(SRC_DIR, file)
        dest_path = os.path.join(dest_folder, os.path.basename(file))
        if os.path.exists(src_path):
            shutil.move(src_path, dest_path)
        elif os.path.isdir(src_path):
            # Move the entire directory
            shutil.move(src_path, dest_folder)

# Add empty __init__.py files to all new folders
for root, dirs, _ in os.walk(SRC_DIR):
    for dir_name in dirs:
        init_path = os.path.join(root, dir_name, "__init__.py")
        if not os.path.exists(init_path):
            open(init_path, "w").close()

print("✅ Reorganization complete.")

