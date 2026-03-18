import os

# Base data directory (edit only this if you move everything later)
DATA_DIR = os.path.expanduser("~/Programming/IonizationCone-Automization/data")

# ---- Cubes ----
CUBES_RAW_REAL       = os.path.join(DATA_DIR, "cubes/raw/real")
CUBES_RAW_SYNTHETIC  = os.path.join(DATA_DIR, "cubes/raw/synthetic")
CUBES_PROCESSED_REAL = os.path.join(DATA_DIR, "cubes/processed/real")
CUBES_PROCESSED_SYN  = os.path.join(DATA_DIR, "cubes/processed/synthetic")
CUBES_MASKS_REAL     = os.path.join(DATA_DIR, "cubes/masks/real")
CUBES_MASKS_SYN      = os.path.join(DATA_DIR, "cubes/masks/synthetic")
CUBES_SORTED         = os.path.join(DATA_DIR, "cubes/sorted")

# ---- 2D ----
TWO_D_RAW_REAL       = os.path.join(DATA_DIR, "2d/raw/real")
TWO_D_PROCESSED_REAL = os.path.join(DATA_DIR, "2d/processed/real")
TWO_D_PROCESSED_SYN  = os.path.join(DATA_DIR, "2d/processed/synthetic")
TWO_D_MASKS_REAL     = os.path.join(DATA_DIR, "2d/masks/real")
TWO_D_MASKS_SYN      = os.path.join(DATA_DIR, "2d/masks/synthetic")
TWO_D_PREDICTED      = os.path.join(DATA_DIR, "2d/predict/predicted")
TWO_D_PREDICT_RAW    = os.path.join(DATA_DIR, "2d/predict/predict raw")

