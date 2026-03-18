# scripts/organize/create_data_dirs.py

from pathlib import Path

BASE = Path("data")
modes = ["2d", "cubes"]
subdirs = ["raw", "masks", "processed"]
sources = ["synthetic", "real"]

for mode in modes:
    base_path = BASE / mode
    (base_path / "predicted").mkdir(parents=True, exist_ok=True)
    for sub in subdirs:
        for source in sources:
            (base_path / sub / source).mkdir(parents=True, exist_ok=True)

print("✅ Created nested data directory structure for 2D and cube modes.")

