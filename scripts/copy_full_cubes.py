# scripts/copy_full_cubes.py
from pathlib import Path
import numpy as np

raw_dir = Path("data/raw")
processed_dir = Path("data/processed")
processed_dir.mkdir(exist_ok=True)

files = sorted(raw_dir.glob("fake_agn_*.npy"))
for i, f in enumerate(files):
    cube = np.load(f)  # already (16, 128, 128)
    out_path = processed_dir / f"input_{i:03}.npy"
    np.save(out_path, cube)
    print(f"Saved {out_path.name}, shape = {cube.shape}")

