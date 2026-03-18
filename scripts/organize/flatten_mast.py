from pathlib import Path
import shutil

ROOT = Path("data/2d/raw/real")

for fits in ROOT.rglob("*_i2d.fits"):
    target = ROOT / fits.name
    if not target.exists():
        shutil.copy(fits, target)
        print(f"Copied {fits.name}")

