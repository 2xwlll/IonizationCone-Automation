#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

# Change these to match your repo layout
RAW_DIR = Path("data/cubes/raw/real")
SORTED_DIR = Path("data/cubes/sorted")

def sort_by_program():
    SORTED_DIR.mkdir(parents=True, exist_ok=True)

    for file in RAW_DIR.glob("*.fits"):
        # Example JWST filename: jw01234-o001_t001_nirspec.fits
        parts = file.name.split("_")
        program_id = parts[0][:7] if file.name.startswith("jw") else "unknown"
        
        target_dir = SORTED_DIR / program_id
        target_dir.mkdir(parents=True, exist_ok=True)
        
        shutil.move(str(file), target_dir / file.name)
        print(f"Moved {file.name} -> {target_dir}")

def sort_by_extension():
    SORTED_DIR.mkdir(parents=True, exist_ok=True)

    for file in RAW_DIR.iterdir():
        if file.is_file():
            ext = file.suffix.lstrip(".")
            if not ext:
                ext = "noext"
            target_dir = SORTED_DIR / ext
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(file), target_dir / file.name)
            print(f"Moved {file.name} -> {target_dir}")

if __name__ == "__main__":
    # pick which one you want:
    sort_by_program()
    # sort_by_extension()

