# scripts/cubes/cleanup_cubes.py

from pathlib import Path

dirs_cubes = [
    "data/processed_cubes",
    "data/masks_cubes",
    "data/synthetic_cubes",
    "data/predicted_cubes",
    "data/raw_cubes"
]

def recursive_cleanup(paths, delete_empty_dirs=True):
    for dir_path in paths:
        p = Path(dir_path)
        if not p.exists():
            print(f"⏭️  Skipped missing: {dir_path}")
            continue

        file_count = 0
        for file in p.rglob("*"):
            if file.is_file():
                file.unlink()
                file_count += 1

        print(f"🧹 Cleared {file_count} files from {dir_path} (including subdirs)")

        if delete_empty_dirs:
            for folder in sorted(p.rglob("*"), reverse=True):
                if folder.is_dir() and not any(folder.iterdir()):
                    folder.rmdir()
            print(f"🗑️  Removed empty subdirectories in {dir_path}")

if __name__ == "__main__":
    recursive_cleanup(dirs_cubes)


