# scripts/2d/cleanup_2d.py

from pathlib import Path

# Targets: all real/synthetic subfolders and flat predicted dir
dirs_2d = [
    "data/2d/raw/real",
    "data/2d/raw/synthetic",
    "data/2d/processed/real",
    "data/2d/processed/synthetic",
    "data/2d/masks/real",
    "data/2d/masks/synthetic",
    "data/2d/predicted"
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
    recursive_cleanup(dirs_2d)

