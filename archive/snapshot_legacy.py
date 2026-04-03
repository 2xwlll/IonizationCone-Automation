import os
import shutil
import datetime

def snapshot_legacy(project_root=".", legacy_dir="legacy"):
    """
    Take a snapshot of the current project into a legacy folder.
    Avoids copying the legacy folder itself (prevents recursion).
    Each snapshot gets a unique timestamp so older ones aren't overwritten.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    snapshot_name = f"IonizationCone-Automization{timestamp}"
    snapshot_path = os.path.join(legacy_dir, snapshot_name)

    # Ensure legacy dir exists
    os.makedirs(legacy_dir, exist_ok=True)

    # Ignore the legacy folder itself + .git, venv, __pycache__
    ignore_list = shutil.ignore_patterns(
        ".git", "venv", "__pycache__", "*.pyc", "*.pyo", "*.pyd", ".DS_Store", legacy_dir
    )

    # Copy project
    shutil.copytree(
        project_root,
        snapshot_path,
        ignore=ignore_list
    )

    print(f"Snapshot created at: {snapshot_path}")
    return snapshot_path

def git_commit_snapshot(snapshot_path):
    """
    Optionally add and commit the snapshot to git.
    """
    import subprocess

    try:
        subprocess.run(["git", "add", snapshot_path], check=True)
        commit_msg = f"Add legacy snapshot {os.path.basename(snapshot_path)}"
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
        subprocess.run(["git", "push"], check=True)
        print("Snapshot committed and pushed to GitHub")
    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {e}")

if __name__ == "__main__":
    path = snapshot_legacy()
    # Uncomment the next line if you want the snapshot committed automatically
    # git_commit_snapshot(path)

