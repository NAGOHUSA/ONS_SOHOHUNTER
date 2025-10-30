#!/usr/bin/env python3
"""
purge_old_data.py
-----------------
Delete every file under detections/ that is older than 24 hours.
Directories are never removed.
"""

import subprocess
from datetime import datetime, timedelta
from pathlib import Path

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
DETECTIONS_ROOT = Path("detections")
KEEP_HOURS = 24                # <-- change to any number of hours you need
DRY_RUN = False                # Set True to see what would be deleted
COMMIT_CHANGES = True          # False → only delete, no git commit


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------
def run_cmd(cmd: str, check: bool = True):
    """Run a shell command in the repo root."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True,
        cwd=DETECTIONS_ROOT.parent
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{result.stderr}")
    return result


def is_file_old(path: Path, cutoff: datetime) -> bool:
    """Return True if the file’s mtime is older than `cutoff`."""
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        return mtime < cutoff
    except OSError:
        return False


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    if not DETECTIONS_ROOT.exists():
        print(f"'{DETECTIONS_ROOT}' not found – nothing to purge.")
        return

    cutoff = datetime.now() - timedelta(hours=KEEP_HOURS)
    print(f"Purging files under '{DETECTIONS_ROOT}' older than {cutoff:%Y-%m-%d %H:%M} "
          f"(keep last {KEEP_HOURS} h)")

    files_to_delete = []
    for file_path in DETECTIONS_ROOT.rglob("*"):
        if file_path.is_file() and is_file_old(file_path, cutoff):
            files_to_delete.append(file_path)

    if not files_to_delete:
        print("No old files found – everything is newer than the cutoff.")
        return

    print(f"\nFound {len(files_to_delete)} file(s) to delete:")
    for p in sorted(files_to_delete):
        mtime = datetime.fromtimestamp(p.stat().st_mtime)
        print(f"  - {p.relative_to(DETECTIONS_ROOT)} (mtime: {mtime:%Y-%m-%d %H:%M})")

    if DRY_RUN:
        print("\n[DRY RUN] – no files deleted, no commit.")
        return

    # ---- DELETE ----
    print("\nDeleting files...")
    deleted = 0
    for p in files_to_delete:
        try:
            p.unlink()
            print(f"  Deleted: {p.name}")
            deleted += 1
        except OSError as e:
            print(f"  Failed to delete {p.name}: {e}")

    # ---- GIT COMMIT (optional) ----
    if COMMIT_CHANGES:
        print("\nCommitting changes to git...")
        try:
            run_cmd("git add detections")
            msg = f"purge: remove files older than {KEEP_HOURS}h [{datetime.now().isoformat()}]"
            run_cmd(f'git commit -m "{msg}"')
            print("  Commit successful.")
        except Exception as e:
            print(f"  Git commit failed: {e}")

    print(f"\nPurge complete! {deleted} file(s) removed.")


if __name__ == "__main__":
    main()
