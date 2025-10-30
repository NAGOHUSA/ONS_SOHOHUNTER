#!/usr/bin/env python3
"""
purge_old_data.py
-----------------
Standalone script that removes every file under detections/ whose
filename contains a date older than today.

Supported filename pattern (your repo):
    candidates_20251029_232405.json
    crop_20251030_010203_0.jpg
    …any file that contains YYYYMMDD_HHMMSS

Commits the deletions automatically.
"""

import os
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
DETECTIONS_ROOT = Path("detections")   # root folder to scan
KEEP_DAYS = 0                          # 0 = keep only today, 7 = keep last 7 days
DRY_RUN = False                        # True → only print, no delete/commit
USE_MTIME_FALLBACK = False             # set True only if you trust file mtime

# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------
def run_cmd(cmd: str, check: bool = True):
    """Run a shell command, raise on error if check=True."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, cwd=DETECTIONS_ROOT.parent
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{result.stderr}")
    return result


def parse_date_from_name(name: str) -> datetime.date | None:
    """
    Extract YYYYMMDD from a filename.
    Returns the date (as datetime.date) or None if not found.
    """
    # Pattern: 20251029 (anywhere in the name)
    m = re.search(r"(\d{8})", name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d").date()
    except ValueError:
        return None


def file_is_old(path: Path, cutoff: datetime.date) -> bool:
    """Return True if file should be deleted."""
    # 1. Try filename date
    name_date = parse_date_from_name(path.name)
    if name_date and name_date < cutoff:
        return True

    # 2. Optional mtime fallback
    if USE_MTIME_FALLBACK:
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime).date()
            if mtime < cutoff:
                return True
        except OSError:
            pass

    return False


def delete_empty_dirs(root: Path):
    """Recursively remove empty directories under root."""
    for dirpath in sorted(root.rglob("*"), reverse=True):
        if dirpath.is_dir():
            try:
                dirpath.rmdir()
                print(f"  Removed empty dir: {dirpath.relative_to(root)}")
            except OSError:
                # not empty or permission issue → ignore
                pass


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    if not DETECTIONS_ROOT.exists():
        print(f"'{DETECTIONS_ROOT}' not found – nothing to purge.")
        return

    today = datetime.now().date()
    cutoff = today - timedelta(days=KEEP_DAYS)

    print(f"Purging files under '{DETECTIONS_ROOT}' older than {cutoff} (keep only {today})")

    files_to_delete = []
    for file_path in DETECTIONS_ROOT.rglob("*"):
        if not file_path.is_file():
            continue
        if file_is_old(file_path, cutoff):
            files_to_delete.append(file_path)

    if not files_to_delete:
        print("No old files found – everything is from today or newer.")
        # Show a few examples for debugging
        print("\nSample files (date extracted from name):")
        for p in sorted(DETECTIONS_ROOT.rglob("*"))[:10]:
            if p.is_file():
                d = parse_date_from_name(p.name)
                print(f"  - {p.relative_to(DETECTIONS_ROOT)} → {d}")
        return

    print(f"\nFound {len(files_to_delete)} file(s) to delete:")
    for p in sorted(files_to_delete):
        d = parse_date_from_name(p.name) or "???"
        print(f"  - {p.relative_to(DETECTIONS_ROOT)} (date: {d})")

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

    # ---- CLEAN EMPTY DIRS ----
    print("\nCleaning empty directories...")
    delete_empty_dirs(DETECTIONS_ROOT)

    # ---- GIT COMMIT ----
    print("\nCommitting changes to git...")
    try:
        run_cmd("git add detections")
        msg = f"purge: remove detections older than {cutoff} [{datetime.now().isoformat()}]"
        run_cmd(f'git commit -m "{msg}"')
        print("  Commit successful.")
    except Exception as e:
        print(f"  Git commit failed: {e}")

    print(f"\nPurge complete! {deleted} files removed.")


if __name__ == "__main__":
    main()
