#!/usr/bin/env python3
"""
purge_old_data.py
-----------------
Purge all files in detections/ older than today based on filename date.
Works on Python 3.8+ (GitHub Actions safe).
"""

import os
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional  # <-- Import Optional


# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
DETECTIONS_ROOT = Path("detections")
KEEP_DAYS = 0
DRY_RUN = False
USE_MTIME_FALLBACK = False


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------
def run_cmd(cmd: str, check: bool = True):
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, cwd=DETECTIONS_ROOT.parent
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{result.stderr}")
    return result


def parse_date_from_name(name: str) -> Optional[datetime.date]:
    """
    Extract YYYYMMDD from filename → return date or None.
    Example: candidates_20251029_232405.json → 2025-10-29
    """
    m = re.search(r"(\d{8})", name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d").date()
    except ValueError:
        return None


def file_is_old(path: Path, cutoff: datetime.date) -> bool:
    name_date = parse_date_from_name(path.name)
    if name_date and name_date < cutoff:
        return True

    if USE_MTIME_FALLBACK:
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime).date()
            if mtime < cutoff:
                return True
        except OSError:
            pass
    return False


def delete_empty_dirs(root: Path):
    for dirpath in sorted(root.rglob("*"), reverse=True):
        if dirpath.is_dir():
            try:
                dirpath.rmdir()
                print(f"  Removed empty dir: {dirpath.relative_to(root)}")
            except OSError:
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

    print(f"Purging files under '{DETECTIONS_ROOT}' older than {cutoff}")

    files_to_delete = []
    for file_path in DETECTIONS_ROOT.rglob("*"):
        if not file_path.is_file():
            continue
        if file_is_old(file_path, cutoff):
            files_to_delete.append(file_path)

    if not files_to_delete:
        print("No old files found.")
        print("\nSample files (date from name):")
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
        print("\n[DRY RUN] – no changes made.")
        return

    # Delete
    print("\nDeleting files...")
    deleted = 0
    for p in files_to_delete:
        try:
            p.unlink()
            print(f"  Deleted: {p.name}")
            deleted += 1
        except OSError as e:
            print(f"  Failed: {p.name} → {e}")

    # Clean dirs
    print("\nCleaning empty directories...")
    delete_empty_dirs(DETECTIONS_ROOT)

    # Git
    print("\nCommitting to git...")
    try:
        run_cmd("git add detections")
        msg = f"purge: remove data before {cutoff} [{datetime.now().isoformat()}]"
        run_cmd(f'git commit -m "{msg}"')
        print("  Commit OK.")
    except Exception as e:
        print(f"  Git failed: {e}")

    print(f"\nPurge complete! {deleted} files removed.")


if __name__ == "__main__":
    main()
