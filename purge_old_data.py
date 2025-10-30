#!/usr/bin/env python3
"""
Standalone Purge Script for SOHO Comet Hunter
============================================
Purge all detection data (JSONs, crops, animations) older than today.
Commits changes to git automatically.

Usage:
  python purge_old_data.py

Requires: git (in PATH)
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import subprocess

# Config
DETECTIONS_DIR = Path("detections")
KEEP_DAYS = 0  # Purge everything before today (change to 7 for >7 days)
DRY_RUN = False  # Set to True to simulate without deleting/committing


def run_cmd(cmd, check=True):
    """Run shell command, raise on error if check=True."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{result.stderr}")
    return result


def main():
    if not DETECTIONS_DIR.exists():
        print("No 'detections/' directory found – nothing to purge.")
        return

    today = datetime.now().date()
    cutoff = today - timedelta(days=KEEP_DAYS)

    print(f"Purging files in '{DETECTIONS_DIR}' older than {cutoff} (before {today}).")

    old_files = []
    for f in DETECTIONS_DIR.rglob("*"):
        if f.is_file():
            try:
                mtime = datetime.fromtimestamp(f.stat().st_mtime).date()
                if mtime < cutoff:
                    old_files.append(f)
            except OSError:
                # Skip unreadable files
                continue

    if not old_files:
        print("No old files found – all data is current.")
        return

    print(f"Found {len(old_files)} file(s) to purge:")
    for f in sorted(old_files):
        print(f"  - {f.relative_to(DETECTIONS_DIR)}")

    if DRY_RUN:
        print("\n[DRY RUN] Would delete above files.")
        return

    # Delete files
    print("\nDeleting files...")
    for f in old_files:
        f.unlink()
        print(f"  Deleted: {f.name}")

    # Git commit
    print("\nCommitting changes...")
    os.chdir(DETECTIONS_DIR.parent)  # Ensure we're in repo root

    run_cmd("git add detections")
    commit_msg = f"Manual purge: remove data before {cutoff} [{datetime.now().isoformat()}]"
    run_cmd(f'git commit -m "{commit_msg}"')

    print(f"\nPurge complete! {len(old_files)} files removed and committed.")


if __name__ == "__main__":
    main()
