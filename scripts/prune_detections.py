#!/usr/bin/env python3
"""
prune_detections.py
-------------------
Keep the detections/ folder from growing without limit.

Deletes detection files whose filename date (YYYYMMDD) is older than
KEEP_DAYS days, then rebuilds detections/reports.json so the site only
lists reports that still exist.

Uses the date embedded in each filename rather than file modification
times, because every file gets a fresh mtime on each checkout in CI.

Covers:
  detections/candidates_YYYYMMDD_HHMMSS.json
  detections/{mp4,gifs,crops,annotated,originals}/C?_YYYYMMDD_*

Never deletes candidates_latest.json, latest_status.json or reports.json.

Usage:
  python scripts/prune_detections.py                # keep 30 days
  python scripts/prune_detections.py --keep-days 45
  python scripts/prune_detections.py --dry-run
"""

import argparse
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

DETECTIONS = Path("detections")
MEDIA_DIRS = ["mp4", "gifs", "crops", "annotated", "originals"]
PROTECTED = {"candidates_latest.json", "latest_status.json", "reports.json"}
DATE_RE = re.compile(r"_(20\d{6})(?:_|$)")


def file_date(name: str):
    m = DATE_RE.search(Path(name).stem)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d").date()
    except ValueError:
        return None


def rebuild_reports_json():
    reports = sorted(p.name for p in DETECTIONS.glob("candidates_*.json")
                     if p.name != "candidates_latest.json")
    if not reports:
        return
    # Same shape the detect workflow writes: newest report name + full list
    latest = max(reports)
    (DETECTIONS / "reports.json").write_text(
        json.dumps({"latest": latest, "reports": reports}, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-days", type=int,
                    default=int(os.environ.get("KEEP_DAYS", "30")))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not DETECTIONS.exists():
        print("detections/ not found; nothing to prune.")
        return

    cutoff = (datetime.now(timezone.utc) - timedelta(days=args.keep_days)).date()
    print(f"Pruning detection files dated before {cutoff} (keeping {args.keep_days} days)")

    targets = [p for p in DETECTIONS.glob("candidates_*.json") if p.name not in PROTECTED]
    for d in MEDIA_DIRS:
        sub = DETECTIONS / d
        if sub.is_dir():
            targets.extend(p for p in sub.iterdir() if p.is_file())

    removed, freed = 0, 0
    for p in targets:
        d = file_date(p.name)
        if d is None or d >= cutoff:
            continue
        size = p.stat().st_size
        if not args.dry_run:
            p.unlink()
        removed += 1
        freed += size

    verb = "Would remove" if args.dry_run else "Removed"
    print(f"{verb} {removed} files ({freed / 1e6:.1f} MB)")

    if not args.dry_run and removed:
        rebuild_reports_json()

    remaining = sum(f.stat().st_size for f in DETECTIONS.rglob("*") if f.is_file())
    print(f"detections/ is now {remaining / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
