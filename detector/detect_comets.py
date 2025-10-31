#!/usr/bin/env python3
"""
detect_comets.py
----------------
Download the most recent LASCO C2/C3 images, run the comet detector
and always update detections/latest_status.json.
"""

import os
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import sys

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
HOURS = int(os.getenv("HOURS", "6"))          # look-back window
STEP_MIN = int(os.getenv("STEP_MIN", "12"))   # image cadence
OUT_DIR = Path(os.getenv("OUT", "detections"))
FRAMES_DIR = Path("frames")
DEBUG = os.getenv("DETECTOR_DEBUG", "0") == "1"

# **CORRECT** LASCO URL templates (no /1024/ folder)
C2_URL = (
    "https://soho.nascom.nasa.gov/data/REPROCESSING/Completed/"
    "{year}/c2/{date}/{date}_{time}_c2_1024.jpg"
)
C3_URL = (
    "https://soho.nascom.nasa.gov/data/REPROCESSING/Completed/"
    "{year}/c3/{date}/{date}_{time}_c3_1024.jpg"
)

# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------
def log(*msg):
    print(" ".join(map(str, msg)))

def run_cmd(cmd):
    """Run a command from the repo root – only used for debugging."""
    if DEBUG:
        log("RUN:", cmd)
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True,
        cwd=Path.cwd()
    )
    if result.returncode != 0 and DEBUG:
        log("CMD FAILED:", result.stderr)
    return result.stdout

# ----------------------------------------------------------------------
# DOWNLOAD FRAMES
# ----------------------------------------------------------------------
def download_frames():
    now = datetime.utcnow()
    start = now - timedelta(hours=HOURS)
    frames = {"C2": [], "C3": []}
    downloaded = {"C2": 0, "C3": 0}

    for instr in ("C2", "C3"):
        url_tmpl = C2_URL if instr == "C2" else C3_URL
        t = start.replace(minute=0, second=0, microsecond=0)

        while t < now:
            t += timedelta(minutes=STEP_MIN)
            if t > now:
                break

            date_str = t.strftime("%Y%m%d")
            time_str = t.strftime("%H%M")
            year = t.year

            url = url_tmpl.format(year=year, date=date_str, time=time_str)
            path = FRAMES_DIR / f"{instr}_{date_str}_{time_str}.jpg"

            if path.exists():
                frames[instr].append(str(path))
                downloaded[instr] += 1
                continue

            try:
                log(f"Fetching {instr} {date_str}_{time_str} …")
                resp = requests.get(url, timeout=12)
                if resp.status_code == 404:
                    log("  404 – image not yet published")
                    continue
                if resp.status_code != 200:
                    log(f"  HTTP {resp.status_code}")
                    continue

                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(resp.content)
                log(f"  Saved {path.name}")
                frames[instr].append(str(path))
                downloaded[instr] += 1
            except Exception as e:
                log(f"  Download error: {e}")

        log(f"{instr}: {downloaded[instr]} frame(s) downloaded")

    return frames, downloaded

# ----------------------------------------------------------------------
# UPDATE STATUS (ALWAYS)
# ----------------------------------------------------------------------
def write_status(c2_frames, c3_frames, tracks):
    status = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "detectors": {
            "C2": {"frames": c2_frames, "tracks": tracks},
            "C3": {"frames": c3_frames, "tracks": tracks}
        }
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    status_path = OUT_DIR / "latest_status.json"
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
    log(f"latest_status.json written → {status_path}")

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    log("=== DETECTION START ===")
    FRAMES_DIR.mkdir(exist_ok=True)

    # 1. Download frames
    frames, dl_counts = download_frames()
    c2_cnt = dl_counts["C2"]
    c3_cnt = dl_counts["C3"]

    # 2. Run detection (placeholder – replace with your AI code)
    # -------------------------------------------------------
    # For now we just count tracks = 0
    tracks = 0
    # -------------------------------------------------------

    # 3. Always write status (even when 0 frames)
    write_status(c2_cnt, c3_cnt, tracks)

    # 4. Debug tree output (mirrors your workflow log)
    log("=== SUMMARY ===")
    log(f"C2 frames: {c2_cnt}, C3 frames: {c3_cnt}")
    log(f"Candidates/Tracks: candidates, {tracks} tracks")
    log(f"JSON files: {len(list(OUT_DIR.glob('candidates_*.json')))}")
    log("frames tree:")
    run_cmd("find frames -type f | sort | sed 's/^/  - /'" if c2_cnt + c3_cnt else "echo '  - .gitkeep'")
    log("detections tree:")
    run_cmd("find detections -type f | head -20 | sort | sed 's/^/  - /'")

    log("=== END ===")

if __name__ == "__main__":
    main()
