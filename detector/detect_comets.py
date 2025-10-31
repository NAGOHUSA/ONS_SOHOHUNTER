#!/usr/bin/env python3
"""
detect_comets.py
----------------
Download LASCO C2/C3 frames, run a lightweight moving-point detector + AI classifier,
write:
  • detections/latest_status.json   (frames + time-range)
  • detections/latest_run.json     (proof the bot ran)
  • detections/candidates_*.json   (only if comets)
"""

import argparse
import os
import json
import requests
import cv2
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import sys

# ----------------------------------------------------------------------
# Add detector/ to Python path so ai_classifier can be imported
# ----------------------------------------------------------------------
sys.path.append(str(Path(__file__).parent))

# ----------------------------------------------------------------------
# CONFIG (CLI > env > defaults)
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--hours", type=int, default=int(os.getenv("HOURS", "6")))
parser.add_argument("--step-min", type=int, default=int(os.getenv("STEP_MIN", "12")))
parser.add_argument("--out", default=os.getenv("OUT", "detections"))
args = parser.parse_args()

HOURS = args.hours
STEP_MIN = args.step_min
OUT_DIR = Path(args.out)
FRAMES_DIR = Path("frames")
DEBUG = os.getenv("DETECTOR_DEBUG", "0") == "1"

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

# ----------------------------------------------------------------------
# DOWNLOAD FRAMES + TIMESTAMPS
# ----------------------------------------------------------------------
def download_frames():
    now = datetime.utcnow()
    start = now - timedelta(hours=HOURS)
    frames = {"C2": [], "C3": []}
    timestamps = {"C2": [], "C3": []}
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
            iso_time = t.strftime("%Y-%m-%dT%H:%M:00Z")

            url = url_tmpl.format(year=year, date=date_str, time=time_str)
            path = FRAMES_DIR / f"{instr}_{date_str}_{time_str}.jpg"

            if path.exists():
                frames[instr].append(str(path))
                timestamps[instr].append(iso_time)
                downloaded[instr] += 1
                continue

            try:
                log(f"Fetching {instr} {date_str}_{time_str} …")
                resp = requests.get(url, timeout=12)
                if resp.status_code == 404:
                    log("  404 – not yet published")
                    continue
                if resp.status_code != 200:
                    log(f"  HTTP {resp.status_code}")
                    continue

                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(resp.content)
                log(f"  Saved {path.name}")
                frames[instr].append(str(path))
                timestamps[instr].append(iso_time)
                downloaded[instr] += 1
            except Exception as e:
                log(f"  Download error: {e}")

        log(f"{instr}: {downloaded[instr]} frame(s) downloaded")

    return frames, downloaded, timestamps

# ----------------------------------------------------------------------
# AI CLASSIFIER IMPORT (with safe fallback)
# ----------------------------------------------------------------------
try:
    from ai_classifier import classify_crop_batch
    log("AI classifier import successful")
except Exception as e:
    log("AI classifier import failed – falling back to dummy:", e)
    def classify_crop_batch(paths):
        return [{"label": "not_comet", "score": 0.0} for _ in paths]

# ----------------------------------------------------------------------
# SIMPLE TRACK DETECTION + CLASSIFICATION
# ----------------------------------------------------------------------
from scipy.ndimage import gaussian_filter
from filterpy.kalman import KalmanFilter

def simple_track_detect(frame_paths, instr, ts_list):
    if len(frame_paths) < 4:
        return []

    # Sort chronologically
    paired = sorted(zip(frame_paths, ts_list), key=lambda x: x[1])
    candidates = []

    # Active tracks: (kf, last_pos, age, track_id)
    active_kfs = []
    next_id = 0

    for idx, (path, ts) in enumerate(paired):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Preprocess
        img_f = gaussian_filter(img.astype(float) / 255.0, sigma=1.0) * 255
        img_f = np.clip(img_f, 0, 255).astype(np.uint8)

        # Otsu threshold
        _, thresh = cv2.threshold(img_f, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Predict all active tracks
        for entry in active_kfs[:]:
            kf, last_pos, age, tid = entry
            kf.predict()
            pred = kf.x[:2, 0].astype(int)
            if age > 3:
                active_kfs.remove(entry)
                continue
            # update age
            i = active_kfs.index(entry)
            active_kfs[i] = (kf, pred, age + 1, tid)

        # Extract measurements
        meas = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] < 12:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            meas.append((cx, cy, cnt))

        # Nearest-neighbor association
        used = set()
        for kf_idx, (kf, pred, age, tid) in enumerate(active_kfs):
            best_dist = float("inf")
            best_meas = None
            for m_idx, (mx, my, cnt) in enumerate(meas):
                if m_idx in used:
                    continue
                d = (mx - pred[0]) ** 2 + (my - pred[1]) ** 2
                if d < best_dist:
                    best_dist = d
                    best_meas = (mx, my)
            if best_meas and best_dist < 80 ** 2:
                kf.update(np.array([[best_meas[0]], [best_meas[1]]]))
                i = active_kfs.index((kf, pred, age, tid))
                active_kfs[i] = (kf, best_meas, 0, tid)
                used.add(meas.index((best_meas[0], best_meas[1], cnt)))
                meas = [m for i, m in enumerate(meas) if i not in used]

        # Start new tracks for leftovers
        for mx, my, cnt in meas:
            kf = KalmanFilter(dim_x=4, dim_z=2)
            kf.x[:2, 0] = np.array([mx, my])
            kf.F = np.array([[1, 0, 1, 0],
                             [0, 1, 0, 1],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
            kf.H = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0]])
            kf.P *= 100
            kf.R *= 8
            kf.Q = np.eye(4) * 0.15
            active_kfs.append((kf, (mx, my), 0, next_id))
            next_id += 1

        # Evaluate mature tracks (after 4+ frames)
        for kf, pos, age, tid in active_kfs[:]:
            if age == 0 and len([p for p in paired[:idx+1] if p[1] <= ts]) >= 4:
                h, w = img_f.shape
                x, y = pos
                crop = img_f[max(0, y-32):min(h, y+32), max(0, x-32):min(w, x+32)]
                if crop.size == 0:
                    continue
                tmp_path = Path(tempfile.mktemp(suffix=".png"))
                cv2.imwrite(str(tmp_path), crop)
                cls = classify_crop_batch([str(tmp_path)])[0]
                os.unlink(tmp_path)

                if cls["score"] >= 0.6:
                    candidates.append({
                        "instrument": instr,
                        "timestamp": ts,
                        "track_id": tid,
                        "bbox": [x-32, y-32, 64, 64],
                        "crop_path": f"crops/{instr}_track{tid}_{ts.replace(':', '')}.png",
                        "ai_label": cls["label"],
                        "ai_score": round(cls["score"], 3)
                    })
                    # Save final crop
                    crop_dir = OUT_DIR / "crops"
                    crop_dir.mkdir(parents=True, exist_ok=True)
                    final_crop = crop_dir / Path(candidates[-1]["crop_path"]).name
                    cv2.imwrite(str(final_crop), crop)
                    candidates[-1]["crop_path"] = str(final_crop.relative_to(OUT_DIR))

    return candidates

# ----------------------------------------------------------------------
# WRITE STATUS & RUN PROOF
# ----------------------------------------------------------------------
def write_status(c2_frames, c3_frames, tracks, c2_times, c3_times):
    total_analyzed = c2_frames + c3_frames
    def time_range(times):
        if not times:
            return "none"
        return f"{min(times).split('T')[1][:5]}–{max(times).split('T')[1][:5]} UTC"
    c2_range = time_range(c2_times)
    c3_range = time_range(c3_times)

    status = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "summary": f"Analyzed {c2_frames} C2 ({c2_range}) + {c3_frames} C3 ({c3_range}) = {total_analyzed} frames",
        "detectors": {
            "C2": {"frames": c2_frames, "tracks": tracks, "time_range": c2_range, "timestamps": c2_times},
            "C3": {"frames": c3_frames, "tracks": tracks, "time_range": c3_range, "timestamps": c3_times}
        },
        "total": {"frames_analyzed": total_analyzed, "tracks_found": tracks}
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "latest_status.json", "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
    log(f"latest_status.json → {OUT_DIR / 'latest_status.json'}")

def write_latest_run(c2_frames, c3_frames, candidates_count):
    run_info = {
        "last_run_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "status": "completed",
        "frames_downloaded": {"C2": c2_frames, "C3": c3_frames},
        "candidates_found": candidates_count,
        "note": "No new frames" if c2_frames + c3_frames == 0 else "OK"
    }
    with open(OUT_DIR / "latest_run.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)
    log(f"latest_run.json → {OUT_DIR / 'latest_run.json'}")

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    log("=== DETECTION START ===")
    FRAMES_DIR.mkdir(exist_ok=True)

    frames, dl_counts, timestamps = download_frames()
    c2_cnt = dl_counts["C2"]
    c3_cnt = dl_counts["C3"]
    c2_times = timestamps["C2"]
    c3_times = timestamps["C3"]

    c2_cands = simple_track_detect(frames["C2"], "C2", c2_times)
    c3_cands = simple_track_detect(frames["C3"], "C3", c3_times)
    all_cands = c2_cands + c3_cands
    tracks = len({c["track_id"] for c in all_cands})

    if all_cands:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_file = OUT_DIR / f"candidates_{timestamp}.json"
        with open(out_file, "w") as f:
            json.dump(all_cands, f, indent=2)
        log(f"Saved {len(all_cands)} candidates → {out_file.name}")

    write_status(c2_cnt, c3_cnt, tracks, c2_times, c3_times)
    write_latest_run(c2_cnt, c3_cnt, len(all_cands))

    log("=== SUMMARY ===")
    log(f"C2 frames: {c2_cnt}, C3 frames: {c3_cnt}")
    log(f"Candidates: {len(all_cands)}, Unique tracks: {tracks}")
    log("=== END ===")

if __name__ == "__main__":
    main()
