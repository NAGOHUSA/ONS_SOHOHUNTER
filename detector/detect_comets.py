#!/usr/bin/env python3
"""
Multi-source comet hunter: LASCO + GOES-19 (CCOR-1)
Organized frames, robust tracking, AI classifier.
"""

import argparse
import os
import json
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
import tempfile
import sys

# Add detector/ to path
sys.path.append(str(Path(__file__).parent))

# Import sources
from sources.lasco import fetch_lasco_frames
from sources.goes import fetch_goes_frames
from sources.stereo import fetch_stereo_frames
from sources.solar_orbiter import fetch_solar_orbiter_frames

# AI Classifier
try:
    from ai_classifier import classify_crop_batch
    print("AI classifier loaded")
except Exception as e:
    print("AI fallback:", e)
    def classify_crop_batch(paths):
        return [{"label": "not_comet", "score": 0.0} for _ in paths]

# Tracker
from scipy.ndimage import gaussian_filter
from filterpy.kalman import KalmanFilter

def simple_track_detect(frame_paths, instr, ts_list):
    if len(frame_paths) < 4:
        return []

    paired = sorted(zip(frame_paths, ts_list), key=lambda x: x[1])
    candidates = []
    active_kfs = []
    next_id = 0

    for idx, (path, ts) in enumerate(paired):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img_f = gaussian_filter(img.astype(float) / 255.0, sigma=1.0) * 255
        img_f = np.clip(img_f, 0, 255).astype(np.uint8)

        _, thresh = cv2.threshold(img_f, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Predict
        for entry in active_kfs[:]:
            kf, last_pos, age, tid = entry
            kf.predict()
            pred = kf.x[:2, 0].astype(int)
            if age > 3:
                active_kfs.remove(entry)
                continue
            i = active_kfs.index(entry)
            active_kfs[i] = (kf, pred, age + 1, tid)

        # Measure
        meas = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] < 12:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            meas.append((cx, cy, cnt))

        # Associate
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

        # New tracks
        for mx, my, cnt in meas:
            kf = KalmanFilter(dim_x=4, dim_z=2)
            kf.x[:2, 0] = np.array([mx, my])
            kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
            kf.H = np.array([[1,0,0,0],[0,1,0,0]])
            kf.P *= 100
            kf.R *= 8
            kf.Q = np.eye(4) * 0.15
            active_kfs.append((kf, (mx, my), 0, next_id))
            next_id += 1

        # Evaluate
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
                    crop_dir = OUT_DIR / "crops"
                    crop_dir.mkdir(parents=True, exist_ok=True)
                    final_crop = crop_dir / f"{instr}_track{tid}_{ts.replace(':', '')}.png"
                    cv2.imwrite(str(final_crop), crop)
                    candidates.append({
                        "instrument": instr,
                        "timestamp": ts,
                        "track_id": tid,
                        "bbox": [x-32, y-32, 64, 64],
                        "crop_path": str(final_crop.relative_to(OUT_DIR)),
                        "ai_label": cls["label"],
                        "ai_score": round(cls["score"], 3)
                    })

    return candidates

# Status & Run
def write_status(frames_dict, timestamps_dict):
    total = sum(len(frames_dict.get(i, [])) for i in INSTRS)
    status = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "instruments": INSTRS,
        "summary": f"Analyzed {total} frames",
        "per_source": {
            i: {
                "frames": len(frames_dict.get(i, [])),
                "time_range": f"{min(timestamps_dict[i]).split('T')[1][:5]}–{max(timestamps_dict[i]).split('T')[1][:5]} UTC" if timestamps_dict[i] else "none"
            } for i in INSTRS
        }
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "latest_status.json", "w") as f:
        json.dump(status, f, indent=2)

def write_latest_run(downloaded, candidates_count):
    with open(OUT_DIR / "latest_run.json", "w") as f:
        json.dump({
            "last_run_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "frames_downloaded": downloaded,
            "candidates_found": candidates_count
        }, f, indent=2)

# Main
parser = argparse.ArgumentParser()
parser.add_argument("--hours", type=int, default=int(os.getenv("HOURS", "6")))
parser.add_argument("--step-min", type=int, default=int(os.getenv("STEP_MIN", "12")))
parser.add_argument("--out", default=os.getenv("OUT", "detections"))
parser.add_argument("--instruments", default=os.getenv("INSTRUMENTS", "lasco"))
args = parser.parse_args()

HOURS = args.hours
STEP_MIN = args.step_min
OUT_DIR = Path(args.out)
INSTRS = [i.strip().lower() for i in args.instruments.split(",")]

def main():
    print("=== DETECTION START ===")
    frames = {}
    timestamps = {}
    downloaded = {}

    for instr in INSTRS:
        print(f"Fetching {instr.upper()}...")
        if instr == "lasco":
            f, t, d = fetch_lasco_frames(HOURS, STEP_MIN)
        elif instr == "goes":
            f, t, d = fetch_goes_frames(HOURS, STEP_MIN)
        else:
            continue
        frames[instr] = f
        timestamps[instr] = t
        downloaded[instr] = d

    all_cands = []
    for instr in INSTRS:
        if instr in frames and len(frames[instr]) >= 4:
            cands = simple_track_detect(frames[instr], instr, timestamps[instr])
            all_cands.extend(cands)

    if all_cands:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        with open(OUT_DIR / f"candidates_{ts}.json", "w") as f:
            json.dump(all_cands, f, indent=2)

    write_status(frames, timestamps)
    write_latest_run(downloaded, len(all_cands))

    print("=== SUMMARY ===")
    for i in INSTRS:
        print(f"{i}: {downloaded.get(i, 0)} frames")
    print(f"Candidates: {len(all_cands)}")
    print("=== END ===")

if __name__ == "__main__":
    main()
