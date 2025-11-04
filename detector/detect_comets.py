#!/usr/bin/env python3
"""
SOHO Comet Detection Pipeline
- Downloads recent LASCO C2/C3 frames
- Lightweight moving-point tracker → crops
- Local AI classifier (ai_classifier.py)
- Optional Groq vision (≤2 calls / 12 h) on the highest-score crops
- Writes:
    detections/candidates_YYYYMMDD_HHMMSS.json
    detections/latest_status.json
    detections/latest_run.json
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
import base64
from groq import Groq

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
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LAST_CALL_FILE = OUT_DIR / "last_groq_call.txt"
GROQ_LIMIT_HOURS = 12
GROQ_MAX_CALLS = 2

# ----------------------------------------------------------------------
# LASCO URL TEMPLATES
# ----------------------------------------------------------------------
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
# 1. DOWNLOAD FRAMES + timestamps
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
                resp = requests.get(url, timeout=12)
                if resp.status_code != 200:
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(resp.content)
                frames[instr].append(str(path))
                timestamps[instr].append(iso_time)
                downloaded[instr] += 1
            except Exception as e:
                log("Download error:", e)

    return frames, downloaded, timestamps

# ----------------------------------------------------------------------
# 2. LOCAL AI CLASSIFIER
# ----------------------------------------------------------------------
try:
    from ai_classifier import classify_crop_batch
except Exception as e:
    log("ai_classifier import failed – using dummy:", e)
    def classify_crop_batch(paths):
        return [{"label": "not_comet", "score": 0.0} for _ in paths]

# ----------------------------------------------------------------------
# 3. SIMPLE TRACKER (Kalman + Otsu)
# ----------------------------------------------------------------------
from scipy.ndimage import gaussian_filter
from filterpy.kalman import KalmanFilter

def simple_track_detect(frame_paths, instr, ts_list):
    if len(frame_paths) < 4:
        return []

    paired = sorted(zip(frame_paths, ts_list), key=lambda x: x[1])
    candidates = []
    active_kfs = []          # list of **tuples**: (kf, last_pos, age, tid)
    next_id = 0

    for idx, (path, ts) in enumerate(paired):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # light preprocessing
        img_f = gaussian_filter(img.astype(float) / 255.0, sigma=1.0) * 255
        img_f = np.clip(img_f, 0, 255).astype(np.uint8)

        # Otsu threshold → contours
        _, thresh = cv2.threshold(img_f, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # ---- predict existing tracks ----
        new_active = []
        for kf, last_pos, age, tid in active_kfs:
            kf.predict()
            pred = kf.x[:2, 0].astype(int)
            if age > 3:
                continue                     # drop stale tracks
            new_active.append((kf, pred, age + 1, tid))
        active_kfs = new_active

        # ---- measurements ----
        meas = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] > 10:
                mx = int(M["m10"] / M["m00"])
                my = int(M["m01"] / M["m00"])
                meas.append((mx, my))

        # ---- associate (nearest, max 80 px) ----
        used = set()
        for i, (kf, pred, age, tid) in enumerate(active_kfs):
            best_dist = float("inf")
            best_meas = None
            best_idx = -1
            for m_idx, (mx, my) in enumerate(meas):
                if m_idx in used:
                    continue
                d = (mx - pred[0]) ** 2 + (my - pred[1]) ** 2
                if d < best_dist:
                    best_dist = d
                    best_meas = (mx, my)
                    best_idx = m_idx
            if best_meas and best_dist < 80 ** 2:
                kf.update(np.array(best_meas).reshape(2, 1))
                active_kfs[i] = (kf, best_meas, 0, tid)
                used.add(best_idx)

        # ---- spawn new tracks ----
        for mx, my in meas:
            if any((mx, my) == p for _, p, _, _ in active_kfs):
                continue
            kf = KalmanFilter(dim_x=4, dim_z=2)
            kf.x[:2] = np.array([mx, my]).reshape(2, 1)
            kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
            kf.H = np.array([[1,0,0,0],[0,1,0,0]])
            kf.P *= 100; kf.R *= 8; kf.Q = np.eye(4) * 0.15
            active_kfs.append((kf, (mx, my), 0, next_id))
            next_id += 1

        # ---- evaluate mature tracks (≥4 frames) ----
        for kf, pos, age, tid in active_kfs:
            if age == 0 and len([p for p in paired[:idx+1] if p[1] <= ts]) >= 4:
                h, w = img_f.shape
                x, y = pos
                crop = img_f[max(0, y-32):min(h, y+32), max(0, x-32):min(w, x+32)]
                if crop.size == 0:
                    continue

                tmp = Path(tempfile.mktemp(suffix=".png"))
                cv2.imwrite(str(tmp), crop)
                cls = classify_crop_batch([str(tmp)])[0]
                os.unlink(tmp)

                if cls["score"] >= 0.5:          # comet confidence threshold
                    crop_dir = OUT_DIR / "crops"
                    crop_dir.mkdir(parents=True, exist_ok=True)
                    crop_name = f"{instr}_track{tid}_{ts.replace(':', '')}.png"
                    final_path = crop_dir / crop_name
                    cv2.imwrite(str(final_path), crop)

                    candidates.append({
                        "instrument": f"LASCO {instr}",
                        "timestamp": ts,
                        "track_id": tid,
                        "bbox": [x-32, y-32, 64, 64],
                        "crop_path": f"crops/{crop_name}",
                        "ai_label": cls["label"],
                        "ai_score": round(cls["score"], 3),
                        "pos": {"x": x, "y": y}
                    })

    return candidates

# ----------------------------------------------------------------------
# 4. GROQ VISION (max 2 calls / 12 h)
# ----------------------------------------------------------------------
def advanced_groq_classify(candidates):
    if not GROQ_API_KEY:
        log("No GROQ_API_KEY – skipping Groq")
        return candidates

    # rate-limit check
    if LAST_CALL_FILE.exists():
        last = datetime.fromtimestamp(float(LAST_CALL_FILE.read_text().strip()))
        if datetime.utcnow() - last < timedelta(hours=GROQ_LIMIT_HOURS):
            log("Groq rate-limit active – skipping")
            return candidates
    else:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    # pick up to 2 highest-score crops
    high = sorted(
        [c for c in candidates if c["ai_score"] > 0.5],
        key=lambda x: x["ai_score"],
        reverse=True
    )[:GROQ_MAX_CALLS]

    if not high:
        log("No high-score crops for Groq")
        return candidates

    client = Groq(api_key=GROQ_API_KEY)
    for c in high:
        path = OUT_DIR / c["crop_path"]
        if not path.exists():
            continue
        b64 = base64.b64encode(path.read_bytes()).decode()

        try:
            resp = client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": "Is this SOHO LASCO crop a comet? Answer with:\n"
                                 "label: comet or not_comet\n"
                                 "score: 0-1 (confidence)\n"
                                 "description: one short sentence."},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{b64}"}}
                    ]
                }],
                model="llama-3.2-90b-vision-preview"
            )
            txt = resp.choices[0].message.content.lower()
            label = "comet" if "comet" in txt else "not_comet"
            score = 0.0
            desc = "N/A"
            if "score:" in txt:
                try:
                    score = float(txt.split("score:")[1].split()[0])
                except:
                    pass
            if "description:" in txt:
                desc = txt.split("description:")[1].strip()

            c["groq_label"] = label
            c["groq_score"] = score
            c["groq_description"] = desc
            log(f"Groq → {c['crop_path']}: {label} ({score:.2f}) – {desc}")
        except Exception as e:
            log("Groq error:", e)

    # record call time
    LAST_CALL_FILE.write_text(str(datetime.utcnow().timestamp()))
    return candidates

# ----------------------------------------------------------------------
# 5. STATUS / RUN LOG (always written)
# ----------------------------------------------------------------------
def write_status(c2_cnt, c3_cnt, tracks, c2_times, c3_times):
    status = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "frames": {"C2": c2_cnt, "C3": c3_cnt},
        "tracks": tracks,
        "time_range": {
            "C2": f"{min(c2_times).split('T')[1][:5]}–{max(c2_times).split('T')[1][:5]} UTC" if c2_times else "none",
            "C3": f"{min(c3_times).split('T')[1][:5]}–{max(c3_times).split('T')[1][:5]} UTC" if c3_times else "none"
        }
    }
    (OUT_DIR / "latest_status.json").write_text(json.dumps(status, indent=2))

def write_run_log(c2_cnt, c3_cnt, cand_cnt):
    log = {
        "last_run_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "frames_downloaded": {"C2": c2_cnt, "C3": c3_cnt},
        "candidates_found": cand_cnt
    }
    (OUT_DIR / "latest_run.json").write_text(json.dumps(log, indent=2))

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    log("=== DETECTION START ===")
    FRAMES_DIR.mkdir(exist_ok=True)

    # 1. download
    frames, dl, ts = download_frames()
    c2_cnt = dl["C2"]
    c3_cnt = dl["C3"]

    # 2. track & classify
    c2_cands = simple_track_detect(frames["C2"], "C2", ts["C2"])
    c3_cands = simple_track_detect(frames["C3"], "C3", ts["C3"])
    all_cands = c2_cands + c3_cands

    # 3. optional Groq
    all_cands = advanced_groq_classify(all_cands)

    # 4. write candidates (if any)
    if all_cands:
        ts_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        (OUT_DIR / f"candidates_{ts_str}.json").write_text(json.dumps(all_cands, indent=2))
        log(f"Saved {len(all_cands)} candidates → candidates_{ts_str}.json")

    # 5. always write status / run log
    unique_tracks = len({c["track_id"] for c in all_cands})
    write_status(c2_cnt, c3_cnt, unique_tracks, ts["C2"], ts["C3"])
    write_run_log(c2_cnt, c3_cnt, len(all_cands))

    log(f"=== DONE ===  C2:{c2_cnt}  C3:{c3_cnt}  Candidates:{len(all_cands)}  Tracks:{unique_tracks}")

if __name__ == "__main__":
    main()
