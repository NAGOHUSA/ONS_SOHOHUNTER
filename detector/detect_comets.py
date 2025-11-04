#!/usr/bin/env python3
"""
SOHO Comet Detection Pipeline
- Fetches LASCO frames
- Masks timestamp + central occulter
- Detects comets only
- Tracks, crops, annotates
- Saves JSON + images + **animated GIFs** (FIXED: uniform size)
"""

from datetime import datetime
from pathlib import Path
import json
import cv2
import numpy as np
import os
import sys
from collections import defaultdict

# -------------------------------------------------
# GIF generation helper (uses imageio)
# -------------------------------------------------
try:
    import imageio
    GIF_AVAILABLE = True
except Exception:  # pragma: no cover
    GIF_AVAILABLE = False
    print("Warning: imageio not installed – GIF generation disabled. Run: pip install imageio[ffmpeg]")

# -------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

try:
    from ai_classifier import classify_crop_batch
except ImportError:
    print("Warning: ai_classifier not found, using dummy")
    def classify_crop_batch(paths):
        return [{"label": "unknown", "score": 0.0} for _ in paths]

# ==================== CONFIG ====================
FRAMES_DIR = Path("frames")
DETECTIONS_DIR = Path("detections")
CROPS_DIR = DETECTIONS_DIR / "crops"
ORIGINALS_DIR = DETECTIONS_DIR / "originals"
ANNOTATED_DIR = DETECTIONS_DIR / "annotated"
GIF_DIR = DETECTIONS_DIR / "gifs"
GIF_DIR.mkdir(parents=True, exist_ok=True)

USE_AI = os.getenv("USE_AI_CLASSIFIER", "1") == "1"
AI_VETO = os.getenv("AI_VETO_ENABLED", "1") == "1"
AI_VETO_LABEL = os.getenv("AI_VETO_LABEL", "not_comet")
AI_VETO_MAX = float(os.getenv("AI_VETO_SCORE_MAX", "0.9"))
MAX_ASSOC_DIST = 100

# ==================== MASKS ====================
def build_timestamp_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), np.uint8)
    cw = max(120, w // 8)
    ch = max(80, h // 8)
    edge = max(28, min(h, w) // 36)
    mask[0:ch, 0:cw] = 255
    mask[0:ch, w-cw:w] = 255
    mask[h-ch:h, 0:cw] = 255
    mask[h-ch:h, w-cw:w] = 255
    mask[0:edge, :] = 255
    mask[h-edge:h, :] = 255
    mask[:, 0:edge] = 255
    mask[:, w-edge:w] = 255
    return mask

def mask_central_occulter(frame: np.ndarray, buffer: int = 20) -> np.ndarray:
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    radius = int(min(w, h) * 0.18)
    y, x = np.ogrid[:h, :w]
    mask = (x - cx)**2 + (y - cy)**2 <= (radius + buffer)**2
    masked = frame.copy()
    masked[mask] = 0
    return masked

# ==================== FRAME FETCHING ====================
def fetch_lasco_frames(hours=6, step_min=12):
    print(f"Fetching LASCO frames (last {hours}h, step {step_min}min)...")
    try:
        from fetch_lasco import fetch_window
        frames = fetch_window(hours_back=hours, step_min=step_min, root=str(FRAMES_DIR))
        timestamps = []
        for p in frames:
            try:
                stem = Path(p).stem.split('_')
                if len(stem) >= 3:
                    ts = datetime.strptime(f"{stem[1]}{stem[2]}", "%Y%m%d%H%M")
                    timestamps.append(ts.strftime("%Y-%m-%dT%H:%M:%SZ"))
                else:
                    timestamps.append(datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))
            except:
                timestamps.append(datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))
        print(f"Fetched {len(frames)} frames")
        return frames, timestamps, len(frames)
    except ImportError:
        print("Using local frames")
        frames = []
        for det in ["C2", "C3"]:
            d = FRAMES_DIR / det
            if d.exists():
                frames.extend([str(f) for f in sorted(d.glob("*.jpg")) + sorted(d.glob("*.png"))])
        timestamps = [datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")] * len(frames)
        return frames, timestamps, 0

# ==================== DETECTION ====================
def detect_candidates(frame_paths, timestamps):
    print(f"Detecting on {len(frame_paths)} frames...")
    candidates = []

    for i, fp in enumerate(frame_paths):
        try:
            frame = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            if frame is None: continue
            h, w = frame.shape

            instr = "LASCO C2" if "c2" in Path(fp).name.lower() else "LASCO C3"

            frame_masked = frame.copy()
            ts_mask = build_timestamp_mask(h, w)
            frame_masked[ts_mask == 255] = 0
            frame_masked = mask_central_occulter(frame_masked, buffer=25)

            _, thresh = cv2.threshold(frame_masked, 200, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3,3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                x, y, bw, bh = cv2.boundingRect(cnt)
                if bw < 8 or bh < 8 or bw > 220 or bh > 220: continue
                if bbox_intersects_mask((x,y,bw,bh), ts_mask, 10): continue

                cx = x + bw // 2
                cy = y + bh // 2

                candidates.append({
                    "instrument": instr,
                    "timestamp": timestamps[i] if i < len(timestamps) else "",
                    "bbox": [x, y, bw, bh],
                    "centroid": [cx, cy],
                    "frame_path": fp,
                    "image_size": [h, w],
                    "_ts_mask": ts_mask,
                    "_frame_gray": frame
                })
        except Exception as e:
            print(f"Error {fp}: {e}")
            continue

    print(f"Found {len(candidates)} candidates")
    return candidates

# ==================== FILTERING ====================
def bbox_intersects_mask(bbox, mask: np.ndarray, min_overlap_px: int = 25) -> bool:
    if len(bbox) < 4: return False
    x, y, w, h = [int(v) for v in bbox[:4]]
    x2, y2 = x + w, y + h
    x, y = max(0, x), max(0, y)
    x2, y2 = min(mask.shape[1], x2), min(mask.shape[0], y2)
    if x >= x2 or y >= y2: return False
    return int(np.count_nonzero(mask[y:y2, x:x2])) >= min_overlap_px

def is_timestamp_artifact(bbox, fw, fh):
    if len(bbox) < 4: return False
    x, y, w, h = bbox[:4]
    cm = max(100, min(fw, fh)//9)
    em = max(50, min(fw, fh)//22)
    in_corner = ((x < cm and y < cm) or (x + w > fw - cm and y < cm) or
                 (x < cm and y + h > fh - cm) or (x + w > fw - cm and y + h > fh - cm))
    on_edge = (y < em or y + h > fh - em or x < em or x + w > fw - em)
    aspect = w / h if h > 0 else 0
    text_shaped = (2.0 < aspect < 10.0 and w < 320 and h < 80)
    tiny = (w < 30 or h < 30) and in_corner
    return (in_corner and (text_shaped or tiny)) or (on_edge and text_shaped and (w < 260 or h < 48))

def filter_detections_spatial(candidates, fw, fh, mask, gray):
    filtered, removed = [], 0
    for c in candidates:
        bbox = c.get("bbox", [])
        if not bbox or len(bbox) < 4: continue
        if bbox_intersects_mask(bbox, mask): removed += 1; continue
        if is_timestamp_artifact(bbox, fw, fh): removed += 1; continue
        x, y, w, h = [int(v) for v in bbox[:4]]
        x2, y2 = x + w, y + h
        x, y = max(0, x), max(0, y)
        x2, y2 = min(fw, x2), min(fh, y2)
        roi = gray[y:y2, x:x2]
        if roi.size == 0: continue
        bright_ratio = np.count_nonzero(roi > 220) / roi.size
        if bright_ratio > 0.35: removed += 1; continue
        filtered.append(c)
    if removed: print(f"  Removed {removed} artifacts")
    return filtered

def apply_spatial_filter(candidates):
    filtered = []
    for c in candidates:
        w, h = c["image_size"][1], c["image_size"][0]
        kept = filter_detections_spatial([c], w, h, c["_ts_mask"], c["_frame_gray"])
        if kept: filtered.append(kept[0])
    return filtered

# ==================== TRACKING ====================
def associate_tracks(candidates):
    print("Tracking...")
    time_groups = defaultdict(list)
    for c in candidates:
        time_groups[c['timestamp']].append(c)

    sorted_times = sorted(time_groups.keys())
    active_tracks = []
    track_id = 0

    for t in sorted_times:
        current = time_groups[t]
        new_active = []
        assigned = set()
        for det in current:
            best, min_d = None, float('inf')
            for j, tr in enumerate(active_tracks):
                if j in assigned: continue
                dx = det['bbox'][0] - tr[-1]['bbox'][0]
                dy = det['bbox'][1] - tr[-1]['bbox'][1]
                d = np.sqrt(dx**2 + dy**2)
                if d < min_d and d < MAX_ASSOC_DIST:
                    min_d, best = d, j
            if best is not None:
                active_tracks[best].append(det)
                new_active.append(active_tracks[best])
                assigned.add(best)
            else:
                new_active.append([det])
        for j, tr in enumerate(active_tracks):
            if j not in assigned:
                new_active.append(tr)
        active_tracks = new_active

    for track in active_tracks:
        positions = []
        for det in track:
            det['track_id'] = track_id
            positions.append({"x": det['centroid'][0], "y": det['centroid'][1]})
        for det in track:
            det['positions'] = positions
        track_id += 1

    print(f"Associated {len(active_tracks)} tracks")
    return candidates

# ==================== SAVE ASSETS ====================
def save_frame_assets(candidates):
    print("Saving originals & annotated...")
    ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for c in candidates:
        groups[c['frame_path']].append(c)

    saved = 0
    for fp, dets in groups.items():
        try:
            frame = cv2.imread(fp)
            if frame is None: continue
            stem = Path(fp).stem
            instr = dets[0]['instrument'].replace(' ', '_').lower()
            orig_fn = f"lasco_{instr}_{stem}.jpg"
            orig_path = ORIGINALS_DIR / orig_fn
            cv2.imwrite(str(orig_path), frame)

            ann = frame.copy()
            for d in dets:
                x, y, w, h = d['bbox']
                cv2.rectangle(ann, (x, y), (x+w, y+h), (0,255,0), 2)

            ann_fn = f"lasco_{instr}_{stem}_annotated.jpg"
            ann_path = ANNOTATED_DIR / ann_fn
            cv2.imwrite(str(ann_path), ann)

            for d in dets:
                d['original_path'] = f"originals/{orig_fn}"
                d['annotated_path'] = f"annotated/{ann_fn}"
            saved += 1
        except Exception as e:
            print(f"Save error {fp}: {e}")
    print(f"Saved {saved} frames")

# ==================== CROP + GIF (FIXED SIZE) ====================
def pad_to_size(img, target_h, target_w):
    """Pad image to target size with black borders (center-aligned)."""
    h, w = img.shape[:2]
    pad_h = target_h - h
    pad_w = target_w - w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

def extract_crops_and_gifs(candidates):
    print(f"Extracting crops + GIFs for {len(candidates)} detections...")
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    tracks = defaultdict(list)
    for c in candidates:
        tracks[c['track_id']].append(c)

    extracted = []
    gif_count = 0

    for tid, seq in tracks.items():
        seq = sorted(seq, key=lambda x: x['timestamp'])
        crop_frames = []
        ann_frames = []

        # Determine max crop size in track
        max_w = max_h = 0
        for c in seq:
            bbox = c.get("bbox", [])
            if len(bbox) < 4: continue
            x, y, w, h = [int(v) for v in bbox]
            pad = 12
            crop_w = w + 2 * pad
            crop_h = h + 2 * pad
            max_w = max(max_w, crop_w)
            max_h = max(max_h, crop_h)
        target_size = (max_w, max_h)

        for c in seq:
            try:
                fp = c.get("frame_path")
                bbox = c.get("bbox", [])
                if not fp or len(bbox) < 4: continue
                frame = cv2.imread(fp, cv2.IMREAD_COLOR)
                if frame is None: continue

                x, y, w, h = [int(v) for v in bbox]
                pad = 12
                x1, y1 = max(0, x-pad), max(0, y-pad)
                x2, y2 = min(frame.shape[1], x+w+pad), min(frame.shape[0], y+h+pad)

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0: continue

                # Resize/pad to uniform size
                crop = pad_to_size(crop, *target_size)
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_frames.append(crop_rgb)

                # Annotated
                ann = crop.copy()
                rx = x - x1; ry = y - y1
                cv2.rectangle(ann, (rx, ry), (rx+w, ry+h), (0,255,0), 2)
                ann_frames.append(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB))

                # Save individual crop
                ts = (c.get("timestamp") or "unknown").replace(":", "").replace("-", "").replace("T", "_").replace("Z", "")
                crop_fn = f"lasco_track{tid}_{ts}.png"
                crop_path = CROPS_DIR / crop_fn
                cv2.imwrite(str(crop_path), cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))
                c["crop_path"] = f"crops/{crop_fn}"

                extracted.append(c)
            except Exception as e:
                print(f"Crop/GIF error: {e}")
                continue

        # Build GIFs if ≥2 frames
        if len(crop_frames) >= 2 and GIF_AVAILABLE:
            gif_base = f"track{tid}"
            crop_gif_path = GIF_DIR / f"{gif_base}_crop.gif"
            ann_gif_path = GIF_DIR / f"{gif_base}_ann.gif"

            imageio.mimsave(str(crop_gif_path), crop_frames, fps=4, loop=0)
            imageio.mimsave(str(ann_gif_path), ann_frames, fps=4, loop=0)

            for c in seq:
                c["animation_gif_crop"] = f"gifs/{crop_gif_path.name}"
                c["animation_gif_ann"] = f"gifs/{ann_gif_path.name}"
            gif_count += 1

    print(f"Extracted {len(extracted)} crops, created {gif_count} GIFs")
    return extracted

# ==================== AI CLASSIFY ====================
def classify_candidates(candidates):
    if not USE_AI:
        for c in candidates:
            c["ai_label"] = "unknown"; c["ai_score"] = 0.0
        return candidates

    print(f"Classifying {len(candidates)} crops...")
    paths, idxs = [], []
    for i, c in enumerate(candidates):
        cp = c.get("crop_path")
        full = DETECTIONS_DIR / cp if cp else None
        if full and full.exists():
            paths.append(str(full)); idxs.append(i)

    if not paths:
        print("No crops to classify")
        return candidates

    try:
        results = classify_crop_batch(paths)
        comets = 0
        for i, res in zip(idxs, results):
            label = res.get("label", "unknown")
            score = res.get("score", 0.0)
            if AI_VETO and label == AI_VETO_LABEL and score > AI_VETO_MAX:
                candidates[i]["ai_label"] = "vetoed"
                candidates[i]["ai_score"] = score
            else:
                candidates[i]["ai_label"] = label
                candidates[i]["ai_score"] = score
            if candidates[i]["ai_label"] == "comet": comets += 1
        print(f"Done: {comets} comets")
    except Exception as e:
        print(f"AI error: {e}")
        for c in candidates:
            c.setdefault("ai_label", "error"); c.setdefault("ai_score", 0.0)
    return candidates

# ==================== SAVE JSON ====================
def save_results(candidates, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    file = out_dir / f"candidates_{ts}.json"

    clean = []
    for c in candidates:
        entry = {
            "instrument": c.get("instrument"),
            "timestamp": c.get("timestamp"),
            "track_id": c.get("track_id"),
            "bbox": c.get("bbox"),
            "centroid": c.get("centroid"),
            "crop_path": c.get("crop_path"),
            "original_path": c.get("original_path"),
            "annotated_path": c.get("annotated_path"),
            "image_size": c.get("image_size"),
            "ai_label": c.get("ai_label", "unknown"),
            "ai_score": c.get("ai_score", 0.0),
            "positions": c.get("positions", []),
        }
        if c.get("animation_gif_crop"):
            entry["animation_gif_crop"] = c["animation_gif_crop"]
        if c.get("animation_gif_ann"):
            entry["animation_gif_ann"] = c["animation_gif_ann"]
        clean.append(entry)

    with open(file, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"Saved {len(clean)} to {file}")

    summary = {
        "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_candidates": len(clean),
        "comets": len([x for x in clean if x["ai_label"] == "comet"]),
        "output_file": str(file)
    }
    with open(out_dir / "latest_run.json", "w") as f:
        json.dump(summary, f, indent=2)
    return file

# ==================== MAIN ====================
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=int, default=6)
    p.add_argument("--step-min", type=int, default=12)
    p.add_argument("--out", type=str, default="detections")
    args = p.parse_args()

    print("="*60)
    print("SOHO COMET HUNTER + GIF (FIXED)")
    print("="*60)

    paths, ts, _ = fetch_lasco_frames(args.hours, args.step_min)
    if not paths:
        print("No frames")
        return

    cand = detect_candidates(paths, ts)
    if not cand:
        print("No detections")
        return

    cand = apply_spatial_filter(cand)
    if not cand:
        print("All filtered")
        return

    cand = associate_tracks(cand)
    save_frame_assets(cand)
    cand = extract_crops_and_gifs(cand)
    if not cand:
        print("No crops")
        return

    cand = classify_candidates(cand)
    save_results(cand, args.out)

    print("\n" + "="*60)
    print("COMPLETE")
    print(f"Comets: {len([c for c in cand if c.get('ai_label')=='comet'])}")
    print("="*60)

if __name__ == "__main__":
    main()
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
import subprocess
import tempfile
import base64
import shutil

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

def run_cmd(cmd):
    if DEBUG:
        log("RUN:", cmd)
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, cwd=Path.cwd()
    )
    if result.return_code != 0 and DEBUG:
        log("CMD FAILED:", result.stderr)
    return result.stdout

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
# AI CLASSIFIER IMPORT
# ----------------------------------------------------------------------
try:
    from ai_classifier import classify_crop_batch
except Exception as e:
    log("AI classifier import failed – falling back to dummy:", e)
    def classify_crop_batch(paths):  # dummy
        return [{"label": "not_comet", "score": 0.0} for _ in paths]

# ----------------------------------------------------------------------
# SIMPLE TRACK DETECTION + CLASSIFICATION
# ----------------------------------------------------------------------
from scipy.ndimage import gaussian_filter
from filterpy.kalman import KalmanFilter

def simple_track_detect(frame_paths, instr, ts_list):
    """
    Very lightweight tracker:
    - Otsu threshold → contours
    - Kalman filter per candidate (nearest-association)
    - Crop 64×64 → AI classifier
    Returns list of candidate dicts.
    """
    if len(frame_paths) < 4:
        return []

    # Sort chronologically
    paired = sorted(zip(frame_paths, ts_list), key=lambda x: x[1])
    candidates = []

    # Kalman per potential track (we keep a list of active KFs)
    active_kfs = []          # each: (kf, last_pos, age, tid)
    next_id = 0

    for idx, (path, ts) in enumerate(paired):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Light preprocessing
        img_f = gaussian_filter(img.astype(float) / 255.0, sigma=1.0) * 255
        img_f = np.clip(img_f, 0, 255).astype(np.uint8)

        # Otsu threshold
        _, thresh = cv2.threshold(img_f, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # ---- Predict all active tracks ----
        for kf_entry in active_kfs[:]:
            kf, last_pos, age, tid = kf_entry
            kf.predict()
            pred = kf.x[:2, 0].astype(int)
            # age out if no measurement for >3 frames
            if age > 3:
                active_kfs.remove(kf_entry)
                continue
            kf_entry = (kf, pred, age + 1, tid)
            idx_kf = active_kfs.index(kf_entry)
            active_kfs[idx_kf] = kf_entry

        # ---- Associate detections to tracks (nearest) ----
        meas = []
        for mx, my in [(int(cv2.moments(cnt)["m10"]/cv2.moments(cnt)["m00"]), int(cv2.moments(cnt)["m01"]/cv2.moments(cnt)["m00"])) for cnt in contours if cv2.moments(cnt)["m00"] > 10]:
            meas.append((mx, my))

        used = set()
        for kf_idx, (kf, pred, age, tid) in enumerate(active_kfs):
            best_dist = float("inf")
            best_meas = None
            for m_idx, (mx, my) in enumerate(meas):
                if m_idx in used:
                    continue
                d = (mx - pred[0]) ** 2 + (my - pred[1]) ** 2
                if d < best_dist:
                    best_dist = d
                    best_meas = (mx, my)
            if best_meas and best_dist < 80 ** 2:   # max 80 px jump
                # update
                kf.update(np.array(best_meas).reshape(2, 1))
                active_kfs[kf_idx] = (kf, best_meas, 0, tid)
                used.add(meas.index(best_meas))

        # ---- Start new tracks for leftover detections ----
        for mx, my in meas:
            if (mx, my) in [last_pos for _, last_pos, _, _ in active_kfs]:
                continue
            kf = KalmanFilter(dim_x=4, dim_z=2)
            kf.x[:2] = np.array([mx, my]).reshape(2, 1)
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

        # ---- After a few frames, evaluate mature tracks ----
        for kf, pos, age, tid in active_kfs[:]:
            if age == 0 and len([p for p in paired[: idx + 1] if p[1] <= ts]) >= 4:
                # extract crop around current pos
                h, w = img_f.shape
                x, y = pos
                crop = img_f[max(0, y - 32):min(h, y + 32), max(0, x - 32):min(w, x + 32)]
                if crop.size == 0:
                    continue
                tmp_path = Path(tempfile.mktemp(suffix=".png"))
                cv2.imwrite(str(tmp_path), crop)

                cls = classify_crop_batch([str(tmp_path)])[0]
                os.unlink(tmp_path)

                if cls["score"] >= 0.6:      # comet confidence
                    candidates.append({
                        "instrument": instr,
                        "timestamp": ts,
                        "track_id": tid,
                        "bbox": [x - 32, y - 32, 64, 64],
                        "crop_path": f"crops/{instr}_track{tid}_{ts.replace(':', '')}.png",
                        "ai_label": cls["label"],
                        "ai_score": round(cls["score"], 3)
                    })
                    # save crop for later inspection
                    crop_dir = OUT_DIR / "crops"
                    crop_dir.mkdir(parents=True, exist_ok=True)
                    final_crop = crop_dir / Path(candidates[-1]["crop_path"]).name
                    cv2.imwrite(str(final_crop), crop)
                    candidates[-1]["crop_path"] = str(final_crop.relative_to(OUT_DIR))

    return candidates

# ----------------------------------------------------------------------
# WRITE ENHANCED STATUS (ALWAYS)
# ----------------------------------------------------------------------
def write_status(c2_frames, c3_frames, tracks, c2_times, c3_times):
    total_analyzed = c2_frames + c3_frames

    def time_range(times):
        if not times:
            return "none"
        start = min(times)
        end = max(times)
        return f"{start.split('T')[1][:5]}–{end.split('T')[1][:5]} UTC"

    c2_range = time_range(c2_times)
    c3_range = time_range(c3_times)

    status = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "summary": f"Analyzed {c2_frames} C2 ({c2_range}) + {c3_frames} C3 ({c3_range}) = {total_analyzed} frames",
        "detectors": {
            "C2": {
                "frames": c2_frames,
                "tracks": tracks,
                "time_range": c2_range,
                "timestamps": c2_times
            },
            "C3": {
                "frames": c3_frames,
                "tracks": tracks,
                "time_range": c3_range,
                "timestamps": c3_times
            }
        },
        "total": {
            "frames_analyzed": total_analyzed,
            "tracks_found": tracks
        }
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    status_path = OUT_DIR / "latest_status.json"
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
    log(f"latest_status.json → {status_path} ({total_analyzed} analyzed)")

# ----------------------------------------------------------------------
# WRITE RUN PROOF (ALWAYS)
# ----------------------------------------------------------------------
def write_latest_run(c2_frames, c3_frames, candidates_count):
    run_info = {
        "last_run_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "status": "completed",
        "frames_downloaded": {"C2": c2_frames, "C3": c3_frames},
        "candidates_found": candidates_count,
        "note": "No new frames" if c2_frames + c3_frames == 0 else "OK"
    }
    run_path = OUT_DIR / "latest_run.json"
    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)
    log(f"latest_run.json → {run_path}")

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    log("=== DETECTION START ===")
    FRAMES_DIR.mkdir(exist_ok=True)

    # 1. Download frames + timestamps
    frames, dl_counts, timestamps = download_frames()
    c2_cnt = dl_counts["C2"]
    c3_cnt = dl_counts["C3"]
    c2_times = timestamps["C2"]
    c3_times = timestamps["C3"]

    # 2. Run detection
    c2_cands = simple_track_detect(frames["C2"], "C2", c2_times)
    c3_cands = simple_track_detect(frames["C3"], "C3", c3_times)
    all_cands = c2_cands + c3_cands
    tracks = len({c["track_id"] for c in all_cands})   # unique tracks

    # 3. Save candidates if any
    if all_cands:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_file = OUT_DIR / f"candidates_{timestamp}.json"
        with open(out_file, "w") as f:
            json.dump(all_cands, f, indent=2)
        log(f"Saved {len(all_cands)} candidates → {out_file.name}")

    # 4. Always write status & run proof
    write_status(c2_cnt, c3_cnt, tracks, c2_times, c3_times)
    write_latest_run(c2_cnt, c3_cnt, len(all_cands))

    # 5. Debug summary
    log("=== SUMMARY ===")
    log(f"C2 frames: {c2_cnt}, C3 frames: {c3_cnt}")
    log(f"Candidates: {len(all_cands)}, Unique tracks: {tracks}")
    log("=== END ===")

if __name__ == "__main__":
    main()
