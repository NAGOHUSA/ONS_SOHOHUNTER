#!/usr/bin/env python3
"""
SOHO Comet Detection Pipeline
- Fetches LASCO frames
- Masks timestamp + central occulter
- Detects comets only
- Tracks, crops, annotates
- Saves JSON + images
"""

from datetime import datetime
from pathlib import Path
import json
import cv2
import numpy as np
import os
import sys
from collections import defaultdict

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
    """Mask central dark disk + bright ring."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    radius = int(min(w, h) * 0.18)  # ~180px
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

            # MASK: timestamp + central occulter
            frame_masked = frame.copy()
            ts_mask = build_timestamp_mask(h, w)
            frame_masked[ts_mask == 255] = 0
            frame_masked = mask_central_occulter(frame_masked, buffer=25)  # KEY FIX

            # Threshold bright features
            _, thresh = cv2.threshold(frame_masked, 200, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3,3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                x, y, bw, bh = cv2.boundingRect(cnt)
                if bw < 8 or bh < 8 or bw > 220 or bh > 220:
                    continue
                if bbox_intersects_mask((x,y,bw,bh), ts_mask, 10):
                    continue

                candidates.append({
                    "instrument": instr,
                    "timestamp": timestamps[i] if i < len(timestamps) else "",
                    "bbox": [x, y, bw, bh],
                    "frame_path": fp,
                    "image_size": [h, w],
                    "_ts_mask": ts_mask,
                    "_frame_gray": frame
                })
        except Exception as e:
            print(f"Error: {fp} → {e}")
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
    in_corner = (
        (x < cm and y < cm) or
        (x + w > fw - cm and y < cm) or
        (x < cm and y + h > fh - cm) or
        (x + w > fw - cm and y + h > fh - cm)
    )
    on_edge = (y < em or y + h > fh - em or x < em or x + w > fw - em)
    aspect = w / h if h > 0 else 0
    text_shaped = (2.0 < aspect < 10.0 and w < 320 and h < 80)
    tiny = (w < 30 or h < 30) and in_corner
    return (in_corner and (text_shaped or tiny)) or (on_edge and text_shaped and (w < 260 or h < 48))

def filter_detections_spatial(candidates, fw, fh, mask, gray):
    filtered = []
    removed = 0
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
        for det in track:
            det['track_id'] = track_id
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

# ==================== CROP EXTRACTION ====================
def extract_crops(candidates):
    print(f"Extracting {len(candidates)} crops...")
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    extracted = []
    for c in candidates:
        try:
            fp = c.get("frame_path")
            bbox = c.get("bbox", [])
            if not fp or len(bbox) < 4: continue
            frame = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            if frame is None: continue
            x, y, w, h = bbox
            pad = 10
            x1, y1 = max(0, x-pad), max(0, y-pad)
            x2, y2 = min(frame.shape[1], x+w+pad), min(frame.shape[0], y+h+pad)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue
            ts = (c.get("timestamp") or "unknown").replace(":", "").replace("-", "").replace("T", "_").replace("Z", "")
            tid = c.get("track_id", 0)
            fn = f"lasco_track{tid}_{ts}.png"
            path = CROPS_DIR / fn
            cv2.imwrite(str(path), crop)
            c["crop_path"] = f"crops/{fn}"
            extracted.append(c)
        except Exception as e:
            print(f"Crop error: {e}")
            continue
    print(f"Extracted {len(extracted)} crops")
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
        clean.append({
            "instrument": c.get("instrument"),
            "timestamp": c.get("timestamp"),
            "track_id": c.get("track_id"),
            "bbox": c.get("bbox"),
            "crop_path": c.get("crop_path"),
            "original_path": c.get("original_path"),
            "annotated_path": c.get("annotated_path"),
            "image_size": c.get("image_size"),
            "ai_label": c.get("ai_label", "unknown"),
            "ai_score": c.get("ai_score", 0.0),
        })

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
    print("SOHO COMET HUNTER")
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
    cand = extract_crops(cand)
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
