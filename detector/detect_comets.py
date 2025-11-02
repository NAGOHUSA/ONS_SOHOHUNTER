#!/usr/bin/env python3
"""
SOHO Comet Detection Pipeline
- Fetches LASCO frames
- Masks timestamp/edge overlays before detection
- Detects moving/bright objects
- Filters timestamp/edge artifacts
- Simple tracking across frames
- Runs AI classification
- Saves originals, annotated frames
- Saves results as JSON with additional paths
"""

from datetime import datetime
from pathlib import Path
import json
import cv2
import numpy as np
import os
import sys
from collections import defaultdict

# Add detector directory to path if needed
sys.path.insert(0, str(Path(__file__).parent))

try:
    from ai_classifier import classify_crop_batch
except ImportError:
    print("Warning: ai_classifier not found, using dummy classifier")
    def classify_crop_batch(paths):
        return [{"label": "unknown", "score": 0.0} for _ in paths]

# ==================== CONFIGURATION ====================
FRAMES_DIR = Path("frames")
DETECTIONS_DIR = Path("detections")
CROPS_DIR = DETECTIONS_DIR / "crops"
ORIGINALS_DIR = DETECTIONS_DIR / "originals"
ANNOTATED_DIR = DETECTIONS_DIR / "annotated"

# Typical LASCO sizes (most LATEST 1024)
LASCO_C2_SIZE = (1024, 1024)
LASCO_C3_SIZE = (1024, 1024)

# AI configuration
USE_AI = os.getenv("USE_AI_CLASSIFIER", "1") == "1"
AI_VETO = os.getenv("AI_VETO_ENABLED", "1") == "1"
AI_VETO_LABEL = os.getenv("AI_VETO_LABEL", "not_comet")
AI_VETO_MAX = float(os.getenv("AI_VETO_SCORE_MAX", "0.9"))

# Tracking params
MAX_ASSOC_DIST = 100  # pixels between frames

# ==================== TIMESTAMP / OVERLAY MASK ====================

def build_timestamp_mask(h: int, w: int) -> np.ndarray:
    """Build a binary mask for LASCO overlays (255 = ignore)."""
    mask = np.zeros((h, w), np.uint8)

    cw = max(120, w // 8)   # corner width
    ch = max(80,  h // 8)   # corner height
    edge = max(28, min(h, w) // 36)

    # Corners
    mask[0:ch, 0:cw] = 255
    mask[0:ch, w-cw:w] = 255
    mask[h-ch:h, 0:cw] = 255
    mask[h-ch:h, w-cw:w] = 255

    # Thin edge bands
    mask[0:edge, :] = 255
    mask[h-edge:h, :] = 255
    mask[:, 0:edge] = 255
    mask[:, w-edge:w] = 255

    return mask

def bbox_intersects_mask(bbox, mask: np.ndarray, min_overlap_px: int = 25) -> bool:
    """Return True if bbox overlaps masked area by >= min_overlap_px."""
    if len(bbox) < 4:
        return False
    x, y, w, h = [int(v) for v in bbox[:4]]
    x2, y2 = x + w, y + h
    x, y = max(0, x), max(0, y)
    x2, y2 = min(mask.shape[1], x2), min(mask.shape[0], y2)
    if x >= x2 or y >= y2:
        return False
    sub = mask[y:y2, x:x2]
    return int(np.count_nonzero(sub)) >= min_overlap_px

def is_timestamp_artifact(bbox, frame_width, frame_height) -> bool:
    """Heuristic spatial filter for text-like corner/edge detections."""
    if len(bbox) < 4:
        return False
    x, y, w, h = bbox[:4]

    corner_margin = max(100, min(frame_width, frame_height)//9)
    edge_margin = max(50, min(frame_width, frame_height)//22)

    in_top_left  = (x < corner_margin and y < corner_margin)
    in_top_right = (x + w > frame_width - corner_margin and y < corner_margin)
    in_bot_left  = (x < corner_margin and y + h > frame_height - corner_margin)
    in_bot_right = (x + w > frame_width - corner_margin and y + h > frame_height - corner_margin)
    in_corner = in_top_left or in_top_right or in_bot_left or in_bot_right

    on_edge = (y < edge_margin) or (y + h > frame_height - edge_margin) or \
              (x < edge_margin) or (x + w > frame_width - edge_margin)

    aspect = w / h if h > 0 else 0.0
    text_shaped = (2.0 < aspect < 10.0 and w < 320 and h < 80)

    tiny = (w < 30 or h < 30) and in_corner

    if in_corner and (text_shaped or tiny):
        return True
    if on_edge and text_shaped and (w < 260 or h < 48):
        return True

    return False

def filter_detections_spatial(candidates, frame_width, frame_height, mask: np.ndarray, frame_gray: np.ndarray):
    """Filter out timestamp/edge artifacts."""
    filtered = []
    removed = 0

    for c in candidates:
        bbox = c.get("bbox", [])
        if not bbox or len(bbox) < 4:
            continue

        if bbox_intersects_mask(bbox, mask):
            removed += 1
            continue

        if is_timestamp_artifact(bbox, frame_width, frame_height):
            removed += 1
            continue

        x, y, w, h = [int(v) for v in bbox[:4]]
        x2, y2 = x + w, y + h
        x, y = max(0, x), max(0, y)
        x2, y2 = min(frame_width, x2), min(frame_height, y2)
        roi = frame_gray[y:y2, x:x2]
        if roi.size == 0:
            continue

        bright_ratio = float(np.count_nonzero(roi > 220)) / float(roi.size)
        if bright_ratio > 0.35:
            removed += 1
            continue

        filtered.append(c)

    if removed:
        print(f"  Removed {removed} timestamp/edge artifacts")
    return filtered

# ==================== FRAME FETCHING ====================

def fetch_lasco_frames(hours=6, step_min=12):
    """Fetch LASCO frames from NASA or local cache."""
    print(f"Fetching LASCO frames (last {hours}h, step {step_min}min)...")
    try:
        from fetch_lasco import fetch_window
        frames = fetch_window(hours_back=hours, step_min=step_min, root=str(FRAMES_DIR))
        timestamps = []
        for frame_path in frames:
            try:
                parts = Path(frame_path).stem.split('_')
                if len(parts) >= 3:
                    date_str, time_str = parts[1], parts[2]
                    ts = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M")
                    timestamps.append(ts.strftime("%Y-%m-%dT%H:%M:%SZ"))
                else:
                    timestamps.append(datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))
            except:
                timestamps.append(datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))
        print(f"Fetched {len(frames)} frames")
        return frames, timestamps, len(frames)
    except ImportError:
        print("Warning: fetch_lasco module not found, using fallback (existing frames)")
        frames = []
        for det in ["C2", "C3"]:
            det_dir = FRAMES_DIR / det
            if det_dir.exists():
                frames.extend([str(f) for f in sorted(det_dir.glob("*.jpg")) + sorted(det_dir.glob("*.png"))])
        timestamps = [datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")] * len(frames)
        return frames, timestamps, 0

# ==================== BASIC DETECTION ====================

def detect_candidates(frame_paths, timestamps):
    """Bright-object detection with pre-masking of timestamp overlays."""
    print(f"Running detection on {len(frame_paths)} frames...")
    if len(frame_paths) < 1:
        print("No frames to process")
        return []

    candidates = []

    for i, frame_path in enumerate(frame_paths):
        try:
            frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            if frame is None:
                continue
            h, w = frame.shape[:2]

            instr = "LASCO C2" if "c2" in Path(frame_path).name.lower() else "LASCO C3"

            ts_mask = build_timestamp_mask(h, w)
            frame_masked = frame.copy()
            frame_masked[ts_mask == 255] = 0

            _, thresh = cv2.threshold(frame_masked, 200, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                x, y, bw, bh = cv2.boundingRect(cnt)

                if bw < 8 or bh < 8 or bw > 220 or bh > 220:
                    continue

                if bbox_intersects_mask((x, y, bw, bh), ts_mask, min_overlap_px=10):
                    continue

                candidates.append({
                    "instrument": instr,
                    "timestamp": timestamps[i] if i < len(timestamps) else "",
                    "bbox": [int(x), int(y), int(bw), int(bh)],
                    "frame_path": frame_path,
                    "image_size": [h, w],
                    "_ts_mask": ts_mask,
                    "_frame_gray": frame
                })

        except Exception as e:
            print(f"Error processing {frame_path}: {e}")
            continue

    print(f"Found {len(candidates)} initial detections")
    return candidates

# ==================== SPATIAL FILTER ====================

def apply_spatial_filter(candidates):
    """Apply per-candidate filtering using stored mask/gray."""
    filtered = []
    for c in candidates:
        bbox = c["bbox"]
        w, h = c["image_size"][1], c["image_size"][0]
        mask = c["_ts_mask"]
        gray = c["_frame_gray"]
        # filter_detections_spatial expects a list
        kept = filter_detections_spatial([c], w, h, mask, gray)
        if kept:
            filtered.append(kept[0])
    return filtered

# ==================== TRACKING ====================

def associate_tracks(candidates):
    """Simple nearest-neighbor association across frames."""
    print("Associating tracks across frames...")
    time_groups = defaultdict(list)
    for c in candidates:
        time_groups[c['timestamp']].append(c)

    sorted_times = sorted(time_groups.keys())
    active_tracks = []
    track_id_counter = 0

    for t in sorted_times:
        current_dets = time_groups[t]
        new_active = []
        assigned = set()

        for det in current_dets:
            best = None
            min_dist = float('inf')
            for j, tr in enumerate(active_tracks):
                if j in assigned:
                    continue
                last = tr[-1]
                dx = det['bbox'][0] - last['bbox'][0]
                dy = det['bbox'][1] - last['bbox'][1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < min_dist and dist < MAX_ASSOC_DIST:
                    min_dist = dist
                    best = j
            if best is not None:
                active_tracks[best].append(det)
                new_active.append(active_tracks[best])
                assigned.add(best)
            else:
                new_track = [det]
                new_active.append(new_track)

        for j, tr in enumerate(active_tracks):
            if j not in assigned:
                new_active.append(tr)

        active_tracks = new_active

    # Assign final track IDs
    for track in active_tracks:
        for det in track:
            det['track_id'] = track_id_counter
        track_id_counter += 1

    print(f"Associated {len(active_tracks)} tracks")
    return candidates

# ==================== SAVE ORIGINALS / ANNOTATED ====================

def save_frame_assets(candidates):
    """Save clean originals + annotated frames (green boxes)."""
    print("Saving originals and annotated frames...")
    ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

    frame_groups = defaultdict(list)
    for c in candidates:
        frame_groups[c['frame_path']].append(c)

    saved = 0
    for frame_path, dets in frame_groups.items():
        try:
            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            stem = Path(frame_path).stem
            instr = dets[0]['instrument'].replace(' ', '_').lower()

            # Original
            orig_filename = f"lasco_{instr}_{stem}.jpg"
            orig_path = ORIGINALS_DIR / orig_filename
            cv2.imwrite(str(orig_path), frame)

            # Annotated
            ann_frame = frame.copy()
            for d in dets:
                x, y, w, h = d['bbox']
                cv2.rectangle(ann_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            ann_filename = f"lasco_{instr}_{stem}_annotated.jpg"
            ann_path = ANNOTATED_DIR / ann_filename
            cv2.imwrite(str(ann_path), ann_frame)

            # Attach paths to every detection in this frame
            for d in dets:
                d['original_path'] = f"originals/{orig_filename}"
                d['annotated_path'] = f"annotated/{ann_filename}"

            saved += 1
        except Exception as e:
            print(f"Error saving assets for {frame_path}: {e}")

    print(f"Saved assets for {saved} frames")

# ==================== CROP EXTRACTION ====================

def extract_crops(candidates):
    """Extract padded crops → detections/crops/."""
    print(f"Extracting {len(candidates)} crops...")
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    extracted = []
    for c in candidates:
        try:
            frame_path = c.get("frame_path")
            bbox = c.get("bbox", [])
            if not frame_path or len(bbox) < 4:
                continue

            frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            if frame is None:
                continue

            x, y, w, h = bbox
            pad = 10
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Filename: lasco_track{TID}_{TIMESTAMP}.png
            ts_clean = (c.get("timestamp") or "unknown").replace(":", "").replace("-", "").replace("T", "_").replace("Z", "")
            tid = c.get("track_id", 0)
            crop_filename = f"lasco_track{tid}_{ts_clean}.png"
            crop_path = CROPS_DIR / crop_filename
            cv2.imwrite(str(crop_path), crop)

            c["crop_path"] = f"crops/{crop_filename}"
            extracted.append(c)
        except Exception as e:
            print(f"Error extracting crop for track {c.get('track_id')}: {e}")
            continue

    print(f"Extracted {len(extracted)} crops")
    return extracted

# ==================== AI CLASSIFICATION ====================

def classify_candidates(candidates):
    if not USE_AI:
        print("AI classification disabled")
        for c in candidates:
            c["ai_label"] = "unknown"
            c["ai_score"] = 0.0
        return candidates

    print(f"Running AI classification on {len(candidates)} candidates...")

    crop_paths = []
    valid_idx = []
    for i, c in enumerate(candidates):
        cp = c.get("crop_path")
        full = (DETECTIONS_DIR / cp) if cp else None
        if full and full.exists():
            crop_paths.append(str(full))
            valid_idx.append(i)

    if not crop_paths:
        print("No valid crops for classification")
        return candidates

    try:
        results = classify_crop_batch(crop_paths)
        comet_count = 0
        for idx, res in zip(valid_idx, results):
            label = res.get("label", "unknown")
            score = res.get("score", 0.0)
            if AI_VETO and label == AI_VETO_LABEL and score > AI_VETO_MAX:
                candidates[idx]["ai_label"] = "vetoed"
                candidates[idx]["ai_score"] = score
            else:
                candidates[idx]["ai_label"] = label
                candidates[idx]["ai_score"] = score
            if candidates[idx]["ai_label"] == "comet":
                comet_count += 1
        print(f"Classification complete: {comet_count} comets identified")
    except Exception as e:
        print(f"Error during classification: {e}")
        for c in candidates:
            c.setdefault("ai_label", "error")
            c.setdefault("ai_score", 0.0)

    return candidates

# ==================== SAVE RESULTS ====================

def save_results(candidates, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"candidates_{timestamp}.json"

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

    with open(out_file, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"Saved {len(clean)} candidates to {out_file}")

    summary = {
        "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_candidates": len(clean),
        "comets": len([x for x in clean if x["ai_label"] == "comet"]),
        "output_file": str(out_file)
    }
    with open(output_dir / "latest_run.json", "w") as f:
        json.dump(summary, f, indent=2)

    return out_file

# ==================== MAIN ====================

def main():
    import argparse
    p = argparse.ArgumentParser(description="SOHO Comet Detection")
    p.add_argument("--hours", type=int, default=6)
    p.add_argument("--step-min", type=int, default=12)
    p.add_argument("--out", type=str, default="detections")
    args = p.parse_args()

    print("="*60)
    print("SOHO COMET HUNTER - Detection Pipeline")
    print("="*60)

    frame_paths, timestamps, _ = fetch_lasco_frames(hours=args.hours, step_min=args.step_min)
    if not frame_paths:
        print("No frames available.")
        return

    candidates = detect_candidates(frame_paths, timestamps)
    if not candidates:
        print("No candidates detected.")
        return

    candidates = apply_spatial_filter(candidates)
    print(f"{len(candidates)} candidates remain after filtering\n")
    if not candidates:
        print("All candidates filtered out.")
        return

    candidates = associate_tracks(candidates)
    save_frame_assets(candidates)
    candidates = extract_crops(candidates)
    if not candidates:
        print("No crops extracted.")
        return

    candidates = classify_candidates(candidates)
    out = save_results(candidates, args.out)

    print("\n" + "="*60)
    print("DETECTION COMPLETE")
    print("="*60)
    print(f"Total detections: {len(candidates)}")
    print(f"AI comets: {len([c for c in candidates if c.get('ai_label') == 'comet'])}")
    print(f"Results: {out}")
    print("="*60)

if __name__ == "__main__":
    main()
