#!/usr/bin/env python3
"""
SOHO Comet Hunter — detect_comets.py
Detects moving objects in LASCO C2/C3 sequences.
Outputs: JSON report with ai_label/ai_score, crops, animations, etc.
"""

import argparse
import json
import os
import sys
import cv2
import numpy as np
import requests
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import imageio
from scipy.ndimage import gaussian_filter
from filterpy.kalman import KalmanFilter
import fetch_lasco  # <-- NEW: use the robust fetcher

# --------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------
BASE_URL = "https://soho.nascom.nasa.gov/data/REPROCESSING/Completed"
VALID_DETS = ["C2", "C3"]
IMG_SIZE = {"C2": (512, 512), "C3": (1024, 1024)}
OCCULTER_FRACTION = float(os.getenv("OCCULTER_RADIUS_FRACTION", "0.18"))
MAX_EDGE_FRACTION = float(os.getenv("MAX_EDGE_RADIUS_FRACTION", "0.98"))
DUAL_MAX_MINUTES = int(os.getenv("DUAL_CHANNEL_MAX_MINUTES", "60"))
DUAL_MAX_ANGLE_DIFF = float(os.getenv("DUAL_CHANNEL_MAX_ANGLE_DIFF", "25"))
SELECT_TOP_N = int(os.getenv("SELECT_TOP_N_FOR_SUBMIT", "3"))
DEBUG = os.getenv("DETECTOR_DEBUG", "0") == "1"
USE_AI = os.getenv("USE_AI_CLASSIFIER", "1") == "1"
AI_VETO = os.getenv("AI_VETO_ENABLED", "1") == "1"
AI_VETO_LABEL = os.getenv("AI_VETO_LABEL", "not_comet")
AI_VETO_MAX = float(os.getenv("AI_VETO_SCORE_MAX", "0.9"))

# --------------------------------------------------------------
# UTILS
# --------------------------------------------------------------
def log(*a, **kw):
    print(*a, **kw, file=sys.stderr)

def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def download_file(url, path):
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        return True
    except Exception as e:
        log(f"Download failed {url}: {e}")
        return False

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        log(f"Failed to load {path}")
    return img

def timestamp_from_name(name):
    # Handle: C2_20251028_1336_c2_1024.jpg or original SOHO names
    name = name.lower()
    # Look for YYYYMMDD_HHMMSS pattern
    import re
    m = re.search(r'(\d{8}_\d{6})', name)
    if m:
        try:
            dt = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
            return dt.isoformat() + "Z"
        except:
            pass
    return ""

# --------------------------------------------------------------
# ANIMATION WRITER
# --------------------------------------------------------------
def write_animation_for_track(det, names, imgs, tr, out_dir, fps=6, radius=4):
    tmin, tmax = tr[0][0], tr[-1][0]
    xy = {t: (int(round(x)), int(round(y))) for t, x, y, _ in tr}
    trail = []
    clean, annot = [], []

    for ti in range(tmin, tmax + 1):
        im = imgs[ti]
        bgr = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) if im.ndim == 2 else im.copy()
        clean.append(bgr.copy())
        if ti in xy:
            trail.append(xy[ti])
        for a, b in zip(trail[:-1], trail[1:]):
            cv2.line(bgr, a, b, (0, 255, 0), 1)
        if ti in xy:
            cv2.circle(bgr, xy[ti], radius, (0, 255, 0), 1)
        annot.append(bgr)

    anim_dir = ensure_dir(Path(out_dir) / "animations")
    ident = tr[0][0]
    base_a = anim_dir / f"{det}_track{ident}_annotated"
    base_c = anim_dir / f"{det}_track{ident}_clean"
    out = {
        "animation_gif_path": None,
        "animation_mp4_path": None,
        "animation_gif_clean_path": None,
        "animation_mp4_clean_path": None,
    }

    if imageio:
        try:
            imageio.mimsave(str(base_a.with_suffix(".gif")), annot, fps=fps)
            out["animation_gif_path"] = str(base_a.with_suffix(".gif"))
            imageio.mimsave(str(base_c.with_suffix(".gif")), clean, fps=fps)
            out["animation_gif_clean_path"] = str(base_c.with_suffix(".gif"))
        except Exception as e:
            log(f"GIF write failed: {e}")

    return out

# --------------------------------------------------------------
# DETECTION IN ONE SEQUENCE
# --------------------------------------------------------------
def detect_in_sequence(det, det_frames, out_dir, hours, step_min):
    # Load and sort frames chronologically (newest first)
    img_paths = sorted(
        [p for p in det_frames.glob("*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif"}],
        key=lambda p: p.name,
        reverse=True,
    )
    if not img_paths:
        log(f"No frames for {det}")
        return []

    names = [p.name for p in img_paths]
    imgs = [load_image(p) for p in img_paths]
    imgs = [im for im in imgs if im is not None]
    if not imgs:
        return []

    # Build timestamp index
    timestamps = [timestamp_from_name(n) for n in names]
    time_idx = {i: ts for i, ts in enumerate(timestamps) if ts}

    # Simple background subtraction
    if len(imgs) < 2:
        log(f"Not enough frames for {det} to compute background")
        return []

    stack = np.stack(imgs[:min(10, len(imgs))])  # Use up to 10 newest
    bg = gaussian_filter(np.median(stack, axis=0).astype(np.float32), sigma=1)

    # Ensure all images are float32
    diff = []
    for im in imgs:
        im_f = im.astype(np.float32)
        d = cv2.absdiff(im_f, bg)
        diff.append(d)

    # Kalman-filter based linking
    tracks = []
    for i, pts in enumerate(points_per_frame):
        for pt in pts:
            x, y = pt
            # simple nearest-neighbor link
            matched = None
            for tr in tracks:
                if abs(tr[-1][0] - i) < 3:
                    dist = np.hypot(tr[-1][1] - x, tr[-1][2] - y)
                    if dist < 30:
                        if matched is None or dist < matched[1]:
                            matched = (tr, dist)
            if matched:
                matched[0].append((i, x, y, 0))
            else:
                tracks.append([(i, x, y, 0)])

    # Filter tracks
    good_tracks = [tr for tr in tracks if len(tr) >= 3]

    # AI classification
    from ai_classifier import classify_crop_batch
    crops_dir = ensure_dir(out_dir / "crops")
    candidates = []
    for idx, tr in enumerate(good_tracks):
        mid_t = len(tr) // 2
        im = imgs[tr[mid_t][0]]
        x, y = int(tr[mid_t][1]), int(tr[mid_t][2])
        sz = 64 if det == "C2" else 128
        crop = im[max(0, y-sz):y+sz, max(0, x-sz):x+sz]
        crop_path = crops_dir / f"{det}_track{idx}_crop.png"
        cv2.imwrite(str(crop_path), crop)

        ai = classify_crop_batch([str(crop_path)])[0]

        # Veto logic
        if AI_VETO and ai["label"] == AI_VETO_LABEL and ai["score"] > AI_VETO_MAX:
            ai["label"] = "vetoed"

        anim_paths = write_animation_for_track(det, names, imgs, tr, out_dir)

        cand = {
            "detector": det,
            "track_index": idx,
            "score": len(tr) * ai["score"],
            "positions": [
                {"time_idx": t, "x": round(x, 1), "y": round(y, 1), "time_utc": timestamps[t] if t in time_idx else ""}
                for t, x, y, _ in tr
            ],
            "crop_path": str(crop_path.relative_to(out_dir)),
            "annotated_overlay_path": f"detections/annotated/{names[mid_t]}_annot_track.png",
            "ai_label": ai["label"],
            "ai_score": ai["score"],
            "auto_selected": True
        }
        cand.update(anim_paths)
        candidates.append(cand)

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:SELECT_TOP_N]

# --------------------------------------------------------------
# DUAL CHANNEL MATCH
# --------------------------------------------------------------
def match_dual_channel(c2_cands, c3_cands):
    for c2 in c2_cands:
        t2 = datetime.fromisoformat(c2["positions"][-1]["time_utc"].replace("Z", "+00:00"))
        for c3 in c3_cands:
            t3 = datetime.fromisoformat(c3["positions"][0]["time_utc"].replace("Z", "+00:00"))
            dt = abs((t3 - t2).total_seconds() / 60)
            if dt > DUAL_MAX_MINUTES:
                continue
            pa2 = np.arctan2(c2["positions"][-1]["y"] - 256, c2["positions"][-1]["x"] - 256) * 180 / np.pi
            pa3 = np.arctan2(c3["positions"][0]["y"] - 512, c3["positions"][0]["x"] - 512) * 180 / np.pi
            diff = min(abs(pa2 - pa3), 360 - abs(pa2 - pa3))
            if diff <= DUAL_MAX_ANGLE_DIFF:
                c2["dual_channel_match"] = {
                    "with": f"{c3['detector']}#{c3['track_index']}",
                    "pa_diff_deg": round(diff, 1),
                }
                break

# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=6)
    parser.add_argument("--step-min", type=int, default=12)
    parser.add_argument("--out", type=str, default="detections")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out)
    frames_dir = ensure_dir("frames")

    # ---- NEW: Use fetch_lasco to download real recent images ----
    saved = fetch_lasco.fetch_window(
        hours_back=args.hours,
        step_min=args.step_min,
        root=str(frames_dir)
    )
    log(f"Downloaded {len(saved)} recent frames")

    all_cands = []
    for det in VALID_DETS:
        det_frames = frames_dir / det
        if not any(det_frames.glob("*.*")):
            continue
        cands = detect_in_sequence(det, det_frames, out_dir, args.hours, args.step_min)
        all_cands.extend(cands)

    # Dual-channel matching
    c2_cands = [c for c in all_cands if c["detector"] == "C2"]
    c3_cands = [c for c in all_cands if c["detector"] == "C3"]
    match_dual_channel(c2_cands, c3_cands)

    # Write report
    now = datetime.utcnow()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    report_path = out_dir / f"candidates_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(all_cands, f, indent=2)

    # Write status
    status = {
        "timestamp_utc": now.isoformat() + "Z",
        "detectors": {
            det: {
                "frames": len(list((frames_dir / det).glob("*.*"))),
                "tracks": len([c for c in all_cands if c["detector"] == det]),
            } for det in VALID_DETS
        }
    }
    with open(out_dir / "latest_status.json", "w") as f:
        json.dump(status, f, indent=2)

    log(f"Detected {len(all_cands)} candidates → {report_path}")

if __name__ == "__main__":
    main()
