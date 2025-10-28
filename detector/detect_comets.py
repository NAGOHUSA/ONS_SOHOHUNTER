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
from pathlib import Path
from datetime import datetime
import imageio
from scipy.ndimage import gaussian_filter
import fetch_lasco  # Robust image fetcher

# --------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------
VALID_DETS = ["C2", "C3"]
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

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        log(f"Failed to load {path}")
    return img

def timestamp_from_name(name):
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
    xy = {t: (int(round(x)), int(round(y))) for t, x, y in tr}
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
        "animation_gif_clean_path": None,
    }

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
    # Load and sort frames (newest first)
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
    if len(imgs) < 2:
        log(f"Not enough valid frames for {det}")
        return []

    timestamps = [timestamp_from_name(n) for n in names]

    # Background subtraction
    stack = np.stack(imgs[:min(10, len(imgs))])
    bg = gaussian_filter(np.median(stack, axis=0).astype(np.float32), sigma=1)
    diff = [cv2.absdiff(im.astype(np.float32), bg) for im in imgs]

    # Threshold & find bright points
    points_per_frame = []
    for d in diff:
        _, thr = cv2.threshold(d, 30, 255, cv2.THRESH_BINARY)
        thr8 = thr.astype(np.uint8)
        contours, _ = cv2.findContours(thr8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pts = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 5 < area < 200:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    pts.append((cx, cy))
        points_per_frame.append(pts)

    # Simple nearest-neighbor tracking
    tracks = []
    for i, pts in enumerate(points_per_frame):
        for pt in pts:
            x, y = pt
            matched = None
            min_dist = 30
            for tr in tracks:
                if abs(tr[-1][0] - i) <= 3:
                    prev_x, prev_y = tr[-1][1], tr[-1][2]
                    dist = np.hypot(x - prev_x, y - prev_y)
                    if dist < min_dist:
                        min_dist = dist
                        matched = tr
            if matched:
                matched.append((i, x, y))
            else:
                tracks.append([(i, x, y)])

    # Filter: at least 3 points
    good_tracks = [tr for tr in tracks if len(tr) >= 3]

    # AI classification
    from ai_classifier import classify_crop_batch
    crops_dir = ensure_dir(out_dir / "crops")
    candidates = []

    for idx, tr in enumerate(good_tracks):
        mid_t = len(tr) // 2
        frame_idx = tr[mid_t][0]
        im = imgs[frame_idx]
        x, y = int(tr[mid_t][1]), int(tr[mid_t][2])
        sz = 64 if det == "C2" else 128
        h, w = im.shape
        y1, y2 = max(0, y - sz), min(h, y + sz)
        x1, x2 = max(0, x - sz), min(w, x + sz)
        crop = im[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_path = crops_dir / f"{det}_track{idx}_crop.png"
        cv2.imwrite(str(crop_path), crop)

        ai = classify_crop_batch([str(crop_path)])[0]

        # Veto
        if AI_VETO and ai["label"] == AI_VETO_LABEL and ai["score"] > AI_VETO_MAX:
            ai["label"] = "vetoed"

        anim_paths = write_animation_for_track(det, names, imgs, tr, out_dir)

        cand = {
            "detector": det,
            "track_index": idx,
            "score": len(tr) * ai["score"],
            "positions": [
                {
                    "time_idx": t,
                    "x": round(x, 1),
                    "y": round(y, 1),
                    "time_utc": timestamps[t] if t < len(timestamps) and timestamps[t] else ""
                }
                for t, x, y in tr
            ],
            "crop_path": str(crop_path.relative_to(out_dir)),
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
        if not c2["positions"]: continue
        t2_str = c2["positions"][-1]["time_utc"]
        if not t2_str: continue
        try:
            t2 = datetime.fromisoformat(t2_str.replace("Z", "+00:00"))
        except:
            continue

        for c3 in c3_cands:
            if not c3["positions"]: continue
            t3_str = c3["positions"][0]["time_utc"]
            if not t3_str: continue
            try:
                t3 = datetime.fromisoformat(t3_str.replace("Z", "+00:00"))
            except:
                continue

            dt = abs((t3 - t2).total_seconds() / 60)
            if dt > 60:
                continue

            pa2 = np.arctan2(c2["positions"][-1]["y"] - 256, c2["positions"][-1]["x"] - 256) * 180 / np.pi
            pa3 = np.arctan2(c3["positions"][0]["y"] - 512, c3["positions"][0]["x"] - 512) * 180 / np.pi
            diff = min(abs(pa2 - pa3), 360 - abs(pa2 - pa3))
            if diff <= 25:
                c2["dual_channel_match"] = {
                    "with": f"{c3['detector']}#{c3['track_index']}",
                    "pa_diff_deg": round(diff, 1),
                }
                break

# --------------------------------------------------------------
# DEBUG IMAGE GENERATION
# --------------------------------------------------------------
def generate_debug_images(det, det_frames, out_dir, all_cands):
    if not DEBUG:
        return

    frame_files = sorted(det_frames.glob("*.*"), key=lambda p: p.name, reverse=True)
    if not frame_files:
        return

    # 1. Last thumb: newest frame
    latest_img = load_image(frame_files[0])
    if latest_img is not None:
        thumb_path = out_dir / f"lastthumb_{det}.png"
        cv2.imwrite(str(thumb_path), latest_img)
        log(f"Generated {thumb_path.name}")

    # 2. Overlay: newest frame + green circles on all midpoints
    overlay_img = latest_img.copy() if latest_img is not None else np.zeros((1024, 1024), dtype=np.uint8)
    for cand in all_cands:
        if cand["detector"] != det or not cand["positions"]:
            continue
        mid_pos = cand["positions"][len(cand["positions"]) // 2]
        cv2.circle(overlay_img, (int(mid_pos["x"]), int(mid_pos["y"])), 6, (0, 255, 0), 2)
    if overlay_img.ndim == 2:
        overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_GRAY2BGR)
    overlay_path = out_dir / f"overlay_{det}.png"
    cv2.imwrite(str(overlay_path), overlay_img)
    log(f"Generated {overlay_path.name}")

    # 3. Contact sheet: 3x3 grid of first 9 frames
    contact_imgs = [load_image(f) for f in frame_files[:9]]
    contact_imgs = [img for img in contact_imgs if img is not None]
    if contact_imgs:
        h, w = contact_imgs[0].shape
        grid_size = int(np.ceil(np.sqrt(len(contact_imgs))))
        contact_h, contact_w = grid_size * h, grid_size * w
        contact = np.zeros((contact_h, contact_w), dtype=np.uint8)
        for i, img in enumerate(contact_imgs):
            row, col = divmod(i, grid_size)
            y1, y2 = row * h, (row + 1) * h
            x1, x2 = col * w, (col + 1) * w
            contact[y1:y2, x1:x2] = img
        contact_path = out_dir / f"contact_{det}.png"
        cv2.imwrite(str(contact_path), contact)
        log(f"Generated {contact_path.name}")

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

    # Download recent images
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
            log(f"No frames found for {det}")
            continue
        cands = detect_in_sequence(det, det_frames, out_dir, args.hours, args.step_min)
        all_cands.extend(cands)

    # Dual-channel matching
    c2_cands = [c for c in all_cands if c["detector"] == "C2"]
    c3_cands = [c for c in all_cands if c["detector"] == "C3"]
    match_dual_channel(c2_cands, c3_cands)

    # Generate debug images (lastthumb, overlay, contact)
    for det in VALID_DETS:
        det_frames = frames_dir / det
        if any(det_frames.glob("*.*")):
            generate_debug_images(det, det_frames, out_dir, all_cands)

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
