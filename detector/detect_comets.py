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
USE_AI = os.getenv("USE_AI_CLASSIFIER", "1") == "1"  # ← NOW DEFAULT ON
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
    try:
        dt = datetime.strptime(name.split("_")[1].split(".")[0], "%Y%m%d%H%M%S")
        return dt.isoformat() + "Z"
    except:
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
            log(f"GIF save failed: {e}")

    h, w = annot[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    try:
        w1 = cv2.VideoWriter(str(base_a.with_suffix(".mp4")), fourcc, fps, (w, h))
        for f in annot:
            w1.write(f)
        w1.release()
        out["animation_mp4_path"] = str(base_a.with_suffix(".mp4"))
    except Exception as e:
        log(f"MP4 annotated failed: {e}")
    try:
        w2 = cv2.VideoWriter(str(base_c.with_suffix(".mp4")), fourcc, fps, (w, h))
        for f in clean:
            w2.write(f)
        w2.release()
        out["animation_mp4_clean_path"] = str(base_c.with_suffix(".mp4"))
    except Exception as e:
        log(f"MP4 clean failed: {e}")

    return out

# --------------------------------------------------------------
# AI CLASSIFIER (REAL MODEL STUB — REPLACE WITH YOUR MODEL)
# --------------------------------------------------------------
def classify_crop(crop):
    """
    Replace this with your real AI model (TensorFlow, PyTorch, ONNX, etc.)
    Input: crop (numpy array, grayscale)
    Output: {"label": "comet" or "not_comet", "score": 0.0 to 1.0}
    """
    if not USE_AI:
        return {"label": "unknown", "score": 0.0}

    # === PLACE YOUR REAL MODEL HERE ===
    # Example: ONNX model
    # import onnxruntime as ort
    # sess = ort.InferenceSession("comet_classifier.onnx")
    # input_name = sess.get_inputs()[0].name
    # pred = sess.run(None, {input_name: crop.astype(np.float32)[None, None, :, :]})
    # score = float(pred[0][0])
    # label = "comet" if score > 0.5 else "not_comet"

    # === TEMP FALLBACK: Smart heuristic (bright + motion) ===
    if crop.size == 0:
        return {"label": "not_comet", "score": 0.0}

    # Bright blob in center?
    h, w = crop.shape
    center = crop[h//4:3*h//4, w//4:3*w//4]
    if center.size == 0:
        return {"label": "not_comet", "score": 0.0}
    brightness = np.mean(center)
    contrast = np.std(center)

    # Motion track length proxy (passed in via global or closure if needed)
    # For now: assume long tracks are comets
    score = min(0.99, (brightness / 255) * 0.6 + (contrast / 50) * 0.4)
    label = "comet" if score > 0.6 else "not_comet"
    return {"label": label, "score": round(score, 3)}

# --------------------------------------------------------------
# DETECTION CORE
# --------------------------------------------------------------
def detect_in_sequence(det, frames_dir, out_dir, hours=6, step_min=12):
    frames_dir = Path(frames_dir)
    out_dir = ensure_dir(out_dir)
    report_dir = ensure_dir(out_dir / "reports")
    crops_dir = ensure_dir(out_dir / "crops")

    img_paths = sorted([p for p in frames_dir.glob(f"*{det}*.jpg")])
    if not img_paths:
        log(f"No {det} frames in {frames_dir}")
        return []

    names = [p.name for p in img_paths]
    imgs = [load_image(str(p)) for p in img_paths]
    if any(i is None for i in imgs):
        return []

    target_w, target_h = IMG_SIZE[det]
    imgs = [cv2.resize(i, (target_w, target_h)) for i in imgs]

    processed = [gaussian_filter(img.astype(float), sigma=1.5) for img in imgs]
    diffs = []
    for i in range(1, len(processed)):
        diff = cv2.absdiff(processed[i], processed[i - 1])
        diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
        diffs.append((diff * 255).astype(np.uint8))

    tracks = []
    min_area = 3
    for t, diff in enumerate(diffs):
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2
            tracks.append((t + 1, cx, cy, area))

    # Kalman tracking
    def init_kf(x, y):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.x[:2] = np.array([x, y])
        kf.F = np.array([[1, 0, 1, 0],
                         [0, 1, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])
        kf.P *= 1000
        kf.R *= 5
        kf.Q *= 0.01
        return kf

    active = {}
    completed = []

    for t, x, y, area in tracks:
        matched = None
        best_dist = float("inf")
        for tid, kf in list(active.items()):
            pred = kf.predict()
            dist = np.linalg.norm(pred[:2] - [x, y])
            if dist < 50 and dist < best_dist:
                best_dist = dist
                matched = tid
        if matched is not None:
            active[matched].update(np.array([[x], [y]]))
            active[matched].points.append((t, x, y, area))
        else:
            tid = len(active) + len(completed)
            kf = init_kf(x, y)
            kf.points = [(t, x, y, area)]
            active[tid] = kf

        expired = [tid for tid, kf in active.items() if t - kf.points[-1][0] > 5]
        for tid in expired:
            completed.append(active.pop(tid).points)
    completed.extend([kf.points for kf in active.values()])

    valid_tracks = [tr for tr in completed if len(tr) >= 3]
    candidates = []

    for idx, tr in enumerate(valid_tracks):
        xs = [p[1] for p in tr]
        ys = [p[2] for p in tr]
        dist = np.hypot(xs[-1] - xs[0], ys[-1] - ys[0])
        speed = dist / max(1, len(tr) - 1)
        score = len(tr) * (1 + speed / 50)

        mid_t = tr[len(tr)//2][0]
        mid_img = imgs[mid_t]
        x0, y0 = int(min(xs)), int(min(ys))
        x1, y1 = int(max(xs)), int(max(ys))
        pad = 32
        crop = mid_img[max(0, y0-pad):min(target_h, y1+pad),
                       max(0, x0-pad):min(target_w, x1+pad)]
        crop_path = crops_dir / f"{det}_track{idx}_crop.png"
        cv2.imwrite(str(crop_path), crop)

        # === AI CLASSIFICATION (NOW ALWAYS RUNS) ===
        ai = classify_crop(crop)
        if AI_VETO and ai["label"] == AI_VETO_LABEL and ai["score"] > AI_VETO_MAX:
            log(f"Vetoed track {idx}: {ai['label']} ({ai['score']:.3f})")
            continue

        # === ANIMATION ===
        anim_paths = write_animation_for_track(det, names, imgs, tr, out_dir)

        # === BUILD CANDIDATE ===
        cand = {
            "detector": det,
            "track_index": idx,
            "score": round(score, 2),
            "positions": [
                {
                    "time_utc": timestamp_from_name(names[t]),
                    "x": round(x, 1),
                    "y": round(y, 1),
                } for t, x, y, _ in tr
            ],
            "series_mid_frame": names[mid_t],
            "crop_path": str(crop_path),
            "image_size": [target_w, target_h],
            "origin": "upper_left",
            "original_mid_path": f"detections/originals/{names[mid_t]}",
            "annotated_mid_path": f"detections/annotated/{names[mid_t]}_annot_track.png",
            "ai_label": ai["label"],
            "ai_score": ai["score"],
            "auto_selected": True  # ← keep your logic
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

    now = datetime.utcnow()
    for det in VALID_DETS:
        det_dir = frames_dir / det
        ensure_dir(det_dir)
        count = 0
        for mins in range(0, args.hours * 60, args.step_min):
            dt = now - timedelta(minutes=mins)
            url = f"{BASE_URL}/{dt.year}/{det}/img/jpg/{dt.strftime('%Y%m%d')}/{det}_{dt.strftime('%Y%m%d_%H%M%S')}.jpg"
            path = det_dir / f"{det}_{dt.strftime('%Y%m%d_%H%M%S')}.jpg"
            if not path.exists():
                if download_file(url, path):
                    count += 1
        log(f"Downloaded {count} {det} frames")

    all_cands = []
    for det in VALID_DETS:
        det_frames = frames_dir / det
        if not any(det_frames.glob("*.jpg")):
            continue
        cands = detect_in_sequence(det, det_frames, out_dir, args.hours, args.step_min)
        all_cands.extend(cands)

    c2_cands = [c for c in all_cands if c["detector"] == "C2"]
    c3_cands = [c for c in all_cands if c["detector"] == "C3"]
    match_dual_channel(c2_cands, c3_cands)

    timestamp = now.strftime("%Y%m%d_%H%M%S")
    report_path = out_dir / f"candidates_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(all_cands, f, indent=2)

    status = {
        "timestamp_utc": now.isoformat() + "Z",
        "detectors": {
            det: {
                "frames": len(list((frames_dir / det).glob("*.jpg"))),
                "tracks": len([c for c in all_cands if c["detector"] == det]),
            } for det in VALID_DETS
        }
    }
    with open(out_dir / "latest_status.json", "w") as f:
        json.dump(status, f, indent=2)

    log(f"Detected {len(all_cands)} candidates → {report_path}")

if __name__ == "__main__":
    main()
