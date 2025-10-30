#!/usr/bin/env python3
"""
SOHO Comet Hunter – detector/detect_comets.py
------------------------------------------------
Detects moving objects (comets) in a sequence of SOHO C2/C3 frames.
Now uses CORRECT SOHO URL format with underscore before instrument.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import requests
from filterpy.kalman import KalmanFilter
from scipy.ndimage import gaussian_filter
from typing import Optional

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
MIN_CROP_SIZE = 16
BORDER_WIDTH = 8
MIN_TRACK_LEN = 3
MAX_GAP = 2
IOU_THRESH = 0.3
DEBUG = os.getenv("DETECTOR_DEBUG", "0") == "1"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def download_image(url: str, dest: Path) -> bool:
    """Download a single image, return True on success."""
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            f.write(r.content)
        return True
    except Exception as e:
        if DEBUG:
            print(f"[fetch] failed {url}: {e}")
        return False


def list_soho_frames(instrument: str, hours: int) -> list:
    """
    Generate (timestamp, url) for the last `hours` using CORRECT SOHO path.
    Example:
        https://soho.nascom.nasa.gov/data/REPROCESSING/Completed/2025/c2/20251030/20251030_1048_c2_1024.jpg
    """
    now = datetime.utcnow()
    start = now - timedelta(hours=hours)
    frames = []

    # Start from current time, round down to nearest 12-min interval
    ts = now.replace(second=0, microsecond=0)
    minute = (ts.minute // 12) * 12
    ts = ts.replace(minute=minute)

    while ts >= start:
        stamp = ts.strftime("%Y%m%d_%H%M")        # e.g., 20251030_1048
        year_dir = str(ts.year)                   # 2025
        instr_dir = instrument.lower()            # c2 or c3
        date_dir = stamp[:8]                      # 20251030
        filename = f"{stamp}_{instr_dir}_1024.jpg" # 20251030_1048_c2_1024.jpg

        url = (
            f"https://soho.nascom.nasa.gov/data/REPROCESSING/Completed/"
            f"{year_dir}/{instr_dir}/{date_dir}/{filename}"
        )
        frames.append((ts, url))
        ts -= timedelta(minutes=12)

    return frames


def fetch_frames(instruments, hours, out_dir):
    """
    Download frames. Only append successful downloads.
    """
    frames = {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for instr in instruments:
        frames[instr] = []
        for ts, url in list_soho_frames(instr, hours):
            stamp = ts.strftime("%Y%m%d_%H%M")
            filename = f"{instr}_{stamp}_{instr.lower()}_1024.jpg"  # Local: C2_20251030_1048_c2_1024.jpg
            fpath = out_dir / filename

            if fpath.exists():
                if DEBUG:
                    print(f"[fetch:{instr}] cached {filename}")
                frames[instr].append((ts, str(fpath)))
                continue

            if download_image(url, fpath):
                print(f"[fetch:{instr}] saved {filename}")
                frames[instr].append((ts, str(fpath)))
            else:
                if DEBUG:
                    print(f"[fetch:{instr}] failed {url}")

    return frames


def preprocess(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read {img_path}")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe(img)
    img = gaussian_filter(img, sigma=1)
    return img.astype(np.float32)


def subtract_background(ref: np.ndarray, cur: np.ndarray) -> np.ndarray:
    diff = cv2.absdiff(cur, ref)
    diff = gaussian_filter(diff, sigma=0.5)
    return diff


def find_candidates(diff: np.ndarray, thresh=30) -> list:
    _, thr = cv2.threshold(diff.astype(np.uint8), thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 5 or h < 5:
            continue
        cands.append((x, y, w, h))
    return cands


def crop_image(img: np.ndarray, box) -> np.ndarray:
    x, y, w, h = box
    return img[y:y+h, x:x+w]


def extract_border(crop: np.ndarray, width: int = BORDER_WIDTH) -> np.ndarray:
    h, w = crop.shape
    if h < MIN_CROP_SIZE or w < MIN_CROP_SIZE:
        return np.array([])
    border = np.zeros((h, w), dtype=crop.dtype)
    border[:width, :] = crop[:width, :]
    border[-width:, :] = crop[-width:, :]
    border[width:-width, :width] = crop[width:-width, :width]
    border[width:-width, -width:] = crop[width:-width, -width:]
    return border


def border_stats(border: np.ndarray) -> dict:
    if border.size == 0:
        return {"mean": 0.0, "std": 0.0}
    flat = border.flatten()
    flat = flat[flat > 0]
    return {
        "mean": float(flat.mean()) if flat.size > 0 else 0.0,
        "std": float(flat.std()) if flat.size > 0 else 0.0
    }


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]
    return inter / float(areaA + areaB - inter + 1e-6)


def associate_tracks(prev_tracks, curr_cands):
    new_tracks = []
    used = set()

    for trk in prev_tracks:
        best_iou = 0
        best_idx = -1
        for i, cand in enumerate(curr_cands):
            if i in used:
                continue
            iou_val = iou(trk["bbox"], cand["bbox"])
            if iou_val > best_iou:
                best_iou = iou_val
                best_idx = i

        if best_iou > IOU_THRESH and best_idx != -1:
            cand = curr_cands[best_idx]
            trk["bbox"] = cand["bbox"]
            trk["crop_path"] = cand["crop_path"]
            trk["frame_idx"] = cand["frame_idx"]
            trk["age"] += 1
            trk["missed"] = 0
            trk["kf"].predict()
            trk["kf"].update(np.array([[cand["bbox"][0] + cand["bbox"][2] / 2],
                                       [cand["bbox"][1] + cand["bbox"][3] / 2]]))
            new_tracks.append(trk)
            used.add(best_idx)
        else:
            trk["kf"].predict()
            pred_x = int(trk["kf"].x[0])
            pred_y = int(trk["kf"].x[1])
            trk["bbox"] = (pred_x - trk["bbox"][2] // 2, pred_y - trk["bbox"][3] // 2,
                           trk["bbox"][2], trk["bbox"][3])
            trk["missed"] += 1
            if trk["missed"] <= MAX_GAP:
                new_tracks.append(trk)

    for i, cand in enumerate(curr_cands):
        if i in used:
            continue
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1, 0, 1, 0],
                         [0, 1, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])
        kf.P *= 1000
        kf.R *= 5
        kf.Q *= 0.01
        cx = cand["bbox"][0] + cand["bbox"][2] / 2
        cy = cand["bbox"][1] + cand["bbox"][3] / 2
        kf.x[:2] = np.array([[cx], [cy]])
        new_tracks.append({
            "id": len(prev_tracks) + i,
            "bbox": cand["bbox"],
            "crop_path": cand["crop_path"],
            "frame_idx": cand["frame_idx"],
            "age": 1,
            "missed": 0,
            "kf": kf,
            "history": [cand["bbox"]]
        })
    return new_tracks


def save_candidate(crop, out_dir, ts, box, idx):
    crop_path = Path(out_dir) / f"crop_{ts.strftime('%Y%m%d_%H%M')}_{idx}.jpg"
    cv2.imwrite(str(crop_path), crop)
    return str(crop_path.resolve())


# ----------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------
def detect_in_sequence(detector_frames, out_dir, hours, step_min):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    tracks = []

    # Use newest frame per instrument as reference
    ref = {}
    for instr, flist in detector_frames.items():
        if not flist:
            continue
        latest_path = max(flist, key=lambda x: x[0])[1]
        try:
            ref[instr] = preprocess(latest_path)
        except ValueError as e:
            print(f"[error] Reference failed for {instr}: {e}")
            continue

    if not ref:
        print("No valid reference images.")
        return [], []

    all_frames = []
    for instr, flist in detector_frames.items():
        all_frames.extend([(ts, path, instr) for ts, path in flist])
    all_frames.sort(key=lambda x: x[0])

    for idx, (ts, fpath, instr) in enumerate(all_frames):
        if idx % (step_min // 12) != 0:
            continue

        try:
            cur = preprocess(fpath)
        except ValueError:
            if DEBUG:
                print(f"[skip] Corrupt frame {fpath}")
            continue

        bg = ref.get(instr, cur)
        diff = subtract_background(bg, cur)
        raw_cands = find_candidates(diff)
        frame_cands = []

        for cidx, box in enumerate(raw_cands):
            crop = crop_image(cur, box)
            if crop.size == 0:
                continue

            border = extract_border(crop)
            bstats = border_stats(border)
            if bstats["std"] > 30:
                if DEBUG:
                    print(f"[debug] noisy border std={bstats['std']:.1f}")
                continue

            crop_path = save_candidate(
                crop.astype(np.uint8), out_dir, ts, box, f"{instr}_{cidx}"
            )

            cand = {
                "instrument": instr,
                "timestamp": ts.isoformat(),
                "bbox": box,
                "crop_path": str(Path(crop_path).relative_to(out_dir)),
                "frame_idx": idx,
                "border_mean": bstats["mean"],
                "border_std": bstats["std"]
            }
            frame_cands.append(cand)

        tracks = associate_tracks(tracks, frame_cands)
        candidates.extend(frame_cands)

    good_tracks = [t for t in tracks if t["age"] >= MIN_TRACK_LEN]
    return candidates, good_tracks


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SOHO Comet Hunter detector")
    parser.add_argument("--hours", type=int, default=6, help="Look-back hours")
    parser.add_argument("--step-min", type=int, default=12, help="Step between frames (minutes)")
    parser.add_argument("--out", type=str, default="detections", help="Output directory")
    parser.add_argument("--instruments", nargs="+", default=["C2", "C3"],
                        help="Instruments to monitor")
    args = parser.parse_args()

    frames_dir = Path("frames")
    frames_dir.mkdir(exist_ok=True)

    print(f"Downloading frames for the last {args.hours} h...")
    detector_frames = fetch_frames(args.instruments, args.hours, frames_dir)

    total = sum(len(v) for v in detector_frames.values())
    print(f"[fetch] summary: saved {total} file(s)")

    valid_frames = {k: v for k, v in detector_frames.items() if v}
    if not valid_frames:
        print("No frames downloaded. Exiting.")
        return

    print("Running detection sequence...")
    cands, tracks = detect_in_sequence(valid_frames, args.out, args.hours, args.step_min)

    # Save results
    out_path = Path(args.out) / f"candidates_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(cands, f, indent=2)

    track_path = Path(args.out) / f"tracks_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.json"
    with open(track_path, "w") as f:
        json.dump([{
            "id": t["id"],
            "length": t["age"],
            "instrument": t["history"][0][0] if t["history"] else "unknown",
            "bboxes": t["history"]
        } for t in tracks], f, indent=2)

    print(f"Detection complete -> {len(cands)} candidates, {len(tracks)} tracks")
    print(f"Results in {args.out}")


if __name__ == "__main__":
    main()
