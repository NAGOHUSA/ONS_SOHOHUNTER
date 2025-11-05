#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SOHO Comet Detection Pipeline (LATEST-driven, no extra heavy deps)

- Parses NASA LATEST pages for LASCO C2/C3
- Downloads frames for the last --hours in --step-min cadence
- Simple centroid-based motion tracker (no SciPy/FilterPy)
- Optional local AI crop classification via ai_classifier.py
- Writes:
    detections/latest_status.json   # includes "candidates": [...]
    detections/latest_run.json
    detections/lastthumb_C2.png
    detections/lastthumb_C3.png
- When any found:
    detections/candidates_YYYYMMDD_HHMMSS.json
"""

from __future__ import annotations
import os, re, io, json, argparse, base64
from pathlib import Path
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import requests

# ------------------------------------------------------------------------------------
# CLI / ENV
# ------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--hours", type=int, default=int(os.getenv("HOURS", "6")))
parser.add_argument("--step-min", type=int, default=int(os.getenv("STEP_MIN", "12")))
parser.add_argument("--out", default=os.getenv("OUT", "detections"))
args = parser.parse_args()

HOURS       = args.hours
STEP_MIN    = args.step_min
OUT_DIR     = Path(args.out)
FRAMES_DIR  = Path("frames")

DETECTOR_DEBUG   = os.getenv("DETECTOR_DEBUG", "0") == "1"
GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "")

# NASA LATEST pages (most up-to-date)
LATEST_PAGES = {
    "C2": "https://soho.nascom.nasa.gov/data/LATEST/latest-lascoC2.html",
    "C3": "https://soho.nascom.nasa.gov/data/LATEST/latest-lascoC3.html",
}

IMG_HREF_RE = re.compile(r'href="([^"]*_(c2|c3)_1024\.jpg)"', re.IGNORECASE)
STAMP_RE    = re.compile(r"/(\d{8})_(\d{4})_c[23]_1024\.jpg$", re.IGNORECASE)

def log(*a): 
    print(*a, flush=True)

# ------------------------------------------------------------------------------------
# Optional local AI classifier
# ------------------------------------------------------------------------------------
try:
    from ai_classifier import classify_crop_batch  # returns [{"label": "...", "score": 0.0}, ...]
except Exception as e:
    log("[ai] Could not import ai_classifier.classify_crop_batch — falling back. Reason:", e)
    def classify_crop_batch(paths):
        # Neutral defaults
        return [{"label": "unknown", "score": 0.0} for _ in paths]

# ------------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------------
def ensure_dirs():
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "crops").mkdir(parents=True, exist_ok=True)

def fetch_latest_list(instr: str):
    """Return list of (absolute_jpg_url, ymd, hm, iso) for instrument."""
    url = LATEST_PAGES[instr]
    try:
        html = requests.get(url, timeout=20).text
    except Exception as e:
        log(f"[{instr}] LATEST fetch failed:", e)
        return []

    items = []
    for m in IMG_HREF_RE.finditer(html):
        href = m.group(1)
        if href.startswith("//"):
            jpg_url = "https:" + href
        elif href.startswith("http"):
            jpg_url = href
        else:
            jpg_url = ("https://soho.nascom.nasa.gov" + href) if href.startswith("/") else ("https://soho.nascom.nasa.gov/data/" + href)

        sm = STAMP_RE.search(jpg_url)
        if not sm:
            continue
        ymd, hm = sm.group(1), sm.group(2)
        iso = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}T{hm[:2]}:{hm[2:]}:00Z"
        items.append((jpg_url, ymd, hm, iso))

    # Unique & sorted
    seen = set()
    uniq = []
    for it in items:
        if it[0] in seen: 
            continue
        seen.add(it[0])
        uniq.append(it)
    uniq.sort(key=lambda x: x[3])
    return uniq

def within_window(iso_str: str, now_utc):
    t = datetime.strptime(iso_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return (now_utc - t) <= timedelta(hours=HOURS)

def matches_step(prev_iso: str, cur_iso: str):
    """Roughly STEP_MIN between frames (±2 min)."""
    if not prev_iso: 
        return True
    p = datetime.strptime(prev_iso, "%Y-%m-%dT%H:%M:%SZ")
    c = datetime.strptime(cur_iso, "%Y-%m-%dT%H:%M:%SZ")
    delta_min = abs((c - p).total_seconds() / 60.0 - STEP_MIN)
    return delta_min <= 2.0

def download_frames_from_latest(instr: str):
    """Download frames guided by LATEST page, honoring window + cadence."""
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    catalog = fetch_latest_list(instr)
    kept, kept_iso = [], []
    last_kept = ""

    for jpg_url, ymd, hm, iso in catalog:
        if not within_window(iso, now):
            continue
        if not matches_step(last_kept, iso):
            continue

        fname = f"{instr}_{ymd}_{hm}.jpg"
        out = FRAMES_DIR / fname
        if not out.exists():
            try:
                r = requests.get(jpg_url, timeout=30)
                if r.status_code == 200:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(r.content)
                else:
                    continue
            except Exception as e:
                log(f"[{instr}] download failed {jpg_url}:", e)
                continue

        kept.append(str(out))
        kept_iso.append(iso)
        last_kept = iso

    return kept, kept_iso

# ------------------------------------------------------------------------------------
# Simple tracker (centroid association)
# ------------------------------------------------------------------------------------
def detect_points(gray: np.ndarray) -> list[tuple[int,int]]:
    """Return list of bright centroids."""
    # denoise & enhance
    blur = cv2.GaussianBlur(gray, (0,0), 1.0)
    # adaptive threshold: Otsu
    _t, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # keep small-ish blobs
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = []
    for c in contours:
        area = cv2.contourArea(c)
        if 5 <= area <= 400:   # heuristic for small moving points
            M = cv2.moments(c)
            if M["m00"] > 0:
                x = int(M["m10"]/M["m00"])
                y = int(M["m01"]/M["m00"])
                pts.append((x,y))
    return pts

def link_tracks(frames: list[str], times: list[str], max_dist: float = 80.0, min_len: int = 4):
    """
    Naive nearest-neighbor track linker across frames.
    Returns list of tracks: dict(id, positions=[{x,y,timestamp}])
    """
    tracks = []  # list of dict: {"id": int, "positions": [{"x":..,"y":..,"timestamp":..}, ...]}
    last_positions = {}  # track_id -> (x,y)

    next_id = 0
    for i,(fpath, ts) in enumerate(sorted(zip(frames, times), key=lambda p: p[1])):
        im = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        pts = detect_points(im)

        # Try to match pts to existing tracks by nearest neighbor
        assigned = set()
        for tid, (lx,ly) in list(last_positions.items()):
            best_j, best_d2 = -1, 1e18
            for j,(x,y) in enumerate(pts):
                if j in assigned: 
                    continue
                d2 = (x-lx)*(x-lx) + (y-ly)*(y-ly)
                if d2 < best_d2:
                    best_d2, best_j = d2, j
            if best_j >= 0 and best_d2 <= max_dist*max_dist:
                # append to that track
                for tr in tracks:
                    if tr["id"] == tid:
                        tr["positions"].append({"x": int(pts[best_j][0]), "y": int(pts[best_j][1]), "timestamp": ts})
                        last_positions[tid] = pts[best_j]
                        assigned.add(best_j)
                        break
            else:
                # If no match, drop from last_positions (track might have ended)
                last_positions.pop(tid, None)

        # Spawn new tracks for unassigned pts
        for j,(x,y) in enumerate(pts):
            if j in assigned:
                continue
            tr = {"id": next_id, "positions": [{"x": int(x), "y": int(y), "timestamp": ts}]}
            tracks.append(tr)
            last_positions[next_id] = (x,y)
            next_id += 1

    # Keep only tracks with length >= min_len
    long_tracks = [t for t in tracks if len(t["positions"]) >= min_len]
    return long_tracks

def make_crop(im_gray: np.ndarray, x: int, y: int, size: int = 64) -> np.ndarray:
    h,w = im_gray.shape
    r = size // 2
    x0, y0 = max(0, x - r), max(0, y - r)
    x1, y1 = min(w, x + r), min(h, y + r)
    crop = im_gray[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    # pad to exact size for consistency
    pad = cv2.copyMakeBorder(crop, 
                             top = 0, 
                             bottom = size - crop.shape[0], 
                             left = 0, 
                             right = size - crop.shape[1],
                             borderType = cv2.BORDER_CONSTANT, value = 0)
    return pad

def build_candidates(instr: str, frames: list[str], times: list[str], tracks: list[dict]):
    """Produce candidate dicts with crops and AI scores."""
    candidates = []
    if not tracks:
        return candidates

    # Index frames by timestamp for quick access
    frame_by_ts = {ts: fp for fp, ts in sorted(zip(frames, times), key=lambda p: p[1])}

    for tr in tracks:
        # use the last observation for crop
        pos_last = tr["positions"][-1]
        ts = pos_last["timestamp"]
        fp = frame_by_ts.get(ts)
        if not fp:
            continue
        im = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        crop = make_crop(im, pos_last["x"], pos_last["y"], 64)
        if crop is None:
            continue

        # save crop
        crops_dir = OUT_DIR / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        cname = f"{instr}_track{tr['id']}_{ts.replace(':','').replace('-','')}.png"
        cpath = crops_dir / cname
        cv2.imwrite(str(cpath), crop)

        # classify
        res = classify_crop_batch([str(cpath)])[0] if callable(classify_crop_batch) else {"label":"unknown","score":0.0}
        ai_label = str(res.get("label","unknown")).lower()
        ai_score = float(res.get("score", 0.0))

        cand = {
            "instrument": f"LASCO {instr}",
            "timestamp": ts,
            "track_id": tr["id"],
            "positions": tr["positions"],  # series for Sungrazer export
            "bbox": [int(pos_last["x"] - 32), int(pos_last["y"] - 32), 64, 64],
            "crop_path": f"crops/{cname}",
            "ai_label": ai_label,
            "ai_score": round(ai_score, 3),
            # placeholders for UI toggle compatibility
            "original_mid_path": None,
            "annotated_mid_path": None
        }
        candidates.append(cand)

    return candidates

# ------------------------------------------------------------------------------------
# Status writers
# ------------------------------------------------------------------------------------
def write_lastthumb(img_path: Path, out_png: Path):
    try:
        im = cv2.imread(str(img_path))
        if im is None:
            return
        h,w = im.shape[:2]
        scale = 320.0 / max(h,w)
        imr = cv2.resize(im, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_png), imr)
    except Exception as e:
        log("[thumb]", e)

def write_status_payload(c2_frames, c3_frames, c2_last, c3_last, c2_times, c3_times, candidates):
    ts_iso = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z")
    payload = {
        "name": "latest_status.json",
        "generated_at": ts_iso,
        "timestamp_utc": ts_iso,
        "hours_back": HOURS,
        "step_min": STEP_MIN,
        "c2_frames": len(c2_frames),
        "c3_frames": len(c3_frames),
        "c2_last_frame": Path(c2_last).name if c2_last else "",
        "c3_last_frame": Path(c3_last).name if c3_last else "",
        "time_range": {
            "C2": f"{(min(c2_times) if c2_times else 'none')} → {(max(c2_times) if c2_times else 'none')}",
            "C3": f"{(min(c3_times) if c3_times else 'none')} → {(max(c3_times) if c3_times else 'none')}",
        },
        "errors": [],
        "auto_selected_count": len(candidates),
        "candidates": candidates,
    }
    (OUT_DIR / "latest_status.json").write_text(json.dumps(payload, indent=2))

def write_run_log(c2n: int, c3n: int, candn: int):
    ts_iso = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z")
    (OUT_DIR / "latest_run.json").write_text(json.dumps({
        "last_run_utc": ts_iso,
        "frames_downloaded": {"C2": c2n, "C3": c3n},
        "candidates_found": candn
    }, indent=2))

# ------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------
def main():
    log("=== DETECTION START ===")
    ensure_dirs()

    # Download from LATEST
    c2_paths, c2_times = download_frames_from_latest("C2")
    c3_paths, c3_times = download_frames_from_latest("C3")
    c2_cnt, c3_cnt = len(c2_paths), len(c3_paths)

    if DETECTOR_DEBUG:
        log(f"[DEBUG] C2 frames: {c2_cnt}, C3 frames: {c3_cnt}")

    # Track & classify
    c2_tracks = link_tracks(c2_paths, c2_times, max_dist=80.0, min_len=4)
    c3_tracks = link_tracks(c3_paths, c3_times, max_dist=80.0, min_len=4)

    c2_cands = build_candidates("C2", c2_paths, c2_times, c2_tracks)
    c3_cands = build_candidates("C3", c3_paths, c3_times, c3_tracks)
    candidates = c2_cands + c3_cands

    # Save timestamped candidates if any
    if candidates:
        ts_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        (OUT_DIR / f"candidates_{ts_str}.json").write_text(json.dumps(candidates, indent=2))
        log(f"Saved {len(candidates)} candidates → detections/candidates_{ts_str}.json")

    # Write last thumbnails
    c2_last = c2_paths[-1] if c2_paths else ""
    c3_last = c3_paths[-1] if c3_paths else ""
    if c2_last: write_lastthumb(Path(c2_last), OUT_DIR / "lastthumb_C2.png")
    if c3_last: write_lastthumb(Path(c3_last), OUT_DIR / "lastthumb_C3.png")

    # Status + run log
    write_status_payload(c2_paths, c3_paths, c2_last, c3_last, c2_times, c3_times, candidates)
    write_run_log(c2_cnt, c3_cnt, len(candidates))

    tracks = len({c["track_id"] for c in candidates})
    log(f"=== DONE ===  C2:{c2_cnt}  C3:{c3_cnt}  Candidates:{len(candidates)}  Tracks:{tracks}")

if __name__ == "__main__":
    main()
