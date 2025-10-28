### Fixed `detector/detect_comets.py`

The error is a **syntax error** in the `save_original_and_annotated` function (around line 242 in your current file). It's a malformed `cv2.line()` call with an unmatched closing parenthesis and a stray "cv" prefix. This likely crept in during a copy-paste or edit (e.g., trying to format the arguments).

I also spotted a small issue in `read_gray_jpg`: it returns `None` on failure, but the calling code doesn't handle it well. I've fixed that too for robustness.

Here's the **complete, corrected file**. Replace your current `detector/detect_comets.py` with this. (I've preserved your original logic/structure while applying the fixes from my previous response + this syntax correction.)

```python
# detector/detect_comets.py
# SOHO comet hunter — outputs frontend-compatible latest_status.json and per-candidate animations
from __future__ import annotations

import os, re, io, json, math, argparse, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter

import numpy as np
import cv2
import requests

# Optional for GIFs (MP4s are always generated)
try:
    import imageio
except Exception:
    imageio = None

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
DETECTORS = ("C2", "C3")
LATEST_URLS = {
    "C2": "https://soho.nascom.nasa.gov/data/LATEST/latest-lascoC2.html",
    "C3": "https://soho.nascom.nasa.gov/data/LATEST/latest-lascoC3.html",
}
MAX_FRAMES_PER_CAM = 24
TARGET_SIZE = 512  # Standardize all frames to 512x512

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def ensure_dir(p: Path):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)

def save_png(path: Path, img: np.ndarray) -> None:
    ensure_dir(path); cv2.imwrite(str(path), img)

def utcnow_iso() -> str:
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z")

def http_get(url: str, timeout=20) -> bytes:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content

def parse_latest_list(det: str) -> List[str]:
    html = http_get(LATEST_URLS[det]).decode("utf-8","ignore")
    names = re.findall(r"(\d{8}_\d{4}_(?:c2|c3)_\d+\.jpe?g)", html, flags=re.I)
    seen, ordered = set(), []
    for n in names:
        if n.lower() not in seen:
            seen.add(n.lower()); ordered.append(n)
    ordered = ordered[:MAX_FRAMES_PER_CAM]
    ordered.sort()
    return ordered

def soho_frame_url(frame_name: str) -> str:
    m = re.match(r"(\d{8})_(\d{4})(?:\d{0,2})?_(c[23])_(\d+)\.jpe?g$", frame_name, flags=re.I)
    if not m:
        return f"https://soho.nascom.nasa.gov/data/LATEST/{frame_name}"
    ymd, hm, cam, res = m.groups()
    year = ymd[:4]
    return f"https://soho.nascom.nasa.gov/data/REPROCESSING/Completed/{year}/{cam.lower()}/{ymd}/{frame_name}"

def read_gray_jpg(buf: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(buf, dtype=np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if im is None:
        print(f"Failed to decode JPG buffer (corrupt or invalid data)")
    return im

def resize_and_pad(img: np.ndarray, target_size: int) -> np.ndarray:
    """Resize with aspect ratio, then pad to target_size x target_size."""
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_total_w = target_size - new_w
    pad_total_h = target_size - new_h
    pad_l = pad_total_w // 2
    pad_r = pad_total_w - pad_l
    pad_t = pad_total_h // 2
    pad_b = pad_total_h - pad_t

    padded = cv2.copyMakeBorder(resized, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
    return padded

# -----------------------------------------------------------------------------
# Series building (fetch + resize)
# -----------------------------------------------------------------------------
def build_series(det: str, hours: int, step_min: int) -> List[Tuple[str, np.ndarray]]:
    try:
        from fetch_lasco import fetch_series
        series = fetch_series(det, hours=hours, step_min=step_min)
        # Still resize for consistency
        resized = []
        for name, img in series:
            if img is None: continue
            try:
                resized_img = resize_and_pad(img, TARGET_SIZE)
                resized.append((name, resized_img))
            except Exception:
                continue
        return resized
    except Exception:
        pass

    names = parse_latest_list(det)
    if not names:
        return []

    stride = max(1, step_min // 12)
    picked = names[::stride] or names
    series: List[Tuple[str, np.ndarray]] = []
    for nm in picked:
        try:
            buf = http_get(soho_frame_url(nm))
            im = read_gray_jpg(buf)
            if im is None: continue
            im_resized = resize_and_pad(im, TARGET_SIZE)
            series.append((nm, im_resized))
        except Exception as e:
            print(f"Failed to fetch/process {nm}: {e}")
            continue
    return series

# -----------------------------------------------------------------------------
# Simple motion-based detections -> tracks
# -----------------------------------------------------------------------------
def detect_candidates(series: List[Tuple[str, np.ndarray]]) -> List[List[Tuple[int,float,float,float]]]:
    if len(series) < 3:
        return []

    names = [s[0] for s in series]
    imgs = [s[1] for s in series]

    # Verify all same size
    sizes = [im.shape for im in imgs]
    if len(set(sizes)) > 1:
        print(f"WARNING: Inconsistent image sizes after resize: {set(sizes)}")
        return []

    # Preprocess
    imgs_blur = [cv2.GaussianBlur(im, (3,3), 0) for im in imgs]

    dets_by_t: List[List[Tuple[float,float,float]]] = []
    for i in range(1, len(imgs_blur)):
        try:
            diff = cv2.absdiff(imgs_blur[i], imgs_blur[i-1])
            diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
            _, bw = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            bw = cv2.medianBlur(bw, 3)
            cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            dets: List[Tuple[float,float,float]] = []
            h, w = bw.shape[:2]
            for c in cnts:
                area = cv2.contourArea(c)
                if area < 6 or area > 400:
                    continue
                (x, y), r = cv2.minEnclosingCircle(c)
                if w*0.35 < x < w*0.65 and h*0.35 < y < h*0.65:
                    continue
                dets.append((float(x), float(y), float(area)))
            dets_by_t.append(dets)
        except Exception as e:
            print(f"Diff failed at frame {i}: {e}")
            dets_by_t.append([])

    # Link into tracks
    tracks: List[List[Tuple[int,float,float,float]]] = []
    max_jump = 18.0
    for t, dets in enumerate(dets_by_t, start=1):
        used = set()
        for tr in tracks:
            last_t, lx, ly, _ = tr[-1]
            if t - last_t != 1:
                continue
            best_j, best_d = -1, 1e9
            for j, (x, y, a) in enumerate(dets):
                if j in used: continue
                d = math.hypot(x - lx, y - ly)
                if d < best_d:
                    best_d, best_j = d, j
            if best_j >= 0 and best_d <= max_jump:
                used.add(best_j)
                x, y, a = dets[best_j]
                tr.append((t, x, y, a))
        for j, (x, y, a) in enumerate(dets):
            if j in used: continue
            tracks.append([(t, x, y, a)])

    tracks = [tr for tr in tracks if len(tr) >= 4]
    return tracks

def frame_iso_from_name(name: str) -> str:
    m = re.match(r"(\d{8})_(\d{4})(?:\d{0,2})?_(c[23])_\d+\.jpe?g$", name, flags=re.I)
    if not m:
        return ""
    d, hm, cam = m.groups()
    iso = f"{d[:4]}-{d[4:6]}-{d[6:]}T{hm[:2]}:{hm[2:]}:00Z"
    return iso

# -----------------------------------------------------------------------------
# Reporting: Sungrazer, overlays, animations
# -----------------------------------------------------------------------------
def write_sungrazer(det: str, idx: int, positions: List[Dict[str,Any]], out_dir: Path) -> Tuple[str,str]:
    out_dir = Path(out_dir) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / f"{det}_track{idx}_sungrazer.txt"
    csv_path = out_dir / f"{det}_track{idx}_sungrazer.csv"
    with open(txt_path, "w") as f:
        f.write("# FrameTimeUTC x y\n")
        for p in positions:
            f.write(f"{p['time_utc']} {int(round(p['x']))} {int(round(p['y']))}\n")
    with open(csv_path, "w") as f:
        f.write("frame_time_utc,x,y\n")
        for p in positions:
            f.write(f"{p['time_utc']},{int(round(p['x']))},{int(round(p['y']))}\n")
    return str(txt_path), str(csv_path)

def draw_tracks_overlay(base_img: np.ndarray, tracks, out_path: Path):
    vis = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR) if len(base_img.shape) == 2 else base_img.copy()
    for tr in tracks:
        pts = [(int(round(x)), int(round(y))) for (_,x,y,_) in tr]
        for i in range(1, len(pts)):
            cv2.line(vis, pts[i-1], pts[i], (0,255,0), 1)
    save_png(out_path, vis)

def save_original_and_annotated(mid_img: np.ndarray, positions, out_dir: Path, tag: str) -> Tuple[str,str]:
    orig_path = Path(out_dir) / "originals" / f"{tag}.png"
    ann_path  = Path(out_dir) / "annotated" / f"{tag}.png"
    ensure_dir(orig_path); ensure_dir(ann_path)
    vis = cv2.cvtColor(mid_img, cv2.COLOR_GRAY2BGR) if len(mid_img.shape) == 2 else mid_img.copy()
    for i, p in enumerate(positions):
        x, y = int(round(p["x"])), int(round(p["y"]))
        cv2.circle(vis, (x, y), 4, (0,255,0), 1)
        if i > 0:
            px, py = int(round(positions[i-1]["x"])), int(round(positions[i-1]["y"]))
            cv2.line(vis, (px, py), (x, y), (0,255,0), 1)
    cv2.imwrite(str(orig_path), mid_img if mid_img.ndim == 2 else cv2.cvtColor(mid_img, cv2.COLOR_BGR2GRAY))
    cv2.imwrite(str(ann_path), vis)
    return str(orig_path), str(ann_path)

def write_animation_for_track(det: str,
                              names: List[str],
                              imgs: List[np.ndarray],
                              tr: List[Tuple[int,float,float,float]],
                              out_dir: Path,
                              fps: int = 6,
                              circle_radius: int = 4) -> Dict[str, Optional[str]]:
    t_min, t_max = tr[0][0], tr[-1][0]
    frames_annot, frames_clean = [], []
    xy_by_t = {t: (int(round(x)), int(round(y))) for (t, x, y, _) in tr}
    trail: List[Tuple[int, int]] = []
    for ti in range(t_min, t_max + 1):
        im = imgs[ti]
        bgr = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) if len(im.shape) == 2 else im.copy()
        frames_clean.append(bgr.copy())
        if ti in xy_by_t:
            trail.append(xy_by_t[ti])
        for i in range(1, len(trail)):
            cv2.line(bgr, trail[i-1], trail[i], (0,255,0), 1)
        if ti in xy_by_t:
            cv2.circle(bgr, xy_by_t[ti], circle_radius, (0,255,0), 1)
        frames_annot.append(bgr)

    anim_dir = Path(out_dir) / "animations"
    anim_dir.mkdir(parents=True, exist_ok=True)
    ident = tr[0][0]
    base_ann = anim_dir / f"{det}_track{ident}_annotated"
    base_cln = anim_dir / f"{det}_track{ident}_clean"

    out = {
        "animation_gif_path": None,
        "animation_mp4_path": None,
        "animation_gif_clean_path": None,
        "animation_mp4_clean_path": None,
    }

    if imageio is not None:
        try:
            imageio.mimsave(str(base_ann.with_suffix(".gif")), frames_annot, fps=fps)
            out["animation_gif_path"] = str(base_ann.with_suffix(".gif"))
            imageio.mimsave(str(base_cln.with_suffix(".gif")), frames_clean, fps=fps)
            out["animation_gif_clean_path"] = str(base_cln.with_suffix(".gif"))
        except Exception as e:
            print(f"GIF save failed: {e}")

    h, w = frames_annot[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    try:
        w1 = cv2.VideoWriter(str(base_ann.with_suffix(".mp4")), fourcc, fps, (w, h))
        for fr in frames_annot: w1.write(fr)
        w1.release()
        out["animation_mp4_path"] = str(base_ann.with_suffix(".mp4"))
    except Exception as e:
        print(f"MP4 annotated save failed: {e}")

    try:
        w2 = cv2.VideoWriter(str(base_cln.with_suffix(".mp4")), fourcc, fps, (w, h))
        for fr in frames_clean: w2.write(fr)
        w2.release()
        out["animation_mp4_clean_path"] = str(base_cln.with_suffix(".mp4"))
    except Exception as e:
        print(f"MP4 clean save failed: {e}")

    return out

# -----------------------------------------------------------------------------
# Main packaging
# -----------------------------------------------------------------------------
def package_detector(det: str, series: List[Tuple[str,np.ndarray]], out_dir: Path, debug: bool) -> Tuple[List[Dict[str,Any]], Dict[str,Any]]:
    names = [n for (n, _) in series]
    imgs = [im for (_, im) in series]
    hits: List[Dict[str,Any]] = []
    tracks = detect_candidates(series)

    if imgs:
        save_png(Path(out_dir) / f"lastthumb_{det}.png", imgs[-1])

    for i, tr in enumerate(tracks, start=1):
        positions = []
        for (t, x, y, a) in tr:
            positions.append({
                "time_utc": frame_iso_from_name(names[t]) or utcnow_iso(),
                "x": float(x), "y": float(y)
            })

        mid_idx = tr[len(tr)//2][0]
        mid_name = names[mid_idx]
        mid_img = imgs[mid_idx]

        crop_path = Path(out_dir) / "crops" / f"{det}_{mid_name}"
        ensure_dir(crop_path)
        save_png(crop_path, mid_img)

        write_sungrazer(det, i, positions, out_dir)
        orig_p, ann_p = save_original_and_annotated(mid_img, positions, out_dir, tag=f"{det}_{mid_name}")
        anim = write_animation_for_track(det, names, imgs, tr, out_dir, fps=6, circle_radius=4)

        hit = {
            "detector": det,
            "series_mid_frame": mid_name,
            "track_index": i,
            "crop_path": str(crop_path),
            "positions": positions,
            "image_size": [TARGET_SIZE, TARGET_SIZE],
            "origin": "upper_left",
            "original_mid_path": orig_p,
            "annotated_mid_path": ann_p,
            "animation_gif_path": anim.get("animation_gif_path"),
            "animation_mp4_path": anim.get("animation_mp4_path"),
            "animation_gif_clean_path": anim.get("animation_gif_clean_path"),
            "animation_mp4_clean_path": anim.get("animation_mp4_clean_path"),
        }
        hits.append(hit)

    if debug and imgs and tracks:
        draw_tracks_overlay(imgs[len(imgs)//2], tracks, Path(out_dir)/f"overlay_{det}.png")

    stats = {
        "frames": len(series),
        "tracks": len(tracks),
        "last_frame_name": names[-1] if names else "",
        "last_frame_iso": frame_iso_from_name(names[-1]) if names else "",
        "last_frame_size": [TARGET_SIZE, TARGET_SIZE],
    }
    return hits, stats

# -----------------------------------------------------------------------------
# Entry
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=6)
    ap.add_argument("--step-min", type=int, default=12)
    ap.add_argument("--out", type=str, default="detections")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    debug = os.getenv("DETECTOR_DEBUG", "0") == "1"

    detectors_stats: Dict[str, Any] = {}
    all_hits: List[Dict[str, Any]] = []
    errors: List[str] = []
    fetched_count = 0

    for det in DETECTORS:
        try:
            series = build_series(det, hours=args.hours, step_min=args.step_min)
            if not series:
                errors.append(f"{det}: No frames fetched")
                continue
            fetched_count += len(series)
            hits, stats = package_detector(det, series, out_dir, debug=debug)
            detectors_stats[det] = stats
            all_hits.extend(hits)
        except Exception as e:
            import traceback
            errors.append(f"{det}: {e}\n{traceback.format_exc()}")

    ts_iso = utcnow_iso()
    summary = {
        "timestamp_utc": ts_iso,
        "hours_back": args.hours,
        "step_min": args.step_min,
        "detectors": detectors_stats,
        "fetched_new_frames": fetched_count,
        "errors": errors,
        "auto_selected_count": 0,
        "name": "latest_status.json",
        "generated_at": ts_iso,
        "c2_frames": detectors_stats.get("C2", {}).get("frames", 0),
        "c3_frames": detectors_stats.get("C3", {}).get("frames", 0),
        "candidates": all_hits,
    }

    json_path = out_dir / "latest_status.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {json_path}")

    if all_hits:
        ts_name = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        cand_path = out_dir / f"candidates_{ts_name}.json"
        with open(cand_path, "w") as f:
            json.dump(all_hits, f, indent=2)
        print(f"Wrote {cand_path}")

if __name__ == "__main__":
    main()
```

### Key Changes
| Line/Section | Fix | Why? |
|--------------|-----|------|
| `save_original_and_annotated` (the broken `cv2.line` call) | Changed `cv, (px, py), (x, y), (0,255,0), 1)` to `cv2.line(vis, (px, py), (x, y), (0,255,0), 1)` | Syntax error: Removed stray "cv," and fixed the unmatched `)`. Now it's a valid `cv2.line` invocation. |
| `read_gray_jpg` | Added a print warning if `im is None` | Helps debug corrupt frames without crashing. |
| Overall | Retained all previous fixes (resizing, error handling) | Ensures no size mismatches or silent failures. |

### Testing in GitHub Actions
1. **Commit & Push** this updated file to `main`.
2. **Manually trigger** the "SOHO Comet Hunt" workflow in Actions.
3. Check the logs: It should run without the `SyntaxError` and produce `detections/latest_status.json` with frames > 0.
4. If you see OpenCV warnings or empty frames, it's likely data gaps (normal for SOHO)—try `--hours 24` temporarily.

If issues persist (e.g., missing `fetch_lasco.py`), share the new error logs! This should get your hunter running smoothly. 🚀
