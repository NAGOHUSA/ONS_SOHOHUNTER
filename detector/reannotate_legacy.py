# detector/reannotate_legacy.py
"""
Rebuilds crops, annotated crops, GIF/MP4 timelapses, and originals for historical
candidates JSON files in detections/.

- Computes a bbox from candidate["positions"] (tight with padding)
- Uses frames/<C2|C3>/<frame_name>.jpg if present; otherwise calls fetch_lasco.fetch_window
  near the candidate time to repopulate frames, then proceeds.
- Writes new assets into:
    detections/crops/, detections/annotated/, detections/gifs/, detections/mp4/, detections/originals/
- Updates fields on each candidate:
    crop_path, annotated_path, original_mid_path, crop_gif_path, crop_mp4_path, bbox
- Skips candidates with missing/invalid positions
"""

from __future__ import annotations
import os, re, json, math, pathlib, shutil, argparse
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import cv2

try:
    import imageio
except Exception:
    imageio = None

# Local fetcher (already in your repo)
from fetch_lasco import fetch_window

CROP_PAD = int(os.getenv("CROP_PAD", "10"))
CROP_SIZE_C2 = int(os.getenv("CROP_SIZE_C2", "96"))
CROP_SIZE_C3 = int(os.getenv("CROP_SIZE_C3", "128"))
GIF_FPS = int(os.getenv("CROP_GIF_FPS", "5"))
MP4_FPS = int(os.getenv("CROP_MP4_FPS", "8"))

DETECTIONS_DIR = pathlib.Path("detections")
FRAMES_DIR = pathlib.Path("frames")

def ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True)

def parse_frame_iso(name: str) -> Optional[datetime]:
    # Accepts e.g. C3_20251106_0930_c3_1024.jpg -> 2025-11-06T09:30:00Z
    m = re.search(r'(\d{8})_(\d{4,6})', name or "")
    if not m: 
        return None
    d, t = m.groups()
    if len(t) == 4: t = t + "00"
    try:
        return datetime(int(d[0:4]), int(d[4:6]), int(d[6:8]),
                        int(t[0:2]), int(t[2:4]), int(t[4:6]))
    except Exception:
        return None

def load_gray(path: pathlib.Path) -> Optional[np.ndarray]:
    if not path.exists(): return None
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return img

def write_gif(seq: List[np.ndarray], out_path: pathlib.Path, fps: int) -> bool:
    if imageio is None or len(seq) < 2:
        return False
    try:
        imgs = [cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) for f in seq]
        imageio.mimsave(str(out_path), imgs, fps=fps, loop=0)
        return True
    except Exception as e:
        print(f"[warn] GIF save failed: {e}")
        return False

def write_mp4(seq: List[np.ndarray], out_path: pathlib.Path, fps: int) -> bool:
    if len(seq) < 2: return False
    try:
        h, w = seq[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h), isColor=False)
        for f in seq:
            vw.write(cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX))
        vw.release()
        return True
    except Exception as e:
        print(f"[warn] MP4 save failed: {e}")
        return False

def detector_from_candidate(c: Dict[str, Any]) -> str:
    d = (c.get("detector") or c.get("instrument") or c.get("camera") or "").upper()
    if "C2" in d: return "C2"
    if "C3" in d: return "C3"
    # last resort: guess from frame name
    mid = c.get("series_mid_frame") or ""
    return "C2" if "_c2_" in mid.lower() else "C3"

def bbox_from_positions(positions: List[Dict[str, Any]], img_shape: Tuple[int,int], pad: int) -> Optional[Tuple[int,int,int,int]]:
    if not positions: return None
    xs = [float(p.get("x", 0)) for p in positions]
    ys = [float(p.get("y", 0)) for p in positions]
    if not xs or not ys: return None
    h, w = img_shape[:2]
    x0 = max(0, int(min(xs) - pad))
    y0 = max(0, int(min(ys) - pad))
    x1 = min(w-1, int(max(xs) + pad))
    y1 = min(h-1, int(max(ys) + pad))
    return (x0, y0, x1, y1)

def pick_mid_index(positions: List[Dict[str, Any]]) -> int:
    return len(positions) // 2

def ensure_original(det: str, mid_name: str, mid_img: np.ndarray):
    orig_dir = DETECTIONS_DIR / "originals"; ensure_dir(orig_dir)
    src = FRAMES_DIR / det / mid_name
    dst = orig_dir / mid_name
    try:
        if src.exists():
            if not dst.exists():
                shutil.copyfile(src, dst)
        else:
            # fallback: write aligned/loaded gray
            cv2.imwrite(str(dst), mid_img)
    except Exception as e:
        print(f"[warn] could not ensure original for {mid_name}: {e}")
    return f"originals/{mid_name}"

def rebuild_assets_for_candidate(c: Dict[str, Any]) -> Dict[str, Any]:
    det = detector_from_candidate(c)
    # Build list of frame names from positions.frame if present; else try series_mid_frame only
    frames = [p.get("frame") for p in (c.get("positions") or []) if p.get("frame")]
    if not frames:
        mid_name = c.get("series_mid_frame") or ""
        if not mid_name:
            return c
        frames = [mid_name]

    # Make sure required frames exist locally. If not, fetch around the mid time.
    mid_idx = pick_mid_index(c.get("positions") or [{}])
    mid_name = frames[mid_idx] if mid_idx < len(frames) else frames[len(frames)//2]
    mid_dt = parse_frame_iso(mid_name)
    ensure_dir(FRAMES_DIR / det)
    missing = [f for f in frames if not (FRAMES_DIR / det / f).exists()]
    if missing and mid_dt:
        try:
            # Fetch 2 hours around mid time to maximize hit rate
            fetch_window(hours_back=2, step_min=12, root=str(FRAMES_DIR), around=mid_dt, detector=det)
        except TypeError:
            # Older fetch_window signature: fallback to default (will pull recent)
            fetch_window(hours_back=6, step_min=12, root=str(FRAMES_DIR))

    # Load sequence (only those we have)
    seq: List[np.ndarray] = []
    present_names: List[str] = []
    for fn in frames:
        im = load_gray(FRAMES_DIR / det / fn)
        if im is not None:
            seq.append(im); present_names.append(fn)

    if not seq:
        # Nothing to do for this candidate
        return c

    # Use mid actually present
    if mid_name not in present_names:
        mid_name = present_names[len(present_names)//2]
    mid_idx_present = present_names.index(mid_name)
    mid_img = seq[mid_idx_present]

    # Compute bbox from positions on the same coordinate system as frames
    bbox = bbox_from_positions(c.get("positions") or [], mid_img.shape, CROP_PAD)
    if bbox is None:
        # fallback: small centered crop to avoid crash
        h, w = mid_img.shape[:2]
        cx, cy = w//2, h//2
        bbox = (max(0, cx-24), max(0, cy-24), min(w-1, cx+24), min(h-1, cy+24))

    x0,y0,x1,y1 = bbox
    # Build cropped sequence with SAME bbox across all available frames
    cropped_seq = [im[y0:y1+1, x0:x1+1].copy() for im in seq if im is not None and im.shape[0]>y1 and im.shape[1]>x1]

    # Representative crop: resize by detector's target longest side
    rep = cropped_seq[mid_idx_present] if mid_idx_present < len(cropped_seq) else cropped_seq[len(cropped_seq)//2]
    target = CROP_SIZE_C2 if det=="C2" else CROP_SIZE_C3
    if max(rep.shape[:2])>0 and max(rep.shape[:2])!=target:
        s = float(target)/float(max(rep.shape[:2]))
        rep_resized = cv2.resize(rep, (int(rep.shape[1]*s), int(rep.shape[0]*s)), interpolation=cv2.INTER_AREA)
        scale = s
    else:
        rep_resized = rep
        scale = 1.0

    # File names + dirs
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = f"{det}_{ts}_trk"
    crops_dir = DETECTIONS_DIR/"crops"; annotated_dir = DETECTIONS_DIR/"annotated"
    gifs_dir = DETECTIONS_DIR/"gifs"; mp4_dir = DETECTIONS_DIR/"mp4"
    for d in (crops_dir, annotated_dir, gifs_dir, mp4_dir): ensure_dir(d)

    crop_path = crops_dir/f"{base}.png"
    ann_path  = annotated_dir/f"{base}_annotated.png"
    gif_path  = gifs_dir/f"{base}.gif"
    mp4_path  = mp4_dir/f"{base}.mp4"

    # Save crop
    cv2.imwrite(str(crop_path), rep_resized)

    # Annotated CROP (draw positions mapped to crop coords with scale)
    ann = cv2.cvtColor(rep_resized, cv2.COLOR_GRAY2BGR)
    for p in (c.get("positions") or []):
        x, y = float(p.get("x", 0)), float(p.get("y", 0))
        cx_local = int(round((x - x0) * scale))
        cy_local = int(round((y - y0) * scale))
        cv2.circle(ann, (cx_local, cy_local), 2, (0,255,255), -1)
    cv2.rectangle(ann, (1,1), (ann.shape[1]-2, ann.shape[0]-2), (0,200,0), 1)
    cv2.imwrite(str(ann_path), ann)

    # Animations (use unscaled sequence to keep geometry consistent)
    gif_ok = write_gif(cropped_seq, gif_path, fps=GIF_FPS)
    mp4_ok = write_mp4(cropped_seq, mp4_path, fps=MP4_FPS)

    # Ensure original mid in detections/originals
    original_rel = ensure_original(det, mid_name, mid_img)

    # Update candidate fields
    c["crop_path"] = f"crops/{crop_path.name}"
    c["annotated_path"] = f"annotated/{ann_path.name}"
    c["original_mid_path"] = original_rel
    c["bbox"] = [int(x0), int(y0), int(x1), int(y1)]
    if gif_ok: c["crop_gif_path"] = f"gifs/{gif_path.name}"
    else: c.pop("crop_gif_path", None)
    if mp4_ok: c["crop_mp4_path"] = f"mp4/{mp4_path.name}"
    else: c.pop("crop_mp4_path", None)

    return c

def process_file(path: pathlib.Path, limit: Optional[int]=None) -> bool:
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        print(f"[skip] {path.name}: JSON parse error: {e}")
        return False

    arr = data if isinstance(data, list) else data.get("candidates") or []
    if not isinstance(arr, list) or not arr:
        print(f"[info] {path.name}: no candidates")
        return False

    changed = False
    count = 0
    for i, cand in enumerate(arr):
        if limit is not None and count >= limit:
            break
        if not cand.get("positions"):
            continue
        before = json.dumps({k: cand.get(k) for k in ("crop_path","annotated_path","original_mid_path","crop_gif_path","crop_mp4_path","bbox")}, sort_keys=True)
        cand = rebuild_assets_for_candidate(cand)
        after  = json.dumps({k: cand.get(k) for k in ("crop_path","annotated_path","original_mid_path","crop_gif_path","crop_mp4_path","bbox")}, sort_keys=True)
        if before != after:
            changed = True
        arr[i] = cand
        count += 1

    # Write back
    if isinstance(data, list):
        new_json = arr
    else:
        data["candidates"] = arr
        new_json = data

    if changed:
        path.write_text(json.dumps(new_json, indent=2), encoding="utf-8")
        print(f"[write] {path.name} (updated {count} candidates)")
    else:
        print(f"[ok] {path.name}: no changes needed ({count} candidates checked)")
    return changed

def main():
    ap = argparse.ArgumentParser(description="Rebuild crops/annotations/GIF/MP4 for historical reports.")
    ap.add_argument("--limit-per-file", type=int, default=None, help="Max candidates per JSON to rebuild (debug).")
    ap.add_argument("--max-files", type=int, default=0, help="If >0, process only the newest N report files.")
    ap.add_argument("--dry-run", action="store_true", help="Scan & fetch frames but do not write JSON or assets.")
    args = ap.parse_args()

    # Prepare dirs
    for sub in ("crops","annotated","gifs","mp4","originals"):
        ensure_dir(DETECTIONS_DIR / sub)

    # Discover report files
    files = sorted(DETECTIONS_DIR.glob("candidates_*.json"), key=lambda p: p.name, reverse=True)
    if args.max_files and args.max_files > 0:
        files = files[:args.max_files]

    any_changed = False
    for f in files:
        if args.dry_run:
            # do the work but don’t write JSON (we still write assets to prove paths)
            changed_here = process_file(f, limit=args.limit_per_file)
            # still counts as changed if assets created AND fields differ
            any_changed = any_changed or changed_here
        else:
            changed_here = process_file(f, limit=args.limit_per_file)
            any_changed = any_changed or changed_here

    print(f"Done. Changed: {any_changed}")

if __name__ == "__main__":
    main()
