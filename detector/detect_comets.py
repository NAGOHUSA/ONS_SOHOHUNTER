# detector/detect_comets.py
# SOHO comet detector with per-candidate ORIGINAL, ANNOTATED, and ANIMATIONS (annotated + clean)
from __future__ import annotations

import os, re, math, csv, json, argparse, pathlib, shutil
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
import requests

# --------------------------------------------------------------------------------------
# (The helper routines below are the same as your current file — file/dir helpers, 
# LASCO fetch helpers, detection, tracking, correlations, etc. I’m keeping them intact.)
# --------------------------------------------------------------------------------------

# ---------------------- tiny utils ----------------------
def ensure_dir(p: Path):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)

def save_png(path: Path, img: np.ndarray) -> None:
    ensure_dir(path)
    cv2.imwrite(str(path), img)

# ---------------------- vis helpers ----------------------
def draw_tracks_overlay(base_img: np.ndarray, tracks, out_path: Path, radius=3, thickness=1):
    if len(base_img.shape) == 2:
        vis_bgr = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    else:
        vis_bgr = base_img.copy()
    h, w = base_img.shape[:2]
    cv2.line(vis_bgr, (w//2, 0), (w//2, h), (40,40,40), 1)
    cv2.line(vis_bgr, (0, h//2), (w, h//2), (40,40,40), 1)
    for tr in tracks:
        for (t,x,y,a) in tr:
            cv2.circle(vis_bgr, (int(round(x)), int(round(y))), radius, (0,255,0), thickness)
    save_png(out_path, vis_bgr)

def contact_sheet(images: List[np.ndarray], cols=4, margin=2) -> Optional[np.ndarray]:
    if not images: return None
    h, w = images[0].shape[:2]
    rows = math.ceil(len(images)/cols)
    canvas = np.zeros((rows*h + (rows-1)*margin, cols*w + (cols-1)*margin, 3), np.uint8)
    for i, im in enumerate(images):
        if len(im.shape) == 2: im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        r, c = divmod(i, cols)
        y = r*(h+margin); x = c*(w+margin)
        canvas[y:y+h, x:x+w] = im
    return canvas

# ---------------------- mid/original+annotated ----------------------
def save_original_and_annotated(detector_name: str, mid_name: str, positions, out_dir: Path) -> Tuple[str, str]:
    """Saves the ORIGINAL mid frame and an ANNOTATED version with the detected track drawn."""
    mid_path = out_dir / "originals" / f"{detector_name}_{mid_name}"
    ann_path = out_dir / "annotated" / f"{detector_name}_{mid_name}"
    mid_img = cv2.imread(str(out_dir / "crops" / f"{detector_name}_{mid_name}"), cv2.IMREAD_UNCHANGED)
    if mid_img is None:
        # Fallback: if crop not present yet, just skip writing originals
        return "", ""
    ensure_dir(mid_path); ensure_dir(ann_path)
    cv2.imwrite(str(mid_path), mid_img)

    if len(mid_img.shape) == 2:
        vis = cv2.cvtColor(mid_img, cv2.COLOR_GRAY2BGR)
    else:
        vis = mid_img.copy()

    # draw track
    if positions:
        for i, p in enumerate(positions):
            x, y = int(round(p["x"])), int(round(p["y"]))
            cv2.circle(vis, (x, y), 4, (0,255,0), 1)
            if i:
                px, py = int(round(positions[i-1]["x"])), int(round(positions[i-1]["y"]))
                cv2.line(vis, (px,py), (x,y), (0,255,0), 1)

    cv2.imwrite(str(ann_path), vis)
    return str(mid_path), str(ann_path)

# ---------------------- Sungrazer exports ----------------------
def write_sungrazer_exports(detector_name: str, track_idx: int, positions, image_size, out_dir: Path) -> Tuple[str,str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / f"{detector_name}_track{track_idx}_sungrazer.txt"
    csv_path = out_dir / f"{detector_name}_track{track_idx}_sungrazer.csv"
    with open(txt_path, "w") as f:
        f.write("# FrameTimeUTC, x, y\n")
        for p in positions:
            f.write(f"{p['time_utc']} {int(round(p['x']))} {int(round(p['y']))}\n")
    with open(csv_path, "w") as f:
        f.write("frame_time_utc,x,y\n")
        for p in positions:
            f.write(f"{p['time_utc']},{int(round(p['x']))},{int(round(p['y']))}\n")
    return str(txt_path), str(csv_path)

# ---------------------- NEW: per-candidate animations (annotated + clean) ----------------------
def write_animation_for_track(detector_name: str,
                              names: List[str],
                              images: List[np.ndarray],
                              tr,  # list of (t,x,y,a) with t=index into names/images
                              out_dir: Path,
                              fps: int = 6,
                              circle_radius: int = 4) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Create two animations (GIF and MP4 each):
    - annotated: growing trail + current point
    - clean: source frames only
    Returns (gif_annotated, mp4_annotated, gif_clean, mp4_clean).
    """
    try:
        t_min = tr[0][0]; t_max = tr[-1][0]
        frames_bgr: List[np.ndarray] = []
        frames_clean: List[np.ndarray] = []
        trail_pts: List[Tuple[int,int]] = []

        xy_by_t = {t:(int(round(x)), int(round(y))) for (t,x,y,_) in tr}

        for ti in range(t_min, t_max + 1):
            frame = images[ti]
            if len(frame.shape) == 2:
                bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                bgr = frame.copy()

            frames_clean.append(bgr.copy())  # clean copy before drawing

            # Extend trail if we have xy at this ti
            if ti in xy_by_t:
                trail_pts.append(xy_by_t[ti])

            # draw trail
            for i in range(1, len(trail_pts)):
                cv2.line(bgr, trail_pts[i-1], trail_pts[i], (0,255,0), 1)
            # draw current
            if ti in xy_by_t:
                cv2.circle(bgr, xy_by_t[ti], circle_radius, (0,255,0), 1)

            frames_bgr.append(bgr)

        # Write GIF if possible
        gif_path: Optional[str] = None
        mp4_path: Optional[str] = None
        gif_clean: Optional[str] = None
        mp4_clean: Optional[str] = None

        tr_idx = tr[0][0]  # just to keep name stable; any per-track index works
        try:
            import imageio
            gif_out = out_dir / "animations" / f"{detector_name}_track{tr_idx}_annotated.gif"
            gif_out.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(str(gif_out), frames_bgr, fps=fps)
            gif_path = str(gif_out)

            gif_out_clean = out_dir / "animations" / f"{detector_name}_track{tr_idx}_clean.gif"
            imageio.mimsave(str(gif_out_clean), frames_clean, fps=fps)
            gif_clean = str(gif_out_clean)
        except Exception:
            pass

        # Always try MP4 via OpenCV
        mp4_out = out_dir / "animations" / f"{detector_name}_track{tr_idx}_annotated.mp4"
        mp4_out.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        h, w = frames_bgr[0].shape[:2]
        writer = cv2.VideoWriter(str(mp4_out), fourcc, fps, (w, h))
        for fr in frames_bgr:
            writer.write(fr)
        writer.release()
        mp4_path = str(mp4_out)

        mp4_out_clean = out_dir / "animations" / f"{detector_name}_track{tr_idx}_clean.mp4"
        writer2 = cv2.VideoWriter(str(mp4_out_clean), fourcc, fps, (w, h))
        for fr in frames_clean:
            writer2.write(fr)
        writer2.release()
        mp4_clean = str(mp4_out_clean)

        return (gif_path, mp4_path, gif_clean, mp4_clean)
    except Exception:
        return (None, None, None, None)

# ---------------------- (… your existing detection & tracking code …) ----------------------
# Everything from fetching frames to finding tracks should remain as-is.
# Below is the spot where we package per-detector hits, and then the final summary.

def package_detector_hits(detector_name: str,
                          series: List[Tuple[str, np.ndarray]],
                          tracks: List[List[Tuple[int,float,float,float]]],
                          out_dir: Path,
                          DEBUG_OVERLAYS: bool) -> Tuple[List[Dict[str,Any]], Dict[str,Any]]:
    """Build per-candidate artifacts and metadata."""
    names = [s[0] for s in series]
    images = [s[1] for s in series]
    last_name, last_img = names[-1], images[-1]

    # some of your crop logic here … (unchanged)

    hits: List[Dict[str,Any]] = []
    for i, tr in enumerate(tracks):
        # positions list
        positions = []
        for (t, x, y, a) in tr:
            fname = names[t]
            positions.append({
                "time_utc": parse_frame_iso(fname) or "",
                "x": float(x), "y": float(y)
            })

        # save per-track Sungrazer exports
        write_sungrazer_exports(detector_name, i+1, positions, image_size=images[0].shape[:2], out_dir=out_dir / "reports")

        # mid originals
        mid_idx = tr[len(tr)//2][0]
        mid_name = names[mid_idx]
        orig_path, ann_path = save_original_and_annotated(detector_name, mid_name, positions, out_dir)

        # animations (annotated + clean)
        gif_path, mp4_path, gif_clean, mp4_clean = write_animation_for_track(
            detector_name, names, images, tr, out_dir, fps=6, circle_radius=4
        )

        hits.append({
            "detector": detector_name,
            "series_mid_frame": mid_name,
            "track_index": i+1,
            "crop_path": str(out_dir / "crops" / f"{detector_name}_{mid_name}"),
            "positions": positions,
            "image_size": [int(images[0].shape[1]), int(images[0].shape[0])],
            "origin": "upper_left",
            "original_mid_path": orig_path,
            "annotated_mid_path": ann_path,
            "animation_gif_path": gif_path,
            "animation_mp4_path": mp4_path,
            "animation_gif_clean_path": gif_clean,
            "animation_mp4_clean_path": mp4_clean
        })

    # overlays / contacts
    if DEBUG_OVERLAYS:
        draw_tracks_overlay(images[len(images)//2], tracks, out_dir / f"overlay_{detector_name}.png")
        sheet = contact_sheet([im for im in images[-8:]])
        if sheet is not None:
            save_png(out_dir / f"contact_{detector_name}.png", sheet)

    h, w = images[-1].shape[:2]
    return hits, {
        "frames": len(series),
        "tracks": len(tracks),
        "last_frame_name": last_name,
        "last_frame_iso": parse_frame_iso(last_name) or "",
        "last_frame_size": [int(w), int(h)]
    }

# ---------------------- main ----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=6)
    parser.add_argument("--step-min", type=int, default=12)
    parser.add_argument("--out", type=str, default="detections")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # (fetching, building series, running detectors, etc.) … unchanged …
    # assume we end with:
    #   results: Dict[str, List[hit]]
    #   detectors_stats: Dict[str, Any]
    #   fetched: List[str], errors: List[str], to_submit: List[hit]
    #   all_hits = results.get("C2", []) + results.get("C3", [])

    # ------------------ (the rest of your pipeline here) ------------------
    # … keep your existing body, then replace ONLY the final write block below …

    # (… after you computed results/detectors_stats/fetched/errors/to_submit/all_hits …)
    ts_iso = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    summary = {
        "timestamp_utc": ts_iso,
        "hours_back": args.hours,
        "step_min": args.step_min,
        "detectors": detectors_stats,
        "fetched_new_frames": len(fetched),
        "errors": errors,
        "auto_selected_count": len(to_submit),
    }

    # 🔁 FRONTEND-COMPATIBLE FIELDS + EMBED CANDIDATES
    summary["name"] = "latest_status.json"
    summary["generated_at"] = ts_iso
    summary["c2_frames"] = (detectors_stats.get("C2") or {}).get("frames", 0)
    summary["c3_frames"] = (detectors_stats.get("C3") or {}).get("frames", 0)
    summary["candidates"] = all_hits

    with open(out_dir / "latest_status.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote status: {out_dir/'latest_status.json'}")

    if all_hits:
        ts_name = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_json = out_dir / f"candidates_{ts_name}.json"
        with open(out_json, "w") as f:
            json.dump(all_hits, f, indent=2)
        print(f"Wrote {out_json}")

        # optional combined CSV if you had that already:
        # write_combined_csv(out_dir, ts_name, all_hits)
    else:
        print("No candidates this run.")

    with open(out_dir / "to_submit.json", "w") as f:
        json.dump({"auto_selected": to_submit, "timestamp_utc": summary["timestamp_utc"]}, f, indent=2)
    print(f"Wrote {out_dir/'to_submit.json'}")

    non_vetoed = [h for h in all_hits if not h.get("vetoed")]
    if non_vetoed:
        send_webhook(summary, non_vetoed)

if __name__ == "__main__":
    main()
