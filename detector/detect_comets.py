#!/usr/bin/env python3
"""
SOHO Comet Detector
Processes LASCO C2 and C3 images to find moving objects
"""

from __future__ import annotations
import os
import re
import math
import json
import argparse
import pathlib
import shutil
import sys
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np

# Optional GIF writer
try:
    import imageio
    import imageio_ffmpeg
    GIF_AVAILABLE = True
except Exception:
    GIF_AVAILABLE = False

# Local modules - UPDATED IMPORT
try:
    from fetch_lasco import SohoImageFetcher
    FETCH_AVAILABLE = True
except ImportError:
    FETCH_AVAILABLE = False
    print("WARNING: fetch_lasco module not available")

try:
    from ai_classifier import classify_crop_batch
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("WARNING: ai_classifier module not available")

# ----------------------------- Tunables -----------------------------
OCCULTER_RADIUS_FRACTION = float(os.getenv("OCCULTER_RADIUS_FRACTION", "0.20"))
MAX_EDGE_RADIUS_FRACTION = float(os.getenv("MAX_EDGE_RADIUS_FRACTION", "0.95"))
CROP_SIZE_C2 = int(os.getenv("CROP_SIZE_C2", "96"))
CROP_SIZE_C3 = int(os.getenv("CROP_SIZE_C3", "128"))
CROP_PAD = int(os.getenv("CROP_PAD", "15"))
AI_MIN_SCORE = float(os.getenv("AI_MIN_SCORE", "0.40"))
DEBUG_OVERLAYS = os.getenv("DETECTOR_DEBUG", "0") == "1"
GIF_FPS = 3
MP4_FPS = 4

# ----------------------------- Helper Functions -----------------------------
def ensure_dir(p: pathlib.Path):
    """Ensure directory exists"""
    p.mkdir(parents=True, exist_ok=True)

def load_series(folder: pathlib.Path) -> List[Tuple[str, np.ndarray]]:
    """Load image series from folder"""
    pairs = []
    if not folder.exists():
        return pairs
    
    # Get all image files
    extensions = (".png", ".jpg", ".jpeg")
    files = sorted([f for f in folder.iterdir() if f.suffix.lower() in extensions])
    
    # Load images (limit to 50 to avoid memory issues)
    for p in files[:50]:
        try:
            im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if im is not None:
                pairs.append((p.name, im))
        except Exception as e:
            print(f"Warning: Could not load {p}: {e}")
    
    return pairs

def parse_frame_iso(name: str) -> Optional[str]:
    """Parse ISO timestamp from filename"""
    m = re.search(r'(\d{8})_(\d{4,6})', name)
    if not m:
        return None
    d, t = m.groups()
    t = t + "00" if len(t) == 4 else t  # Pad to 6 digits if needed
    return f"{d[0:4]}-{d[4:6]}-{d[6:8]}T{t[0:2]}:{t[2:4]}:{t[4:6]}Z"

def is_within_last_hours(filename: str, hours: int = 12) -> bool:
    """Check if frame is within last N hours"""
    iso = parse_frame_iso(filename)
    if not iso:
        return False
    try:
        frame_time = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return frame_time >= cutoff
    except Exception:
        return False

def stabilize_fast(base: np.ndarray, img: np.ndarray, max_iterations: int = 20) -> np.ndarray:
    """Fast stabilization with iteration limit"""
    try:
        # Convert to 8-bit
        base_8bit = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img_8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Find transformation
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, 1e-4)
        
        cc, warp_matrix = cv2.findTransformECC(
            base_8bit, img_8bit, warp_matrix, 
            cv2.MOTION_EUCLIDEAN, 
            criteria
        )
        
        # Apply transformation
        aligned = cv2.warpAffine(
            img, warp_matrix, (img.shape[1], img.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )
        return aligned
    except Exception as e:
        print(f"Stabilization failed: {e}")
        return img

def build_static_mask_fast(images: List[np.ndarray]) -> np.ndarray:
    """Fast static mask generation"""
    if len(images) < 3:
        return np.zeros_like(images[0], dtype=np.uint8)
    
    # Use only last 5 images for speed
    sample = images[-min(5, len(images)):]
    median = np.median(sample, axis=0).astype(np.uint8)
    
    # Simple threshold
    threshold = np.median(median) + 10
    _, mask = cv2.threshold(median, threshold, 255, cv2.THRESH_BINARY)
    
    # Clean up
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return mask

def find_moving_points_fast(series: List[Tuple[str, np.ndarray]], 
                           static_mask: Optional[np.ndarray] = None,
                           frame_skip: int = 1) -> List[Tuple[int, float, float, float]]:
    """Fast moving point detection"""
    pts = []
    
    for i in range(frame_skip, len(series), frame_skip):
        _, prev = series[i - frame_skip]
        _, curr = series[i]
        
        # Fast diff
        diff = cv2.absdiff(curr, prev)
        
        if static_mask is not None:
            diff = cv2.bitwise_and(diff, cv2.bitwise_not(static_mask))
        
        # Simple threshold
        _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 3 <= area <= 200:  # Reasonable area range for comets
                moments = cv2.moments(contour)
                if moments['m00'] != 0:
                    cx = moments['m10'] / moments['m00']
                    cy = moments['m01'] / moments['m00']
                    pts.append((i, float(cx), float(cy), float(area)))
    
    return pts

def link_tracks_fast(points: List[Tuple[int, float, float, float]], 
                    min_len: int = 3, 
                    max_jump: float = 25.0) -> List[List[Tuple[int, float, float, float]]]:
    """Fast track linking"""
    if not points:
        return []
    
    tracks = []
    by_frame = {}
    
    for t, x, y, a in points:
        by_frame.setdefault(t, []).append((x, y, a))
    
    # Process frames in order
    sorted_frames = sorted(by_frame.keys())
    
    for frame in sorted_frames:
        for x, y, a in by_frame[frame]:
            # Try to extend existing tracks
            best_track = None
            best_dist = max_jump * 2
            
            for track in tracks:
                last_frame, last_x, last_y, _ = track[-1]
                if last_frame == frame - 1:
                    dist = math.hypot(x - last_x, y - last_y)
                    if dist < best_dist and dist <= max_jump:
                        best_dist = dist
                        best_track = track
            
            if best_track is not None:
                best_track.append((frame, x, y, a))
            else:
                tracks.append([(frame, x, y, a)])
    
    # Filter by minimum length
    return [tr for tr in tracks if len(tr) >= min_len]

def radial_guard_ok(x: float, y: float, w: int, h: int, 
                   rmin_frac: float, rmax_frac: float) -> bool:
    """Check if point is within valid radial range"""
    cx, cy = w / 2.0, h / 2.0
    r = math.hypot(x - cx, y - cy)
    rmax = min(cx, cy)
    if rmax == 0:
        return False
    frac = r / rmax
    return rmin_frac <= frac <= rmax_frac

def to_bgr(img_gray: np.ndarray) -> np.ndarray:
    """Convert grayscale to BGR for annotations"""
    return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

def write_gif(frames_gray: List[np.ndarray], out_path: pathlib.Path, fps: int = 5) -> bool:
    """Write GIF animation"""
    if not GIF_AVAILABLE:
        return False
    try:
        imgs = []
        for f in frames_gray:
            # Normalize to 8-bit
            f_norm = cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            imgs.append(f_norm)
        imageio.mimsave(str(out_path), imgs, fps=fps, loop=0)
        return True
    except Exception as e:
        print(f"GIF save failed: {e}")
        return False

def write_mp4(frames_gray: List[np.ndarray], out_path: pathlib.Path, fps: int = 8) -> bool:
    """Write MP4 animation"""
    try:
        h, w = frames_gray[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h), isColor=False)
        
        for f in frames_gray:
            # Normalize to 8-bit
            f_norm = cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            vw.write(f_norm)
        
        vw.release()
        return True
    except Exception as e:
        print(f"MP4 save failed: {e}")
        return False

# ----------------------------- Main Processing -----------------------------
def process_detector(det: str, 
                     series: List[Tuple[str, np.ndarray]], 
                     max_frames: int = 12) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Process a single detector"""
    if not series:
        return [], {"frames": 0, "tracks": 0}
    
    # Limit number of frames
    if len(series) > max_frames:
        series = series[-max_frames:]
    
    names = [n for n, _ in series]
    images = [im for _, im in series]
    
    print(f"  Aligning {len(images)} frames...")
    aligned_names = [names[0]]
    aligned_images = [images[0]]
    
    for i in range(1, len(images)):
        aligned = stabilize_fast(images[0], images[i])
        aligned_names.append(names[i])
        aligned_images.append(aligned)
    
    # Create static mask
    print(f"  Creating static mask...")
    static_mask = build_static_mask_fast(aligned_images)
    
    # Find moving points
    print(f"  Finding moving points...")
    points = find_moving_points_fast(list(zip(aligned_names, aligned_images)), static_mask)
    
    # Apply radial guard
    h, w = images[0].shape
    guarded_points = [
        (t, x, y, a) for (t, x, y, a) in points
        if radial_guard_ok(x, y, w, h, OCCULTER_RADIUS_FRACTION, MAX_EDGE_RADIUS_FRACTION)
    ]
    
    # Link tracks
    print(f"  Linking tracks...")
    tracks = link_tracks_fast(guarded_points, min_len=2, max_jump=25.0)
    
    # Statistics
    stats = {
        "frames": len(series),
        "tracks": len(tracks),
        "last_frame_name": names[-1] if names else "",
        "last_frame_iso": parse_frame_iso(names[-1]) if names else "",
        "last_frame_size": [int(w), int(h)]
    }
    
    # Process tracks into candidates
    candidates = process_tracks(det, aligned_names, aligned_images, tracks)
    
    return candidates, stats

def process_tracks(det: str, 
                   names: List[str], 
                   images: List[np.ndarray], 
                   tracks: List[List[Tuple[int, float, float, float]]]) -> List[Dict[str, Any]]:
    """Process found tracks into candidates"""
    candidates = []
    
    for i, track in enumerate(tracks, 1):
        try:
            # Extract positions
            positions = []
            for t, x, y, a in track:
                iso = parse_frame_iso(names[t]) or ""
                positions.append({
                    "frame": names[t],
                    "time_utc": iso,
                    "x": float(x),
                    "y": float(y)
                })
            
            # Find bounding box for track
            xs = [x for _, x, _, _ in track]
            ys = [y for _, _, y, _ in track]
            
            # Get mid frame
            mid_idx = len(images) // 2
            mid_name = names[mid_idx]
            mid_img = images[mid_idx]
            
            # Calculate crop box
            x0 = int(max(0, min(xs) - CROP_PAD))
            y0 = int(max(0, min(ys) - CROP_PAD))
            x1 = int(min(mid_img.shape[1] - 1, max(xs) + CROP_PAD))
            y1 = int(min(mid_img.shape[0] - 1, max(ys) + CROP_PAD))
            
            # Ensure directory structure
            crops_dir = pathlib.Path("detections") / "crops"
            anno_dir = pathlib.Path("detections") / "annotated"
            orig_dir = pathlib.Path("detections") / "originals"
            gifs_dir = pathlib.Path("detections") / "gifs"
            mp4_dir = pathlib.Path("detections") / "mp4"
            
            for d in (crops_dir, anno_dir, orig_dir, gifs_dir, mp4_dir):
                ensure_dir(d)
            
            # Create crop filename
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            base = f"{det}_{ts}_trk{i}"
            crop_name = f"{base}.png"
            anno_name = f"{base}_annotated.png"
            gif_name = f"{base}.gif"
            mp4_name = f"{base}.mp4"
            
            # Create crop from mid frame
            crop = mid_img[y0:y1+1, x0:x1+1].copy()
            
            # Resize to standard size
            target_size = CROP_SIZE_C2 if det == "C2" else CROP_SIZE_C3
            if crop.size > 0:
                scale = target_size / max(crop.shape)
                new_w = int(crop.shape[1] * scale)
                new_h = int(crop.shape[0] * scale)
                crop_resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(crops_dir / crop_name), crop_resized)
            
            # Create annotated crop
            ann_crop = to_bgr(crop_resized) if crop.size > 0 else np.zeros((100, 100, 3), dtype=np.uint8)
            if crop.size > 0:
                # Draw track points
                for t, x, y, a in track:
                    # Map to crop coordinates
                    x_local = (x - x0) * scale
                    y_local = (y - y0) * scale
                    cv2.circle(ann_crop, (int(x_local), int(y_local)), 2, (0, 255, 255), -1)
                cv2.rectangle(ann_crop, (1, 1), (ann_crop.shape[1]-2, ann_crop.shape[0]-2), (0, 200, 0), 1)
                cv2.imwrite(str(anno_dir / anno_name), ann_crop)
            
            # Create animation sequences
            gif_ok = False
            mp4_ok = False
            
            if len(images) >= 3:
                # Create cropped sequence
                cropped_seq = []
                for img in images:
                    crop_frame = img[y0:y1+1, x0:x1+1].copy()
                    if crop_frame.size > 0:
                        cropped_seq.append(crop_frame)
                
                if cropped_seq:
                    # Write GIF
                    gif_path = gifs_dir / gif_name
                    gif_ok = write_gif(cropped_seq, gif_path, GIF_FPS)
                    
                    # Write MP4
                    mp4_path = mp4_dir / mp4_name
                    mp4_ok = write_mp4(cropped_seq, mp4_path, MP4_FPS)
            
            # Copy original mid frame
            src = pathlib.Path("frames") / det / mid_name
            dst = orig_dir / mid_name
            try:
                if src.exists():
                    shutil.copy2(src, dst)
                else:
                    # Save the aligned image if original not found
                    cv2.imwrite(str(dst), mid_img)
            except Exception as e:
                print(f"Warning: Could not save original: {e}")
            
            # Build candidate object
            candidate = {
                "detector": det,
                "track_index": i,
                "positions": positions,
                "series_mid_frame": mid_name,
                "image_size": [images[0].shape[1], images[0].shape[0]],
                "origin": "upper_left",
                "crop_path": f"crops/{crop_name}",
                "annotated_path": f"annotated/{anno_name}",
                "original_mid_path": f"originals/{mid_name}",
                "ai_label": "unknown",
                "ai_score": 0.0
            }
            
            if gif_ok:
                candidate["crop_gif_path"] = f"gifs/{gif_name}"
            if mp4_ok:
                candidate["crop_mp4_path"] = f"mp4/{mp4_name}"
            
            candidates.append(candidate)
            
        except Exception as e:
            print(f"Error processing track {i}: {e}")
            import traceback
            traceback.print_exc()
    
    return candidates

# ----------------------------- Main Function -----------------------------
def main():
    parser = argparse.ArgumentParser(description="SOHO Comet Detector")
    parser.add_argument("--hours", type=int, default=12, help="Hours to look back")
    parser.add_argument("--step-min", type=int, default=12, help="Minutes between frames")
    parser.add_argument("--max-images", type=int, default=24, help="Max images per run")
    parser.add_argument("--timeout", type=int, default=2400, help="Timeout in seconds")
    parser.add_argument("--out", type=str, default="detections", help="Output directory")
    args = parser.parse_args()
    
    start_time = datetime.utcnow()
    print(f"=== SOHO Comet Detection ===")
    print(f"Started at: {start_time.isoformat()}Z")
    print(f"Hours: {args.hours}, Step: {args.step_min}min, Max Images: {args.max_images}")
    
    # Create output directory
    out_dir = pathlib.Path(args.out)
    ensure_dir(out_dir)
    
    # Fetch images - UPDATED FETCH SECTION
    print("\n[1/4] Fetching images...")
    fetched = []
    if FETCH_AVAILABLE:
        try:
            fetcher = SohoImageFetcher(root_dir="frames")
            fetched = fetcher.fetch_window(
                hours_back=args.hours,
                step_min=args.step_min,
                root="frames",
                max_frames=args.max_images // 2
            )
            print(f"  Downloaded {len(fetched)} new files")
        except Exception as e:
            print(f"  WARNING: Fetch failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  WARNING: fetch_lasco module not available, using existing frames only")
    
    # Load series
    print("\n[2/4] Loading series...")
    all_c2 = load_series(pathlib.Path("frames") / "C2")
    all_c3 = load_series(pathlib.Path("frames") / "C3")
    
    # Filter by time
    series_c2 = [item for item in all_c2 if is_within_last_hours(item[0], hours=args.hours)]
    series_c3 = [item for item in all_c3 if is_within_last_hours(item[0], hours=args.hours)]
    
    print(f"  C2: {len(all_c2)} total → {len(series_c2)} in last {args.hours}h")
    print(f"  C3: {len(all_c3)} total → {len(series_c3)} in last {args.hours}h")
    
    # Process detectors
    print("\n[3/4] Processing detectors...")
    max_frames_per_det = max(1, args.max_images // 2)
    
    c2_candidates, c2_stats = process_detector("C2", series_c2, max_frames_per_det)
    c3_candidates, c3_stats = process_detector("C3", series_c3, max_frames_per_det)
    
    all_candidates = c2_candidates + c3_candidates
    
    # AI Classification
    print("\n[4/4] AI Classification...")
    if all_candidates and AI_AVAILABLE:
        try:
            crop_paths = [c["crop_path"] for c in all_candidates if "crop_path" in c]
            if crop_paths:
                ai_results = classify_crop_batch(crop_paths)
                for cand, ai in zip(all_candidates, ai_results):
                    cand["ai_label"] = ai.get("label", "unknown")
                    cand["ai_score"] = float(ai.get("score", 0.0))
                print(f"  AI classified {len(crop_paths)} crops")
            else:
                print("  No crop paths for AI classification")
        except Exception as e:
            print(f"  AI classification failed: {e}")
    else:
        print("  Skipping AI classification")
    
    # Create output summary
    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    runtime = (datetime.utcnow() - start_time).total_seconds()
    
    summary = {
        "timestamp_utc": timestamp,
        "hours_back": args.hours,
        "step_min": args.step_min,
        "runtime_seconds": runtime,
        "detectors": {
            "C2": c2_stats,
            "C3": c3_stats
        },
        "fetched_new_frames": len(fetched),
        "candidates_count": len(all_candidates),
        "comet_candidates": sum(1 for c in all_candidates if c.get("ai_score", 0) >= AI_MIN_SCORE),
        "generated_at": timestamp
    }
    
    # Write output files
    ensure_dir(out_dir)
    
    # Summary file
    with open(out_dir / "latest_status.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Candidates file
    if all_candidates:
        ts_name = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        with open(out_dir / f"candidates_{ts_name}.json", "w") as f:
            json.dump(all_candidates, f, indent=2)
        with open(out_dir / "candidates_latest.json", "w") as f:
            json.dump(all_candidates, f, indent=2)
    else:
        with open(out_dir / "candidates_latest.json", "w") as f:
            json.dump([], f)
    
    print(f"\n=== COMPLETE ===")
    print(f"Runtime: {runtime:.1f}s")
    print(f"Candidates found: {len(all_candidates)}")
    print(f"Comet candidates (AI score ≥ {AI_MIN_SCORE}): {summary['comet_candidates']}")
    
    # List files created
    print(f"\nFiles created in {args.out}/:")
    for f in out_dir.glob("*"):
        if f.is_file():
            print(f"  {f.name}")

if __name__ == "__main__":
    main()
