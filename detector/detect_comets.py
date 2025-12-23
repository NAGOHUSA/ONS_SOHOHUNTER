# detector/detect_comets.py
from __future__ import annotations
import os, re, math, json, argparse, pathlib, shutil, sys, time, signal
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any, Optional
import cv2, numpy as np

# Optional GIF writer
try:
    import imageio
    import imageio_ffmpeg
except Exception:
    imageio = None

# Local modules
from fetch_lasco import fetch_window
from ai_classifier import classify_crop_batch

# ----------------------------- Tunables / env -----------------------------
OCCULTER_RADIUS_FRACTION = float(os.getenv("OCCULTER_RADIUS_FRACTION", "0.20"))
MAX_EDGE_RADIUS_FRACTION = float(os.getenv("MAX_EDGE_RADIUS_FRACTION", "0.95"))
CROP_SIZE_C2 = int(os.getenv("CROP_SIZE_C2", "96"))
CROP_SIZE_C3 = int(os.getenv("CROP_SIZE_C3", "128"))
CROP_PAD = int(os.getenv("CROP_PAD", "15"))
AI_MIN_SCORE = float(os.getenv("AI_MIN_SCORE", "0.40"))
DEBUG_OVERLAYS = os.getenv("DETECTOR_DEBUG", "0") == "1"
GIF_FPS = 3
MP4_FPS = 4

# ----------------------------- Timeout Handler -----------------------------
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Processing timeout")

# ----------------------------- Optimized Helpers -----------------------------
def ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True)

def load_series(folder: pathlib.Path) -> List[Tuple[str, np.ndarray]]:
    """Load image series with timeout protection"""
    pairs = []
    if not folder.exists():
        return pairs
    
    files = sorted(folder.glob("*.*"))
    for p in files[:50]:  # Limit to 50 files
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        try:
            im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if im is not None:
                pairs.append((p.name, im))
        except Exception as e:
            print(f"Warning: Could not load {p}: {e}")
    return pairs

def parse_frame_iso(name: str) -> Optional[str]:
    m = re.search(r'(\d{8})_(\d{4,6})', name)
    if not m:
        return None
    d, t = m.groups()
    t = t + "00" if len(t) == 4 else t
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

def stabilize_fast(base, img, max_iterations=20):
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

def find_moving_points_fast(series, static_mask=None, frame_skip=1):
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
        _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 5 <= area <= 150:  # Adjusted area range
                moments = cv2.moments(contour)
                if moments['m00'] != 0:
                    cx = moments['m10'] / moments['m00']
                    cy = moments['m01'] / moments['m00']
                    pts.append((i, float(cx), float(cy), float(area)))
    
    return pts

def link_tracks_fast(points, min_len=3, max_jump=30):
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

def radial_guard_ok(x, y, w, h, rmin_frac, rmax_frac):
    """Check if point is within valid radial range"""
    cx, cy = w / 2.0, h / 2.0
    r = math.hypot(x - cx, y - cy)
    rmax = min(cx, cy)
    if rmax == 0:
        return False
    frac = r / rmax
    return rmin_frac <= frac <= rmax_frac

# ----------------------------- Main Processing -----------------------------
def process_detector(det: str, series: List[Tuple[str, np.ndarray]], max_frames: int = 12):
    """Process a single detector"""
    if not series:
        return [], {"frames": 0, "tracks": 0}
    
    # Limit number of frames for processing
    if len(series) > max_frames:
        series = series[-max_frames:]
    
    names = [n for n, _ in series]
    images = [im for _, im in series]
    
    # Align images
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
    tracks = link_tracks_fast(guarded_points, min_len=2, max_jump=25)  # Reduced min_len to 2
    
    stats = {
        "frames": len(series),
        "tracks": len(tracks),
        "last_frame_name": names[-1] if names else "",
        "last_frame_iso": parse_frame_iso(names[-1]) if names else "",
        "last_frame_size": [int(w), int(h)]
    }
    
    return process_tracks(det, aligned_names, aligned_images, tracks), stats

def process_tracks(det: str, names: List[str], images: List[np.ndarray], tracks: List):
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
            
            # Create simple crop path
            mid_idx = len(images) // 2
            mid_name = names[mid_idx]
            
            # Save basic info
            candidate = {
                "detector": det,
                "track_index": i,
                "positions": positions,
                "series_mid_frame": mid_name,
                "image_size": [images[0].shape[1], images[0].shape[0]],
                "origin": "upper_left",
                "crop_path": f"crops/{det}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_trk{i}.png",
                "original_mid_path": f"originals/{mid_name}"
            }
            
            candidates.append(candidate)
            
        except Exception as e:
            print(f"Error processing track {i}: {e}")
    
    return candidates

# ----------------------------- Main Function -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=12, help="Hours to look back")
    parser.add_argument("--step-min", type=int, default=30, help="Minutes between frames")
    parser.add_argument("--max-images", type=int, default=24, help="Max images per run")
    parser.add_argument("--timeout", type=int, default=1800, help="Timeout in seconds")
    parser.add_argument("--out", type=str, default="detections")
    args = parser.parse_args()
    
    # Set timeout signal
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.timeout)
    
    try:
        out_dir = pathlib.Path(args.out)
        ensure_dir(out_dir)
        
        print(f"=== SOHO Comet Detection ===")
        print(f"Hours: {args.hours}, Step: {args.step_min}min, Max Images: {args.max_images}")
        
        # Fetch images
        print("\n[1/4] Fetching images...")
        fetched = fetch_window(
            hours_back=args.hours, 
            step_min=args.step_min, 
            root="frames",
            max_frames=args.max_images // 2  # Split between C2 and C3
        )
        print(f"  Fetched {len(fetched)} new files")
        
        # Load series
        print("\n[2/4] Loading series...")
        all_c2 = load_series(pathlib.Path("frames") / "C2")
        all_c3 = load_series(pathlib.Path("frames") / "C3")
        
        # Filter by time
        series_c2 = [item for item in all_c2 if is_within_last_hours(item[0], hours=args.hours)]
        series_c3 = [item for item in all_c3 if is_within_last_hours(item[0], hours=args.hours)]
        
        print(f"  C2: {len(series_c2)} frames, C3: {len(series_c3)} frames")
        
        # Process detectors
        print("\n[3/4] Processing detectors...")
        max_frames_per_det = args.max_images // 2
        
        c2_candidates, c2_stats = process_detector("C2", series_c2, max_frames_per_det)
        c3_candidates, c3_stats = process_detector("C3", series_c3, max_frames_per_det)
        
        all_candidates = c2_candidates + c3_candidates
        
        # AI Classification (if available)
        print("\n[4/4] AI Classification...")
        if all_candidates and hasattr(sys.modules[__name__], 'classify_crop_batch'):
            try:
                crop_paths = [c["crop_path"] for c in all_candidates if "crop_path" in c]
                if crop_paths:
                    ai_results = classify_crop_batch(crop_paths)
                    for cand, ai in zip(all_candidates, ai_results):
                        cand["ai_label"] = ai.get("label", "unknown")
                        cand["ai_score"] = float(ai.get("score", 0.0))
            except Exception as e:
                print(f"  AI classification failed: {e}")
        
        # Create output
        timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        
        summary = {
            "timestamp_utc": timestamp,
            "hours_back": args.hours,
            "step_min": args.step_min,
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
        
        # Summary
        with open(out_dir / "latest_status.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Candidates
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
        print(f"Candidates: {len(all_candidates)}")
        print(f"Comet candidates (AI score ≥ {AI_MIN_SCORE}): {summary['comet_candidates']}")
        
        # Disable alarm
        signal.alarm(0)
        
    except TimeoutException:
        print("\n!!! TIMEOUT !!!")
        print("Detection timed out. Creating empty results.")
        
        # Create empty results
        out_dir = pathlib.Path(args.out)
        ensure_dir(out_dir)
        
        timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        
        empty_summary = {
            "timestamp_utc": timestamp,
            "error": "timeout",
            "detectors": {},
            "candidates_count": 0,
            "comet_candidates": 0,
            "generated_at": timestamp
        }
        
        with open(out_dir / "latest_status.json", "w") as f:
            json.dump(empty_summary, f, indent=2)
        
        with open(out_dir / "candidates_latest.json", "w") as f:
            json.dump([], f)
        
        sys.exit(0)  # Exit cleanly
        
    except Exception as e:
        print(f"\n!!! ERROR: {e}")
        raise

if __name__ == "__main__":
    main()
