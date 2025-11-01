#!/usr/bin/env python3
"""
SOHO Comet Detection Pipeline
- Fetches LASCO frames
- Detects moving objects
- Filters timestamp/edge artifacts
- Runs AI classification
- Saves results as JSON
"""

from datetime import datetime, timedelta
from pathlib import Path
import requests
import json
import cv2
import numpy as np
import os
import sys

# Add detector directory to path if needed
sys.path.insert(0, str(Path(__file__).parent))

try:
    from ai_classifier import classify_crop_batch
except ImportError:
    print("Warning: ai_classifier not found, using dummy classifier")
    def classify_crop_batch(paths):
        return [{"label": "unknown", "score": 0.0} for _ in paths]

# ==================== CONFIGURATION ====================
FRAMES_DIR = Path("frames")
DETECTIONS_DIR = Path("detections")
CROPS_DIR = DETECTIONS_DIR / "crops"

# Frame dimensions for LASCO (1024x1024 for standard images)
LASCO_C2_SIZE = (1024, 1024)
LASCO_C3_SIZE = (1024, 1024)

# AI configuration
USE_AI = os.getenv("USE_AI_CLASSIFIER", "1") == "1"
AI_VETO = os.getenv("AI_VETO_ENABLED", "1") == "1"
AI_VETO_LABEL = os.getenv("AI_VETO_LABEL", "not_comet")
AI_VETO_MAX = float(os.getenv("AI_VETO_SCORE_MAX", "0.9"))

# ==================== TIMESTAMP FILTERING ====================

def is_timestamp_artifact(bbox, frame_width, frame_height):
    """
    Check if bounding box is likely a timestamp overlay or corner artifact.
    
    Args:
        bbox: [x, y, width, height] of detection
        frame_width: width of source frame
        frame_height: height of source frame
    
    Returns:
        True if likely a timestamp/corner artifact, False otherwise
    """
    if len(bbox) < 4:
        return False
    
    x, y, w, h = bbox[:4]
    
    # Define corner margin (pixels from edge)
    corner_margin = 100
    
    # Check if detection is in a corner region
    in_top_left = (x < corner_margin and y < corner_margin)
    in_top_right = (x + w > frame_width - corner_margin and y < corner_margin)
    in_bottom_left = (x < corner_margin and y + h > frame_height - corner_margin)
    in_bottom_right = (x + w > frame_width - corner_margin and y + h > frame_height - corner_margin)
    
    in_corner = in_top_left or in_top_right or in_bottom_left or in_bottom_right
    
    # Check if detection is along an edge (not just corner)
    edge_margin = 50
    in_top_edge = y < edge_margin
    in_bottom_edge = y + h > frame_height - edge_margin
    in_left_edge = x < edge_margin
    in_right_edge = x + w > frame_width - edge_margin
    
    on_edge = in_top_edge or in_bottom_edge or in_left_edge or in_right_edge
    
    # Timestamps are typically small, horizontally-oriented rectangles
    aspect_ratio = w / h if h > 0 else 0
    is_text_shaped = 2.0 < aspect_ratio < 8.0 and w < 250 and h < 60
    
    # Also check for very small detections in corners (likely noise)
    is_tiny = (w < 30 or h < 30) and in_corner
    
    # Filter criteria
    if in_corner and (is_text_shaped or is_tiny):
        return True
    
    # Also filter edge artifacts that are text-shaped
    if on_edge and is_text_shaped and (w < 200 or h < 40):
        return True
    
    return False


def filter_detections_spatial(candidates, frame_width, frame_height):
    """
    Filter out spatial artifacts like timestamps and edge detections.
    
    Args:
        candidates: list of detection candidates with 'bbox' key
        frame_width: width of source frame
        frame_height: height of source frame
    
    Returns:
        Filtered list of candidates
    """
    filtered = []
    removed_count = 0
    
    for candidate in candidates:
        bbox = candidate.get('bbox', [])
        
        if is_timestamp_artifact(bbox, frame_width, frame_height):
            removed_count += 1
            track_id = candidate.get('track_id', '?')
            print(f"  ✓ Filtered timestamp/edge artifact track_{track_id} at bbox {bbox}")
            continue
        
        filtered.append(candidate)
    
    if removed_count > 0:
        print(f"  ✓ Removed {removed_count} timestamp/edge artifacts")
    
    return filtered


# ==================== FRAME FETCHING ====================

def fetch_lasco_frames(hours=6, step_min=12):
    """
    Fetch LASCO C2/C3 frames from NASA.
    Returns: (frame_paths, timestamps, download_count)
    """
    print(f"Fetching LASCO frames (last {hours}h, step {step_min}min)...")
    
    try:
        # Import from fetch_lasco if available
        from fetch_lasco import fetch_window
        frames = fetch_window(hours_back=hours, step_min=step_min, root=str(FRAMES_DIR))
        
        # Generate timestamps from filenames
        timestamps = []
        for frame_path in frames:
            try:
                # Parse timestamp from filename (e.g., C2_20251027_1248_c2_1024.jpg)
                parts = Path(frame_path).stem.split('_')
                if len(parts) >= 3:
                    date_str = parts[1]  # 20251027
                    time_str = parts[2]  # 1248
                    ts = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M")
                    timestamps.append(ts.strftime("%Y-%m-%dT%H:%M:%SZ"))
                else:
                    timestamps.append(datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))
            except:
                timestamps.append(datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))
        
        print(f"✓ Fetched {len(frames)} frames")
        return frames, timestamps, len(frames)
    
    except ImportError:
        print("Warning: fetch_lasco module not found, using fallback")
        # Fallback: look for existing frames
        frames = []
        for det in ["C2", "C3"]:
            det_dir = FRAMES_DIR / det
            if det_dir.exists():
                frames.extend([str(f) for f in sorted(det_dir.glob("*.jpg")) + sorted(det_dir.glob("*.png"))])
        
        timestamps = [datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")] * len(frames)
        return frames, timestamps, 0


# ==================== BASIC DETECTION ====================

def detect_candidates(frame_paths, timestamps):
    """
    Simple motion detection between frames.
    Returns list of candidates with bbox, timestamp, etc.
    """
    print(f"Running detection on {len(frame_paths)} frames...")
    
    if len(frame_paths) < 2:
        print("Need at least 2 frames for motion detection")
        return []
    
    candidates = []
    track_id = 0
    
    for i, frame_path in enumerate(frame_paths):
        try:
            # Read frame
            frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            if frame is None:
                continue
            
            frame_h, frame_w = frame.shape[:2]
            
            # Simple detection: find bright spots
            # (This is a placeholder - replace with your actual detection logic)
            _, thresh = cv2.threshold(frame, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Basic size filter
                if w < 10 or h < 10 or w > 200 or h > 200:
                    continue
                
                # Create candidate
                candidate = {
                    "instrument": "lasco",
                    "timestamp": timestamps[i] if i < len(timestamps) else "",
                    "track_id": track_id,
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "frame_path": frame_path,
                    "frame_width": frame_w,
                    "frame_height": frame_h
                }
                
                candidates.append(candidate)
                track_id += 1
        
        except Exception as e:
            print(f"Error processing {frame_path}: {e}")
            continue
    
    print(f"✓ Found {len(candidates)} initial detections")
    return candidates


# ==================== CROP EXTRACTION ====================

def extract_crops(candidates):
    """
    Extract crop images for each candidate.
    Returns updated candidates with crop_path.
    """
    print(f"Extracting {len(candidates)} crops...")
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    
    extracted = []
    
    for candidate in candidates:
        try:
            frame_path = candidate.get("frame_path")
            bbox = candidate.get("bbox", [])
            
            if not frame_path or len(bbox) < 4:
                continue
            
            # Read frame
            frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            if frame is None:
                continue
            
            # Extract crop with padding
            x, y, w, h = bbox
            pad = 10
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)
            
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue
            
            # Save crop
            timestamp = candidate.get("timestamp", "unknown").replace(":", "")
            track_id = candidate.get("track_id", 0)
            crop_filename = f"lasco_track{track_id}_{timestamp}.png"
            crop_path = CROPS_DIR / crop_filename
            
            cv2.imwrite(str(crop_path), crop)
            
            # Update candidate
            candidate["crop_path"] = f"crops/{crop_filename}"
            extracted.append(candidate)
        
        except Exception as e:
            print(f"Error extracting crop for track {candidate.get('track_id')}: {e}")
            continue
    
    print(f"✓ Extracted {len(extracted)} crops")
    return extracted


# ==================== AI CLASSIFICATION ====================

def classify_candidates(candidates):
    """
    Run AI classification on candidates.
    Returns updated candidates with ai_label and ai_score.
    """
    if not USE_AI:
        print("AI classification disabled")
        for c in candidates:
            c["ai_label"] = "unknown"
            c["ai_score"] = 0.0
        return candidates
    
    print(f"Running AI classification on {len(candidates)} candidates...")
    
    # Collect crop paths
    crop_paths = []
    valid_indices = []
    
    for i, candidate in enumerate(candidates):
        crop_path = candidate.get("crop_path", "")
        full_path = DETECTIONS_DIR / crop_path if crop_path else None
        
        if full_path and full_path.exists():
            crop_paths.append(str(full_path))
            valid_indices.append(i)
    
    if not crop_paths:
        print("No valid crops for classification")
        return candidates
    
    # Run batch classification
    try:
        results = classify_crop_batch(crop_paths)
        
        # Apply results
        comet_count = 0
        for idx, result in zip(valid_indices, results):
            candidates[idx]["ai_label"] = result.get("label", "unknown")
            candidates[idx]["ai_score"] = result.get("score", 0.0)
            
            # Apply veto if enabled
            if AI_VETO and result.get("label") == AI_VETO_LABEL and result.get("score", 0) > AI_VETO_MAX:
                candidates[idx]["ai_label"] = "vetoed"
            
            if candidates[idx]["ai_label"] == "comet":
                comet_count += 1
        
        print(f"✓ Classification complete: {comet_count} comets identified")
    
    except Exception as e:
        print(f"Error during classification: {e}")
        for c in candidates:
            if "ai_label" not in c:
                c["ai_label"] = "error"
                c["ai_score"] = 0.0
    
    return candidates


# ==================== SAVE RESULTS ====================

def save_results(candidates, output_dir):
    """
    Save detection results as JSON.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"candidates_{timestamp}.json"
    
    # Clean up candidates for JSON (remove frame_path, etc.)
    clean_candidates = []
    for c in candidates:
        clean = {
            "instrument": c.get("instrument"),
            "timestamp": c.get("timestamp"),
            "track_id": c.get("track_id"),
            "bbox": c.get("bbox"),
            "crop_path": c.get("crop_path"),
            "ai_label": c.get("ai_label", "unknown"),
            "ai_score": c.get("ai_score", 0.0)
        }
        clean_candidates.append(clean)
    
    with open(output_file, 'w') as f:
        json.dump(clean_candidates, f, indent=2)
    
    print(f"✓ Saved {len(clean_candidates)} candidates to {output_file}")
    
    # Also save summary
    summary = {
        "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_candidates": len(clean_candidates),
        "comets": len([c for c in clean_candidates if c["ai_label"] == "comet"]),
        "output_file": str(output_file)
    }
    
    summary_file = output_dir / "latest_run.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return output_file


# ==================== MAIN PIPELINE ====================

def main():
    """
    Main detection pipeline.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="SOHO Comet Detection")
    parser.add_argument("--hours", type=int, default=6, help="Hours of data to fetch")
    parser.add_argument("--step-min", type=int, default=12, help="Step size in minutes")
    parser.add_argument("--out", type=str, default="detections", help="Output directory")
    args = parser.parse_args()
    
    print("=" * 60)
    print("SOHO COMET HUNTER - Detection Pipeline")
    print("=" * 60)
    
    # Step 1: Fetch frames
    frame_paths, timestamps, downloaded = fetch_lasco_frames(
        hours=args.hours,
        step_min=args.step_min
    )
    
    if not frame_paths:
        print("No frames available for detection")
        return
    
    # Step 2: Detect candidates
    candidates = detect_candidates(frame_paths, timestamps)
    
    if not candidates:
        print("No candidates detected")
        return
    
    # Step 3: SPATIAL FILTERING (remove timestamp artifacts)
    if candidates:
        # Get frame dimensions from first candidate
        frame_w = candidates[0].get("frame_width", 1024)
        frame_h = candidates[0].get("frame_height", 1024)
        
        print(f"\nApplying spatial filters (frame size: {frame_w}x{frame_h})...")
        candidates = filter_detections_spatial(candidates, frame_w, frame_h)
        print(f"✓ {len(candidates)} candidates remain after filtering\n")
    
    if not candidates:
        print("All candidates filtered out")
        return
    
    # Step 4: Extract crops
    candidates = extract_crops(candidates)
    
    if not candidates:
        print("No crops extracted")
        return
    
    # Step 5: AI classification
    candidates = classify_candidates(candidates)
    
    # Step 6: Save results
    output_file = save_results(candidates, args.out)
    
    print("\n" + "=" * 60)
    print("DETECTION COMPLETE")
    print("=" * 60)
    print(f"Total detections: {len(candidates)}")
    print(f"AI comets: {len([c for c in candidates if c.get('ai_label') == 'comet'])}")
    print(f"Results: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
