# Add at the top after imports:
import traceback
import gc

# Replace the main() function with this optimized version:
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=4, help="Hours to look back")
    parser.add_argument("--step-min", type=int, default=60, help="Minutes between frames")
    parser.add_argument("--max-images", type=int, default=8, help="Max images per run")
    parser.add_argument("--timeout", type=int, default=840, help="Timeout in seconds")
    parser.add_argument("--out", type=str, default="detections")
    parser.add_argument("--skip-ai", action="store_true", help="Skip AI classification")
    args = parser.parse_args()
    
    # Strict memory management
    gc.enable()
    
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(args.timeout)
        
        out_dir = pathlib.Path(args.out)
        ensure_dir(out_dir)
        
        print(f"=== SOHO Comet Detection (QUICK MODE) ===")
        print(f"Hours: {args.hours}, Step: {args.step_min}min, Max Images: {args.max_images}")
        
        # Fetch images - small batch
        print("\n[1/3] Fetching limited images...")
        try:
            fetched = fetch_window(
                hours_back=args.hours, 
                step_min=args.step_min, 
                root="frames",
                max_frames=4  # Max 2 per detector
            )
            print(f"  Fetched {len(fetched)} new files")
        except Exception as e:
            print(f"  Warning: Fetch failed: {e}")
            fetched = []
        
        # Load series with strict limits
        print("\n[2/3] Loading series (max 4 frames each)...")
        series_c2 = load_series(pathlib.Path("frames") / "C2")
        series_c3 = load_series(pathlib.Path("frames") / "C3")
        
        # Take only the most recent frames
        series_c2 = series_c2[-4:] if len(series_c2) > 4 else series_c2
        series_c3 = series_c3[-4:] if len(series_c3) > 4 else series_c3
        
        print(f"  C2: {len(series_c2)} frames, C3: {len(series_c3)} frames")
        
        # Quick processing - skip if no frames
        all_candidates = []
        summary = {
            "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "hours_back": args.hours,
            "step_min": args.step_min,
            "detectors": {},
            "fetched_new_frames": len(fetched),
            "candidates_count": 0,
            "comet_candidates": 0,
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z"
        }
        
        if series_c2 or series_c3:
            print("\n[3/3] Quick processing...")
            if series_c2:
                c2_candidates, c2_stats = process_detector("C2", series_c2, max_frames=2)
                all_candidates.extend(c2_candidates)
                summary["detectors"]["C2"] = c2_stats
            
            if series_c3:
                c3_candidates, c3_stats = process_detector("C3", series_c3, max_frames=2)
                all_candidates.extend(c3_candidates)
                summary["detectors"]["C3"] = c3_stats
            
            # Skip AI to save time
            if all_candidates and not args.skip_ai:
                print("  Skipping AI classification for speed")
                for cand in all_candidates:
                    cand["ai_label"] = "not_evaluated"
                    cand["ai_score"] = 0.0
            
            summary["candidates_count"] = len(all_candidates)
            summary["comet_candidates"] = len([c for c in all_candidates if c.get("ai_score", 0) >= AI_MIN_SCORE])
        
        # Write minimal output
        print(f"\n=== COMPLETE ({(datetime.utcnow() - start_time).seconds}s) ===")
        print(f"Candidates: {len(all_candidates)}")
        
        with open(out_dir / "latest_status.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        with open(out_dir / "candidates_latest.json", "w") as f:
            json.dump(all_candidates, f)
        
        # Force cleanup
        gc.collect()
        signal.alarm(0)
        
    except TimeoutException:
        print("\n!!! TIMEOUT - Creating empty results")
        create_empty_results(args.out)
        sys.exit(0)
        
    except Exception as e:
        print(f"\n!!! ERROR: {e}")
        traceback.print_exc()
        create_empty_results(args.out)
        sys.exit(0)

def create_empty_results(out_dir: str):
    """Create empty result files"""
    out_path = pathlib.Path(out_dir)
    ensure_dir(out_path)
    
    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    
    empty_summary = {
        "timestamp_utc": timestamp,
        "error": "processing_error",
        "detectors": {},
        "candidates_count": 0,
        "comet_candidates": 0,
        "generated_at": timestamp
    }
    
    with open(out_path / "latest_status.json", "w") as f:
        json.dump(empty_summary, f, indent=2)
    
    with open(out_path / "candidates_latest.json", "w") as f:
        json.dump([], f)

# Add this after main() definition:
if __name__ == "__main__":
    start_time = datetime.utcnow()
    main()
