import cv2, json, os, math, pathlib, argparse, numpy as np
from datetime import datetime
from typing import List, Tuple
from fetch_lasco import fetch_window
from pathlib import Path

# ---------- helpers (draw + sheets) ----------
def draw_tracks_overlay(base_img: np.ndarray, tracks, out_path: Path, radius=3, thickness=1):
    vis = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    for tr in tracks:
        pts = [(int(x), int(y)) for (_,x,y,_) in tr]
        for p in pts:
            cv2.circle(vis, p, radius, (0,255,255), -1)
        for i in range(1, len(pts)):
            cv2.line(vis, pts[i-1], pts[i], (0,200,255), thickness)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)

def contact_sheet(images: list[np.ndarray], cols=4, pad=4):
    if not images:
        return None
    h, w = images[0].shape[:2]
    rows = int(np.ceil(len(images)/cols))
    sheet = np.full((rows*h + (rows+1)*pad, cols*w + (cols+1)*pad), 10, dtype=np.uint8)
    idx = 0
    y = pad
    for r in range(rows):
        x = pad
        for c in range(cols):
            if idx < len(images):
                img = images[idx]
                if img.shape != (h,w):
                    img = cv2.resize(img, (w,h))
                sheet[y:y+h, x:x+w] = img
            x += w + pad
            idx += 1
        y += h + pad
    return sheet

# ---------- load + stabilize ----------
def load_series(folder: pathlib.Path):
    pairs = []
    for p in sorted(folder.glob("*.png")):
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is not None:
            pairs.append((p.name, im))
    return pairs

def stabilize(base: np.ndarray, img: np.ndarray) -> np.ndarray:
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _cc, warp = cv2.findTransformECC(base, img, warp, cv2.MOTION_EUCLIDEAN,
                                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4))
        aligned = cv2.warpAffine(img, warp, (img.shape[1], img.shape[0]),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return aligned
    except cv2.error:
        return img

# ---------- moving-point detection ----------
def find_moving_points(series: List[Tuple[str, np.ndarray]]) -> List[Tuple[int, float, float, float]]:
    pts = []
    for i in range(1, len(series)):
        _, a = series[i-1]
        _, b = series[i]
        diff = cv2.absdiff(b, a)
        blur = cv2.GaussianBlur(diff, (5,5), 0)
        thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 31, -5)
        clean = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
        cnts,_ = cv2.findContours(clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if 3 <= area <= 200:
                (x,y), _ = cv2.minEnclosingCircle(c)
                pts.append((i, float(x), float(y), float(area)))
    return pts

def link_tracks(points, min_len=3, max_jump=25):
    tracks = []
    by_t = {}
    for t,x,y,a in points:
        by_t.setdefault(t, []).append((x,y,a))
    for t in sorted(by_t.keys()):
        for x,y,a in by_t[t]:
            best = None; bi = -1
            for i,tr in enumerate(tracks):
                if tr[-1][0] == t-1:
                    dx = x - tr[-1][1]; dy = y - tr[-1][2]
                    if dx*dx + dy*dy <= max_jump*max_jump:
                        d = math.hypot(dx,dy)
                        if best is None or d < best:
                            best, bi = d, i
            if bi >= 0:
                tracks[bi].append((t,x,y,a))
            else:
                tracks.append([(t,x,y,a)])
    return [tr for tr in tracks if len(tr) >= min_len]

def score_track(tr):
    coords = np.array([[x,y] for (_,x,y,_) in tr])
    v = coords[-1]-coords[0]
    L = np.linalg.norm(v) + 1e-6
    proj = coords[0] + np.outer(np.linspace(0,1,len(coords)), v)
    err = np.mean(np.linalg.norm(coords - proj, axis=1))
    speed = L / (len(coords)-1)
    return float(1.0/(err+1e-3) + 0.2*speed)

def crop_along_track(img, tr, pad=16):
    _, x0,y0,_ = tr[0]; _, x1,y1,_ = tr[-1]
    x_min = int(max(0, min(x0,x1)-pad)); y_min = int(max(0, min(y0,y1)-pad))
    x_max = int(min(img.shape[1], max(x0,x1)+pad)); y_max = int(min(img.shape[0], max(y0,y1)+pad))
    return img[y_min:y_max, x_min:x_max]

# ---------- optional AI second-stage ----------
def maybe_classify_with_ai(hits: list[dict]) -> list[dict]:
    if os.getenv("USE_AI_CLASSIFIER", "0") != "1":
        return hits
    try:
        from ai_classifier import classify_crop_batch
    except Exception:
        return hits
    crops = [h["crop_path"] for h in hits]
    scores = classify_crop_batch(crops)
    for h, s in zip(hits, scores):
        h["ai_score"] = float(s.get("score", 0.0))
        h["ai_label"] = s.get("label", "unknown")
    return hits

# ---------- main pipeline ----------
def process_detector(detector_name: str, out_dir: pathlib.Path, debug=False):
    folder = pathlib.Path("frames") / detector_name
    series = load_series(folder)
    hits = []
    if len(series) < 4:
        return hits, {"frames": len(series), "tracks": 0}

    base = series[0][1]
    aligned = [(series[0][0], base)]
    for name, im in series[1:]:
        aligned.append((name, stabilize(base, im)))

    pts = find_moving_points(aligned)
    tracks = link_tracks(pts, min_len=3)
    track_count = len(tracks)
    if track_count:
        tracks = sorted(tracks, key=score_track, reverse=True)[:5]

    mid_name, mid_img = aligned[len(aligned)//2]

    for i,tr in enumerate(tracks):
        crop = crop_along_track(mid_img, tr)
        if crop.size == 0:
            continue
        det_dir = out_dir
        det_dir.mkdir(parents=True, exist_ok=True)
        crop_path = det_dir / f"{detector_name}_{mid_name}_track{i+1}.png"
        cv2.imwrite(str(crop_path), crop)
        hits.append({
            "detector": detector_name,
            "series_mid_frame": mid_name,
            "track_index": i+1,
            "crop_path": str(crop_path)
        })

    if debug:
        overlay_path = out_dir / f"overlay_{detector_name}.png"
        draw_tracks_overlay(mid_img, tracks, overlay_path)
        sheet = contact_sheet([im for _,im in aligned[-8:]])
        if sheet is not None:
            cv2.imwrite(str(out_dir / f"contact_{detector_name}.png"), sheet)

    return hits, {"frames": len(series), "tracks": track_count}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=6)
    ap.add_argument("--step-min", type=int, default=12, help="minutes between frames")
    ap.add_argument("--out", type=str, default="detections")
    args = ap.parse_args()

    debug = os.getenv("DETECTOR_DEBUG", "0") == "1"

    fetched = fetch_window(hours_back=args.hours, step_min=args.step_min, root="frames")
    print(f"Fetched {len(fetched)} new frames.")

    out_dir = pathlib.Path(args.out)
    all_hits = []
    summary = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "hours_back": args.hours,
        "step_min": args.step_min,
        "detectors": {},
        "fetched_new_frames": len(fetched),
        "errors": []
    }

    for det in ["C2", "C3"]:
        try:
            hits, stats = process_detector(det, out_dir, debug=debug)
            for h in hits:
                h["timestamp_utc"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            all_hits.extend(hits)
            summary["detectors"][det] = stats
        except Exception as e:
            print(f"[ERROR] {det} failed: {e}")
            summary["detectors"][det] = {"frames": 0, "tracks": 0}
            summary["errors"].append(f"{det}: {repr(e)}")

    all_hits = maybe_classify_with_ai(all_hits)

    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "latest_status.json"
    with open(status_path, "w") as f:
        json.dump({**summary, "candidates_in_report": len(all_hits)}, f, indent=2)
    print(f"Wrote status: {status_path}")

    if all_hits:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_json = out_dir / f"candidates_{ts}.json"
        with open(out_json, "w") as f:
            json.dump(all_hits, f, indent=2)
        print(f"Wrote {out_json}")

        url = os.getenv("ALERT_WEBHOOK_URL")
        if url:
            try:
                import requests
                text = f"SOHO comet candidates: {len(all_hits)} — {out_json.name}"
                requests.post(url, json={"text": text}, timeout=10)
            except Exception:
                pass
    else:
        print("No candidates this run.")

if __name__ == "__main__":
    main()
