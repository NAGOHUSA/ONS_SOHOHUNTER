import cv2, json, os, math, pathlib, argparse, numpy as np
from datetime import datetime
from typing import List, Tuple
from fetch_lasco import fetch_window

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

def process_detector(detector_name: str, out_dir: pathlib.Path):
    folder = pathlib.Path("frames") / detector_name
    series = load_series(folder)
    hits = []
    if len(series) < 4:
        return hits

    base = series[0][1]
    aligned = [(series[0][0], base)]
    for name, im in series[1:]:
        aligned.append((name, stabilize(base, im)))

    pts = find_moving_points(aligned)
    tracks = link_tracks(pts, min_len=3)
    if not tracks:
        return hits

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
    return hits

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=6)
    ap.add_argument("--step-min", type=int, default=12, help="minutes between frames")
    ap.add_argument("--out", type=str, default="detections")
    args = ap.parse_args()

    # 1) Fetch frames
    fetched = fetch_window(hours_back=args.hours, step_min=args.step_min, root="frames")
    print(f"Fetched {len(fetched)} new frames.")

    # 2) Run detector for C2 & C3
    out_dir = pathlib.Path(args.out)
    all_hits = []
    for det in ["C2", "C3"]:
        hits = process_detector(det, out_dir)
        for h in hits:
            h["timestamp_utc"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        all_hits.extend(hits)

    # 3) Optional AI second stage
    all_hits = maybe_classify_with_ai(all_hits)

    # 4) Persist report + notify
    if all_hits:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_json = out_dir / f"candidates_{ts}.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(all_hits, f, indent=2)
        print(f"Wrote {out_json}")

        # Optional webhook
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
