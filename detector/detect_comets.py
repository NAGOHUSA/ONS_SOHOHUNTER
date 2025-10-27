# detector/detect_comets.py
# End-to-end SOHO comet detector pipeline:
# - Fetch recent C2/C3 frames (via fetch_lasco.fetch_window)
# - Stabilize, build static star/hot-pixel mask, motion-diff, link tracks
# - Radial (occulter/edge) guard
# - Optional AI classifier stub annotation (ai_classifier.classify_crop_batch)
# - C2<->C3 correlation by position angle (PA) within time window
# - Exports: per-candidate TXT/CSV, overlays, thumbnails, latest_status.json,
#            candidates_<ts>.json, to_submit.json, and combined CSV per run.

from __future__ import annotations

import os, re, math, csv, json, argparse, pathlib
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
import requests

from fetch_lasco import fetch_window  # must exist in detector/

# ----------------- Config (env) -----------------
AI_VETO_ENABLED            = os.getenv("AI_VETO_ENABLED", "1") == "1"
AI_VETO_LABEL              = os.getenv("AI_VETO_LABEL", "not_comet")
AI_VETO_SCORE_MAX          = float(os.getenv("AI_VETO_SCORE_MAX", "0.9"))
ALERT_WEBHOOK_URL          = os.getenv("ALERT_WEBHOOK_URL", "").strip()

SELECT_TOP_N_FOR_SUBMIT    = int(os.getenv("SELECT_TOP_N_FOR_SUBMIT", "3"))

OCCULTER_RADIUS_FRACTION   = float(os.getenv("OCCULTER_RADIUS_FRACTION", "0.18"))
MAX_EDGE_RADIUS_FRACTION   = float(os.getenv("MAX_EDGE_RADIUS_FRACTION", "0.98"))

DUAL_CHANNEL_MAX_MINUTES   = int(os.getenv("DUAL_CHANNEL_MAX_MINUTES", "60"))
DUAL_CHANNEL_MAX_ANGLE_DIFF= int(os.getenv("DUAL_CHANNEL_MAX_ANGLE_DIFF", "25"))

DEBUG_OVERLAYS             = os.getenv("DETECTOR_DEBUG", "0") == "1"

# ----------------- IO helpers -----------------
def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def save_png(path: Path, img: np.ndarray) -> None:
    ensure_dir(path)
    cv2.imwrite(str(path), img)

# ----------------- Vis helpers -----------------
def draw_tracks_overlay(base_img: np.ndarray, tracks, out_path: Path, radius=3, thickness=1):
    """Draw tracks on a BGR copy of the grayscale base image."""
    if len(base_img.shape) == 2:
        vis_bgr = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    else:
        vis_bgr = base_img.copy()
    h, w = base_img.shape[:2]
    cv2.line(vis_bgr, (w//2, 0), (w//2, h), (40,40,40), 1)
    cv2.line(vis_bgr, (0, h//2), (w, h//2), (40,40,40), 1)
    for idx, tr in enumerate(tracks, 1):
        pts = [(int(x), int(y)) for (_,x,y,_) in tr]
        for p in pts:
            cv2.circle(vis_bgr, p, radius, (0,255,255), -1)
        for i in range(1, len(pts)):
            cv2.line(vis_bgr, pts[i-1], pts[i], (0,200,255), thickness)
        if pts:
            cv2.putText(vis_bgr, f"#{idx}", (pts[-1][0]+6, pts[-1][1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,215,255), 1, cv2.LINE_AA)
    save_png(out_path, vis_bgr)

def contact_sheet(images: List[np.ndarray], cols=4, pad=4) -> Optional[np.ndarray]:
    if not images:
        return None
    h, w = images[0].shape[:2]
    rows = int(np.ceil(len(images)/cols))
    sheet = np.full((rows*h + (rows+1)*pad, cols*w + (cols+1)*pad), 10, dtype=np.uint8)
    idx = 0
    y = pad
    for _r in range(rows):
        x = pad
        for _c in range(cols):
            if idx < len(images):
                img = images[idx]
                if img.shape[:2] != (h, w):
                    img = cv2.resize(img, (w,h))
                sheet[y:y+h, x:x+w] = img
            x += w + pad
            idx += 1
        y += h + pad
    return sheet

def save_thumbnail(img: np.ndarray, out_path: Path, max_w=960):
    h, w = img.shape[:2]
    disp = img
    if w and w > max_w:
        scale = max_w / w
        disp = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    save_png(out_path, disp)

def parse_frame_iso(name: str) -> Optional[str]:
    m = re.search(r'(\d{8})_(\d{4,6})', name)
    if not m:
        return None
    d, t = m.groups()
    if len(t) == 4:
        t = t + "00"
    try:
        return f"{d[0:4]}-{d[4:6]}-{d[6:8]}T{t[0:2]}:{t[2:4]}:{t[4:6]}Z"
    except Exception:
        return None

def make_annotated_thumb(gray_img: np.ndarray, detector: str, last_name: str,
                         hours_back: int, step_min: int, frames: int, tracks, out_path: Path):
    vis = gray_img.copy() if len(gray_img.shape) == 2 else cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    for idx, tr in enumerate(tracks, 1):
        pts = [(int(x), int(y)) for (_,x,y,_) in tr]
        for p in pts:
            cv2.circle(vis_bgr, p, 3, (0,255,255), -1)
        for i in range(1, len(pts)):
            cv2.line(vis_bgr, pts[i-1], pts[i], (0,200,255), 1)
        if pts:
            cv2.putText(vis_bgr, f"#{idx}", (pts[-1][0]+6, pts[-1][1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,215,255), 1, cv2.LINE_AA)
    h, w = vis_bgr.shape[:2]
    banner_h = 64
    overlay = vis_bgr.copy()
    cv2.rectangle(overlay, (0,0), (w, banner_h), (12, 22, 45), -1)
    vis_bgr = cv2.addWeighted(overlay, 0.7, vis_bgr, 0.3, 0)

    iso = parse_frame_iso(last_name) or ""
    lines = [
        f"SOHO Comet Hunter — {detector}",
        f"Last frame: {last_name}  {iso}",
        f"Window: {hours_back}h  Step: {step_min}m  | Frames: {frames}  Tracks: {len(tracks)}",
    ]
    y = 22
    for line in lines:
        cv2.putText(vis_bgr, line, (12, y), cv2.FONT_HERSHEY_DUPLEX, 0.6, (230,238,252), 1, cv2.LINE_AA)
        y += 20

    save_png(out_path, vis_bgr)

# ----------------- Data loading & stabilization -----------------
def load_series(folder: pathlib.Path) -> List[Tuple[str, np.ndarray]]:
    pairs: List[Tuple[str, np.ndarray]] = []
    if not folder.exists():
        return pairs
    for p in sorted(folder.glob("*.*")):
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is not None:
            pairs.append((p.name, im))
    return pairs

def stabilize(base: np.ndarray, img: np.ndarray) -> np.ndarray:
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _cc, warp = cv2.findTransformECC(
            base.astype(np.uint8), img.astype(np.uint8), warp, cv2.MOTION_EUCLIDEAN,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
        )
        aligned = cv2.warpAffine(img, warp, (img.shape[1], img.shape[0]),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return aligned
    except cv2.error:
        return img

# ----------------- Static mask (stars/hot pixels) -----------------
def build_static_mask(images: List[np.ndarray], ksize=5, thresh=10) -> np.ndarray:
    if len(images) < 3:
        return np.zeros_like(images[0], dtype=np.uint8)
    stack = np.stack(images, axis=0)
    med = np.median(stack, axis=0).astype(np.uint8)
    blur = cv2.GaussianBlur(med, (ksize, ksize), 0)
    _, mask = cv2.threshold(blur, np.median(blur)+thresh, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3,3), np.uint8), iterations=1)
    return mask

# ----------------- Motion detection & tracking -----------------
def find_moving_points(series: List[Tuple[str, np.ndarray]], static_mask=None) -> List[Tuple[int, float, float, float]]:
    pts: List[Tuple[int, float, float, float]] = []
    for i in range(1, len(series)):
        _, a = series[i-1]
        _, b = series[i]
        diff = cv2.absdiff(b, a)
        if static_mask is not None:
            diff = cv2.bitwise_and(diff, cv2.bitwise_not(static_mask))
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
    by_t: Dict[int, List[Tuple[float,float,float]]] = {}
    for t,x,y,a in points:
        by_t.setdefault(t, []).append((x,y,a))
    for t in sorted(by_t.keys()):
        for x,y,a in by_t[t]:
            best = None; bi = -1
            for i,tr in enumerate(tracks):
                if tr[-1][0] == t-1:
                    dx = x - tr[-1][1]; dy = y - tr[-1][2]
                    if dx*dx + dy*dy <= max_jump*max_jump:
                        d = float(np.hypot(dx,dy))
                        if best is None or d < best:
                            best, bi = d, i
            if bi >= 0:
                tracks[bi].append((t,x,y,a))
            else:
                tracks.append([(t,x,y,a)])
    return [tr for tr in tracks if len(tr) >= min_len]

def score_track(tr) -> float:
    coords = np.array([[x,y] for (_,x,y,_) in tr], dtype=np.float32)
    v = coords[-1] - coords[0]
    L = float(np.linalg.norm(v) + 1e-6)
    proj = coords[0] + np.outer(np.linspace(0,1,len(coords)), v)
    err = float(np.mean(np.linalg.norm(coords - proj, axis=1)))
    speed = L / max(1, (len(coords)-1))
    return float(1.0/(err+1e-3) + 0.2*speed)

def crop_along_track(img: np.ndarray, tr, pad=16) -> np.ndarray:
    _, x0,y0,_ = tr[0]; _, x1,y1,_ = tr[-1]
    x_min = int(max(0, min(x0,x1)-pad)); y_min = int(max(0, min(y0,y1)-pad))
    x_max = int(min(img.shape[1], max(x0,x1)+pad)); y_max = int(min(img.shape[0], max(y0,y1)+pad))
    return img[y_min:y_max, x_min:x_max]

# ----------------- AI classifier (optional) -----------------
def maybe_classify_with_ai(hits: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
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

# ----------------- Guards -----------------
def radial_guard_ok(x: float, y: float, w: int, h: int, r_min_frac: float, r_max_frac: float) -> bool:
    cx, cy = w/2.0, h/2.0
    r = math.hypot(x-cx, y-cy)
    rmax = min(cx, cy)
    rfrac = r / max(1e-6, rmax)
    return (r_min_frac <= rfrac <= r_max_frac)

# ----------------- Sungrazer exports per track -----------------
def write_sungrazer_exports(detector_name: str, track_idx: int, positions: List[Dict[str,Any]],
                            image_size: Tuple[int,int], out_dir: Path) -> Tuple[str,str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    first = positions[0] if positions else {}
    date_first = (first.get("time_utc","") or "").split("T")[0]
    img_w, img_h = image_size
    lines = []
    lines.append("=== Sungrazer Report Helper ===")
    lines.append(f"Camera: {detector_name}")
    lines.append(f"Image Size: {img_w}x{img_h}")
    lines.append("Your (0,0) position: Upper Left")
    lines.append(f"Date of FIRST IMAGE: {date_first}")
    lines.append("Frames (time_utc, x, y):")
    for p in positions:
        t = (p["time_utc"] or "").replace("T"," ").replace("Z","")
        lines.append(f"  {t}, {int(round(p['x']))}, {int(round(p['y']))}")
    txt = "\n".join(lines) + "\n"
    txt_path = out_dir / f"{detector_name}_track{track_idx}_sungrazer.txt"
    with open(txt_path, "w") as f:
        f.write(txt)

    csv_path = out_dir / f"{detector_name}_track{track_idx}_sungrazer.csv"
    with open(csv_path, "w") as f:
        f.write("frame_time_utc,x,y\n")
        for p in positions:
            f.write(f"{p['time_utc']},{int(round(p['x']))},{int(round(p['y']))}\n")
    return str(txt_path), str(csv_path)

# ----------------- Detector pipeline (per C2/C3) -----------------
def process_detector(detector_name: str, out_dir: Path,
                     debug: bool=False, hours_back: int=6, step_min: int=12):
    folder = Path("frames") / detector_name
    series = load_series(folder)
    hits: List[Dict[str,Any]] = []

    if len(series) == 0:
        return hits, {"frames": 0, "tracks": 0, "last_frame_name": "", "last_frame_iso": "", "last_frame_size": [0,0]}

    last_name, last_img = series[-1]
    if debug:
        save_thumbnail(last_img, out_dir / f"lastframe_{detector_name}.png", max_w=960)
        make_annotated_thumb(last_img, detector_name, last_name, hours_back, step_min, len(series), [],
                             out_dir / f"lastthumb_{detector_name}.png")

    tracks = []
    if len(series) >= 4:
        base = series[0][1]
        aligned: List[Tuple[str,np.ndarray]] = [(series[0][0], base)]
        for name, im in series[1:]:
            aligned.append((name, stabilize(base, im)))

        names = [name for name,_ in aligned]
        images = [img for _,img in aligned]

        static_mask = build_static_mask(images[-min(8, len(images)):])
        pts_all = find_moving_points(aligned, static_mask=static_mask)

        w, h = images[0].shape[1], images[0].shape[0]
        guarded = [(t,x,y,a) for (t,x,y,a) in pts_all
                   if radial_guard_ok(x,y,w,h,OCCULTER_RADIUS_FRACTION,MAX_EDGE_RADIUS_FRACTION)]

        tracks = link_tracks(guarded, min_len=3)
        if tracks:
            tracks = sorted(tracks, key=score_track, reverse=True)[:8]

        mid_name, mid_img = aligned[len(aligned)//2]

        for i, tr in enumerate(tracks):
            positions: List[Dict[str,Any]] = []
            for (t_idx, x, y, _a) in tr:
                fname = names[t_idx]
                iso = parse_frame_iso(fname) or ""
                positions.append({"frame": fname, "time_utc": iso, "x": float(x), "y": float(y)})

            crop = crop_along_track(mid_img, tr)
            if crop.size == 0:
                continue
            crop_path = out_dir / f"{detector_name}_{mid_name}_track{i+1}.png"
            save_png(crop_path, crop)

            write_sungrazer_exports(detector_name, i+1, positions, image_size=(w,h), out_dir=out_dir / "reports")

            hits.append({
                "detector": detector_name,
                "series_mid_frame": mid_name,
                "track_index": i+1,
                "crop_path": str(crop_path),
                "positions": positions,
                "image_size": [int(w), int(h)],
                "origin": "upper_left"
            })

        if DEBUG_OVERLAYS:
            draw_tracks_overlay(mid_img, tracks, out_dir / f"overlay_{detector_name}.png")
            sheet = contact_sheet([im for im in images[-8:]])
            if sheet is not None:
                save_png(out_dir / f"contact_{detector_name}.png", sheet)

    lh, lw = last_img.shape[:2]
    return hits, {
        "frames": len(series),
        "tracks": len(tracks),
        "last_frame_name": last_name,
        "last_frame_iso": parse_frame_iso(last_name) or "",
        "last_frame_size": [int(lw), int(lh)]
    }

# ----------------- Cross-detector correlation -----------------
def pa_of_track_from_positions(positions: List[Dict[str,Any]], w: int, h: int) -> Optional[float]:
    if not positions or len(positions) < 2:
        return None
    x0, y0 = positions[0]["x"], positions[0]["y"]
    x1, y1 = positions[-1]["x"], positions[-1]["y"]
    cx, cy = w/2.0, h/2.0
    vx, vy = (x1-cx)-(x0-cx), (y1-cy)-(y0-cy)
    ang = (math.degrees(math.atan2(-vx, -vy)) + 360.0) % 360.0  # PA: 0 up, CW
    return ang

def correlate_c2_c3(hits_c2: List[Dict[str,Any]], hits_c3: List[Dict[str,Any]]) -> None:
    """Annotate likely C2↔C3 matches by similar PA within time window."""
    def first_time(h: Dict[str,Any]) -> str:
        return (h.get("positions") or [{}])[0].get("time_utc", "")

    def parse_t(s: str) -> Optional[datetime]:
        try:
            return datetime.strptime(s.replace("Z",""), "%Y-%m-%dT%H:%M:%S")
        except Exception:
            return None

    # C2 -> C3
    for hc2 in hits_c2:
        t2 = parse_t(first_time(hc2))
        if not t2:
            continue
        w2, h2 = hc2["image_size"]
        pa2 = pa_of_track_from_positions(hc2.get("positions", []), w2, h2)
        best = None
        best_hit = None
        for hc3 in hits_c3:
            t3 = parse_t(first_time(hc3))
            if not t3:
                continue
            dt_min = abs((t3 - t2).total_seconds()) / 60.0
            if dt_min > DUAL_CHANNEL_MAX_MINUTES:
                continue
            w3, h3 = hc3["image_size"]
            pa3 = pa_of_track_from_positions(hc3.get("positions", []), w3, h3)
            if pa2 is None or pa3 is None:
                continue
            diff = min(abs(pa2 - pa3), 360 - abs(pa2 - pa3))
            if diff <= DUAL_CHANNEL_MAX_ANGLE_DIFF:
                if best is None or diff < best:
                    best = diff
                    best_hit = hc3
        if best_hit is not None and best is not None:
            hc2["dual_channel_match"] = {"with": f"C3#{best_hit['track_index']}", "pa_diff_deg": round(best, 1)}

    # C3 -> C2
    for hc3 in hits_c3:
        t3 = parse_t(first_time(hc3))
        if not t3:
            continue
        w3, h3 = hc3["image_size"]
        pa3 = pa_of_track_from_positions(hc3.get("positions", []), w3, h3)
        best = None
        best_hit = None
        for hc2 in hits_c2:
            t2 = parse_t(first_time(hc2))
            if not t2:
                continue
            dt_min = abs((t3 - t2).total_seconds()) / 60.0
            if dt_min > DUAL_CHANNEL_MAX_MINUTES:
                continue
            w2, h2 = hc2["image_size"]
            pa2 = pa_of_track_from_positions(hc2.get("positions", []), w2, h2)
            if pa2 is None or pa3 is None:
                continue
            diff = min(abs(pa2 - pa3), 360 - abs(pa2 - pa3))
            if diff <= DUAL_CHANNEL_MAX_ANGLE_DIFF:
                if best is None or diff < best:
                    best = diff
                    best_hit = hc2
        if best_hit is not None and best is not None:
            hc3["dual_channel_match"] = {"with": f"C2#{best_hit['track_index']}", "pa_diff_deg": round(best, 1)}

# ----------------- Webhook -----------------
def send_webhook(summary: Dict[str,Any], hits: List[Dict[str,Any]]) -> None:
    if not ALERT_WEBHOOK_URL:
        return
    try:
        payload = {
            "timestamp_utc": summary.get("timestamp_utc"),
            "fetched_new_frames": summary.get("fetched_new_frames", 0),
            "detectors": summary.get("detectors", {}),
            "top_candidates": [
                {
                    "detector": h.get("detector"),
                    "track_index": h.get("track_index"),
                    "series_mid_frame": h.get("series_mid_frame"),
                    "ai_label": h.get("ai_label"),
                    "ai_score": h.get("ai_score"),
                    "dual_channel_match": h.get("dual_channel_match"),
                    "crop_path": h.get("crop_path")
                } for h in hits[:5]
            ]
        }
        requests.post(ALERT_WEBHOOK_URL, json=payload, timeout=10)
    except Exception:
        pass

# ----------------- Combined CSV (backend) -----------------
def write_combined_csv(out_dir: Path, run_ts: str, hits: List[Dict[str,Any]]) -> str:
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / f"report_{run_ts}_combined.csv"
    header = ["detector","track_index","series_mid_frame","frame_time_utc","x","y",
              "image_width","image_height","origin","ai_label","ai_score",
              "dual_channel_with","dual_channel_pa_diff_deg","vetoed","auto_selected"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for h in hits:
            det = (h.get("detector") or "").upper()
            w_img,h_img = (h.get("image_size") or [None,None])
            ai_label = h.get("ai_label")
            ai_score = h.get("ai_score")
            dual = h.get("dual_channel_match") or {}
            for p in (h.get("positions") or []):
                w.writerow([
                    det, h.get("track_index"), h.get("series_mid_frame"),
                    p.get("time_utc"), int(round(p.get("x",0))), int(round(p.get("y",0))),
                    w_img, h_img, h.get("origin"),
                    ai_label, ai_score,
                    dual.get("with"), dual.get("pa_diff_deg"),
                    h.get("vetoed") is True, h.get("auto_selected") is True
                ])
    return str(csv_path)

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=6)
    ap.add_argument("--step-min", type=int, default=12, help="minutes between frames")
    ap.add_argument("--out", type=str, default="detections")
    args = ap.parse_args()

    fetched = fetch_window(hours_back=args.hours, step_min=args.step_min, root="frames")
    print(f"Fetched {len(fetched)} new frames.")
    for p in fetched:
        print(f" - {p}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    detectors_stats: Dict[str,Any] = {}
    errors: List[str] = []
    results: Dict[str, List[Dict[str,Any]]] = {"C2": [], "C3": []}

    # Run detectors
    for det in ["C2", "C3"]:
        try:
            hits, stats = process_detector(det, out_dir, debug=DEBUG_OVERLAYS,
                                           hours_back=args.hours, step_min=args.step_min)
            for h in hits:
                h["timestamp_utc"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            results[det] = hits
            detectors_stats[det] = stats
        except Exception as e:
            print(f"[ERROR] {det} failed: {e}")
            detectors_stats[det] = {"frames":0,"tracks":0,"last_frame_name":"","last_frame_iso":"","last_frame_size":[0,0]}
            errors.append(f"{det}: {repr(e)}")

    # AI annotate & soft veto
    for det in ["C2","C3"]:
        results[det] = maybe_classify_with_ai(results.get(det, []))
        if AI_VETO_ENABLED:
            kept = []
            for h in results[det]:
                lbl = (h.get("ai_label") or "").lower()
                sc  = float(h.get("ai_score") or 0.0)
                if lbl == AI_VETO_LABEL and sc >= AI_VETO_SCORE_MAX:
                    h["vetoed"] = True
                kept.append(h)
            results[det] = kept

    # Dual-channel correlation
    correlate_c2_c3(results.get("C2", []), results.get("C3", []))

    # Auto shortlist
    to_submit: List[Dict[str,Any]] = []
    for det in ["C2","C3"]:
        cands = results.get(det, [])
        # Prefer higher AI score; tie-break by having a dual-channel match
        cands_sorted = sorted(
            cands,
            key=lambda h: (
                float(h.get("ai_score") or 0.0),
                h.get("dual_channel_match") is not None
            ),
            reverse=True
        )
        chosen = [h for h in cands_sorted if not h.get("vetoed")][:SELECT_TOP_N_FOR_SUBMIT]
        for h in chosen:
            h["auto_selected"] = True
            to_submit.append({
                "detector": det,
                "track_index": h["track_index"],
                "series_mid_frame": h["series_mid_frame"],
                "crop_path": h["crop_path"]
            })

    # Flatten
    all_hits: List[Dict[str,Any]] = results.get("C2", []) + results.get("C3", [])

    # Summary & status
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

    # Write status
    with open(out_dir / "latest_status.json", "w") as f:
        json.dump({**summary, "candidates_in_report": len(all_hits)}, f, indent=2)
    print(f"Wrote status: {out_dir/'latest_status.json'}")

    # Write candidates json
    if all_hits:
        ts_name = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_json = out_dir / f"candidates_{ts_name}.json"
        with open(out_json, "w") as f:
            json.dump(all_hits, f, indent=2)
        print(f"Wrote {out_json}")

        # Combined CSV for the whole run
        combined_csv = write_combined_csv(out_dir, ts_name, all_hits)
        print(f"Wrote combined CSV: {combined_csv}")
    else:
        print("No candidates this run.")

    # to_submit.json (auto shortlist)
    with open(out_dir / "to_submit.json", "w") as f:
        json.dump({"auto_selected": to_submit, "timestamp_utc": summary["timestamp_utc"]}, f, indent=2)
    print(f"Wrote {out_dir/'to_submit.json'}")

    # Optional webhook
    non_vetoed = [h for h in all_hits if not h.get("vetoed")]
    if non_vetoed:
        send_webhook(summary, non_vetoed)

if __name__ == "__main__":
    main()
