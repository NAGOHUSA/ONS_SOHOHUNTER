#!/usr/bin/env python3
"""
SOHO Comet Detection Pipeline (LATEST-driven)
- Parses NASA LATEST pages for LASCO C2/C3 (freshest frames)
- Downloads frames for the past --hours in --step-min cadence
- Simple tracker + local AI classifier (ai_classifier.py)
- Optional Groq vision on the top-scoring crops (rate-limited)
- Always writes:
    detections/latest_status.json   <-- includes embedded "candidates": [...]
    detections/latest_run.json
- Optionally writes (when any found):
    detections/candidates_YYYYMMDD_HHMMSS.json
- Also writes thumbnails of the last C2/C3 frame:
    detections/lastthumb_C2.png
    detections/lastthumb_C3.png
"""

import argparse, os, re, json, base64, tempfile
from pathlib import Path
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import requests

# -------- CLI / ENV --------
parser = argparse.ArgumentParser()
parser.add_argument("--hours", type=int, default=int(os.getenv("HOURS", "6")))
parser.add_argument("--step-min", type=int, default=int(os.getenv("STEP_MIN", "12")))
parser.add_argument("--out", default=os.getenv("OUT", "detections"))
args = parser.parse_args()

HOURS = args.hours
STEP_MIN = args.step_min
OUT_DIR = Path(args.out)
FRAMES_DIR = Path("frames")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_LIMIT_HOURS = 12
GROQ_MAX_CALLS = 2
LAST_CALL_FILE = OUT_DIR / "last_groq_call.txt"

# -------- LATEST page sources (more up-to-date than REPROCESSING) --------
LATEST_PAGES = {
    "C2": "https://soho.nascom.nasa.gov/data/LATEST/latest-lascoC2.html",
    "C3": "https://soho.nascom.nasa.gov/data/LATEST/latest-lascoC3.html",
}
# Typical img href on those pages includes *_c2_1024.jpg or *_c3_1024.jpg
IMG_HREF_RE = re.compile(r'href="([^"]*_(c2|c3)_1024\.jpg)"', re.IGNORECASE)
STAMP_RE = re.compile(r"/(\d{8})_(\d{4})_c[23]_1024\.jpg$", re.IGNORECASE)

def log(*a): print(*a)

# -------- ai classifier (local) --------
try:
    from ai_classifier import classify_crop_batch
except Exception as e:
    log("ai_classifier import failed; using dummy:", e)
    def classify_crop_batch(paths):
        return [{"label": "not_comet", "score": 0.0} for _ in paths]

# -------- utilities --------
def ensure_dirs():
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "crops").mkdir(parents=True, exist_ok=True)

def fetch_latest_list(instr: str):
    """Return list of (absolute_jpg_url, ymd, hm, iso) for instrument."""
    url = LATEST_PAGES[instr]
    try:
        html = requests.get(url, timeout=15).text
    except Exception as e:
        log(f"[{instr}] LATEST fetch failed:", e)
        return []

    items = []
    for m in IMG_HREF_RE.finditer(html):
        href = m.group(1)
        # Make absolute if necessary
        if href.startswith("//"):
            jpg_url = "https:" + href
        elif href.startswith("http"):
            jpg_url = href
        else:
            # the page tends to link relative to /data/
            jpg_url = "https://soho.nascom.nasa.gov" + href if href.startswith("/") else "https://soho.nascom.nasa.gov/data/" + href

        sm = STAMP_RE.search(jpg_url)
        if not sm:
            continue
        ymd, hm = sm.group(1), sm.group(2)
        iso = f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}T{hm[:2]}:{hm[2:]}:00Z"
        items.append((jpg_url, ymd, hm, iso))

    # Unique + sorted by iso time ascending
    seen = set()
    uniq = []
    for it in items:
        if it[0] in seen: continue
        seen.add(it[0])
        uniq.append(it)
    uniq.sort(key=lambda x: x[3])
    return uniq

def within_window(iso_str: str, now_utc):
    t = datetime.strptime(iso_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return (now_utc - t) <= timedelta(hours=HOURS)

def matches_step(prev_iso: str, cur_iso: str):
    """Return True if cur is roughly STEP_MIN after prev (±2 min to be forgiving)."""
    p = datetime.strptime(prev_iso, "%Y-%m-%dT%H:%M:%SZ")
    c = datetime.strptime(cur_iso, "%Y-%m-%dT%H:%M:%SZ")
    delta = abs((c - p).total_seconds() / 60.0 - STEP_MIN)
    return delta <= 2.0 or prev_iso == ""  # always allow first

def download_frames_from_latest(instr: str):
    """Download frames guided by LATEST page, honoring window + step cadence."""
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    catalog = fetch_latest_list(instr)
    kept = []
    kept_iso = []
    last_kept = ""

    for jpg_url, ymd, hm, iso in catalog:
        if not within_window(iso, now):
            continue
        if last_kept and not matches_step(last_kept, iso):
            continue

        fname = f"{instr}_{ymd}_{hm}.jpg"
        out = FRAMES_DIR / fname
        if not out.exists():
            try:
                r = requests.get(jpg_url, timeout=20)
                if r.status_code == 200:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(r.content)
                else:
                    continue
            except Exception as e:
                log(f"[{instr}] download failed {jpg_url}:", e)
                continue

        kept.append(str(out))
        kept_iso.append(iso)
        last_kept = iso

    return kept, kept_iso

# -------- tracker (very light) --------
from scipy.ndimage import gaussian_filter
from filterpy.kalman import KalmanFilter

def simple_track_detect(frame_paths, instr, ts_list):
    if len(frame_paths) < 4: return []
    pairs = sorted(zip(frame_paths, ts_list), key=lambda x: x[1])
    cands, active, next_id = [], [], 0  # active: list of (kf, pos, age, tid)

    for idx, (p, ts) in enumerate(pairs):
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None: continue

        sm = gaussian_filter(im.astype(float)/255.0, sigma=1.0)*255
        sm = np.clip(sm, 0, 255).astype(np.uint8)

        _, thr = cv2.threshold(sm, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # predict
        new_active = []
        for kf, pos, age, tid in active:
            kf.predict()
            pred = kf.x[:2, 0].astype(int)
            if age > 3: continue
            new_active.append((kf, tuple(pred), age+1, tid))
        active = new_active

        # measurements
        meas = []
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] > 10:
                mx = int(M["m10"]/M["m00"]); my = int(M["m01"]/M["m00"])
                meas.append((mx, my))

        # associate nearest <= 80 px
        used = set()
        for i,(kf,pred,age,tid) in enumerate(active):
            best, bi = None, -1
            bestd = 1e9
            for j,(mx,my) in enumerate(meas):
                if j in used: continue
                d = (mx-pred[0])**2+(my-pred[1])**2
                if d < bestd: bestd, best, bi = d, (mx,my), j
            if best and bestd < 80**2:
                kf.update(np.array(best).reshape(2,1))
                active[i] = (kf, best, 0, tid)
                used.add(bi)

        # spawn
        for (mx,my) in meas:
            if any(px==mx and py==my for _,(px,py),_,_ in active): continue
            kf = KalmanFilter(dim_x=4, dim_z=2)
            kf.x[:2] = np.array([mx,my]).reshape(2,1)
            kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
            kf.H = np.array([[1,0,0,0],[0,1,0,0]])
            kf.P *= 100; kf.R *= 8; kf.Q = np.eye(4)*0.15
            active.append((kf,(mx,my),0,next_id)); next_id += 1

        # mature tracks → crops
        for kf,pos,age,tid in active:
            if age==0 and len([t for _,t in pairs[:idx+1] if t<=ts])>=4:
                h,w = sm.shape; x,y = pos
                x0,y0 = max(0,x-32), max(0,y-32)
                x1,y1 = min(w,x+32), min(h,y+32)
                crop = sm[y0:y1, x0:x1]
                if crop.size==0: continue
                tmp = Path(tempfile.mktemp(suffix=".png"))
                cv2.imwrite(str(tmp), crop)
                cls = classify_crop_batch([str(tmp)])[0]
                try: os.unlink(tmp)
                except: pass
                if cls.get("score",0.0) >= 0.5:
                    crops_dir = OUT_DIR / "crops"; crops_dir.mkdir(parents=True, exist_ok=True)
                    cname = f"{instr}_track{tid}_{ts.replace(':','')}.png"
                    cpath = crops_dir / cname
                    cv2.imwrite(str(cpath), crop)
                    cands.append({
                        "instrument": f"LASCO {instr}",
                        "timestamp": ts,
                        "track_id": tid,
                        "bbox": [int(x-32), int(y-32), 64, 64],
                        "crop_path": f"crops/{cname}",
                        "ai_label": cls.get("label","unknown"),
                        "ai_score": float(round(cls.get("score",0.0),3)),
                        "pos": {"x": int(x), "y": int(y)},
                        # placeholders for UI toggles (can wire later)
                        "original_mid_path": None,
                        "annotated_mid_path": None
                    })
    return cands

# -------- optional Groq vision (rate-limited) --------
def groq_refine(candidates):
    if not GROQ_API_KEY or not candidates:
        return candidates

    # rate-limit window
    try:
        if LAST_CALL_FILE.exists():
            last = datetime.fromtimestamp(float(LAST_CALL_FILE.read_text().strip()))
            if datetime.utcnow() - last < timedelta(hours=GROQ_LIMIT_HOURS):
                log("Groq rate-limit active — skipping")
                return candidates
    except Exception:
        pass

    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        log("Groq SDK not available:", e)
        return candidates

    # top-scoring crops
    tops = sorted([c for c in candidates if c["ai_score"]>0.5],
                  key=lambda x: x["ai_score"], reverse=True)[:GROQ_MAX_CALLS]
    if not tops:
        log("No high-score crops for Groq")
        return candidates

    for c in tops:
        p = OUT_DIR / c["crop_path"]
        if not p.exists(): continue
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        try:
            resp = client.chat.completions.create(
                messages=[{
                    "role":"user",
                    "content":[
                        {"type":"text","text":"Is this SOHO LASCO crop a comet? Reply as:\nlabel: comet/not_comet\nscore: 0-1\nreason: short."},
                        {"type":"image_url","image_url":{"url": f"data:image/png;base64,{b64}"}}
                    ]
                }],
                model="llama-3.2-90b-vision-preview"
            )
            txt = resp.choices[0].message.content.lower()
            label = "comet" if "comet" in txt else "not_comet"
            # try to parse a numeric score
            score = 0.0
            if "score:" in txt:
                try: score = float(txt.split("score:")[1].split()[0])
                except: pass
            c["groq_label"] = label
            c["groq_score"] = float(score)
            c["groq_reason"] = txt.strip()[:300]
        except Exception as e:
            log("Groq error:", e)

    try:
        LAST_CALL_FILE.write_text(str(datetime.utcnow().timestamp()))
    except Exception:
        pass
    return candidates

# -------- status writers --------
def write_lastthumb(img_path: Path, out_png: Path):
    try:
        im = cv2.imread(str(img_path))
        if im is None: return
        h,w = im.shape[:2]
        sc = 320.0 / max(h,w)
        imr = cv2.resize(im, (int(w*sc), int(h*sc)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_png), imr)
    except Exception: pass

def write_status_payload(c2_frames, c3_frames, c2_last, c3_last, c2_times, c3_times, candidates):
    ts_iso = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z")
    payload = {
        "name": "latest_status.json",
        "generated_at": ts_iso,
        "timestamp_utc": ts_iso,
        "hours_back": HOURS,
        "step_min": STEP_MIN,
        "c2_frames": len(c2_frames),
        "c3_frames": len(c3_frames),
        "c2_last_frame": Path(c2_last).name if c2_last else "",
        "c3_last_frame": Path(c3_last).name if c3_last else "",
        "time_range": {
            "C2": f"{(min(c2_times) if c2_times else 'none')} → {(max(c2_times) if c2_times else 'none')}",
            "C3": f"{(min(c3_times) if c3_times else 'none')} → {(max(c3_times) if c3_times else 'none')}",
        },
        "errors": [],
        "auto_selected_count": len(candidates),
        "candidates": candidates,  # <-- embed for the frontend
    }
    (OUT_DIR / "latest_status.json").write_text(json.dumps(payload, indent=2))

def write_run_log(c2n, c3
