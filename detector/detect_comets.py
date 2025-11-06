# detector/detect_comets.py
from __future__ import annotations
import os, re, math, json, argparse, pathlib, shutil
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any, Optional

import cv2, numpy as np
from fetch_lasco import fetch_window                 # robust fetcher (LATEST → dir → GIF)
from ai_classifier import classify_crop_batch        # your trainable AI filter

# ----------------------------- Tunables / env -----------------------------
OCCULTER_RADIUS_FRACTION = float(os.getenv("OCCULTER_RADIUS_FRACTION", "0.18"))
MAX_EDGE_RADIUS_FRACTION = float(os.getenv("MAX_EDGE_RADIUS_FRACTION", "0.98"))

CROP_SIZE_C2 = int(os.getenv("CROP_SIZE_C2", "96"))
CROP_SIZE_C3 = int(os.getenv("CROP_SIZE_C3", "128"))
CROP_PAD     = int(os.getenv("CROP_PAD", "10"))     # extra pixels around bbox
AI_MIN_SCORE = float(os.getenv("AI_MIN_SCORE", "0.55"))  # UI can filter; we still write all

DEBUG_OVERLAYS = os.getenv("DETECTOR_DEBUG", "0") == "1"

# ----------------------------- Helpers -----------------------------
def ensure_dir(p: pathlib.Path): p.parent.mkdir(parents=True, exist_ok=True)

def load_series(folder: pathlib.Path) -> List[Tuple[str, np.ndarray]]:
    pairs=[]
    if not folder.exists(): return pairs
    for p in sorted(folder.glob("*.*")):
        if p.suffix.lower() not in (".png",".jpg",".jpeg"): continue
        im=cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is not None: pairs.append((p.name, im))
    return pairs

def parse_frame_iso(name:str)->Optional[str]:
    m=re.search(r'(\d{8})_(\d{4,6})', name)
    if not m: return None
    d,t=m.groups(); t=t+"00" if len(t)==4 else t
    return f"{d[0:4]}-{d[4:6]}-{d[6:8]}T{t[0:2]}:{t[2:4]}:{t[4:6]}Z"

def stabilize(base, img):
    warp=np.eye(2,3,dtype=np.float32)
    try:
        _cc, warp = cv2.findTransformECC(base.astype(np.uint8), img.astype(np.uint8), warp, cv2.MOTION_EUCLIDEAN,
                                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,50,1e-4))
        return cv2.warpAffine(img, warp, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP)
    except cv2.error:
        return img

def build_static_mask(images: List[np.ndarray], ksize=5, thresh=10)->np.ndarray:
    if len(images)<3: return np.zeros_like(images[0], dtype=np.uint8)
    stack=np.stack(images,axis=0); med=np.median(stack,axis=0).astype(np.uint8)
    blur=cv2.GaussianBlur(med,(ksize,ksize),0)
    _,mask=cv2.threshold(blur, np.median(blur)+thresh, 255, cv2.THRESH_BINARY)
    return cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3,3),np.uint8), iterations=1)

def find_moving_points(series, static_mask=None):
    pts=[]
    for i in range(1,len(series)):
        _,a=series[i-1]; _,b=series[i]
        diff=cv2.absdiff(b,a)
        if static_mask is not None: diff=cv2.bitwise_and(diff, cv2.bitwise_not(static_mask))
        blur=cv2.GaussianBlur(diff,(5,5),0)
        thr=cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,31,-5)
        clean=cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations=1)
        cnts,_=cv2.findContours(clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area=cv2.contourArea(c)
            if 3<=area<=200:
                (x,y),_ = cv2.minEnclosingCircle(c)
                pts.append((i,float(x),float(y),float(area)))
    return pts

def link_tracks(points, min_len=3, max_jump=25):
    tracks=[]; by_t={}
    for t,x,y,a in points: by_t.setdefault(t,[]).append((x,y,a))
    for t in sorted(by_t.keys()):
        for x,y,a in by_t[t]:
            best=None; bi=-1
            for i,tr in enumerate(tracks):
                if tr[-1][0]==t-1:
                    dx=x-tr[-1][1]; dy=y-tr[-1][2]
                    if dx*dx+dy*dy<=max_jump*max_jump:
                        d=float(np.hypot(dx,dy))
                        if best is None or d<best: best,bi=d,i
            if bi>=0: tracks[bi].append((t,x,y,a))
            else: tracks.append([(t,x,y,a)])
    return [tr for tr in tracks if len(tr)>=min_len]

def radial_guard_ok(x,y,w,h,rmin,rmax):
    cx,cy=w/2.0,h/2.0; r=math.hypot(x-cx,y-cy); rmax=min(cx,cy); frac=r/max(1e-6,rmax)
    return (rmin<=frac<=rmax)

def crop_for_track(det:str, aligned_names:List[str], aligned_imgs:List[np.ndarray], tr:List[Tuple[int,float,float,float]])->Dict[str,str]:
    # mid index & bbox with padding
    mid_idx = len(aligned_imgs)//2
    _, mid_img = aligned_names[mid_idx], aligned_imgs[mid_idx]
    xs=[x for (t,x,y,a) in tr]; ys=[y for (t,x,y,a) in tr]
    x0=int(max(0, min(xs) - CROP_PAD)); y0=int(max(0, min(ys) - CROP_PAD))
    x1=int(min(mid_img.shape[1]-1, max(xs) + CROP_PAD)); y1=int(min(mid_img.shape[0]-1, max(ys) + CROP_PAD))
    crop = mid_img[y0:y1+1, x0:x1+1].copy()
    # normalize size by detector
    target = CROP_SIZE_C2 if det=="C2" else CROP_SIZE_C3
    if max(crop.shape[:2])>0 and max(crop.shape[:2])!=target:
        s = float(target)/float(max(crop.shape[:2]))
        crop = cv2.resize(crop, (int(crop.shape[1]*s), int(crop.shape[0]*s)), interpolation=cv2.INTER_AREA)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base=f"{det}_{ts}_trk"
    crops_dir = pathlib.Path("detections")/"crops"
    anno_dir  = pathlib.Path("detections")/"annotated"
    crops_dir.mkdir(parents=True, exist_ok=True)
    anno_dir.mkdir(parents=True, exist_ok=True)

    crop_name = f"{base}.png"
    crop_path = crops_dir/crop_name
    cv2.imwrite(str(crop_path), crop)

    # annotated (draw the trail on the mid image)
    ann = cv2.cvtColor(mid_img, cv2.COLOR_GRAY2BGR)
    for (t,x,y,a) in tr:
        cv2.circle(ann, (int(x),int(y)), 2, (0,255,255), -1)
    cv2.rectangle(ann, (x0,y0), (x1,y1), (0,200,0), 1)
    anno_name = f"{base}_annotated.png"
    anno_path = anno_dir/anno_name
    cv2.imwrite(str(anno_path), ann)

    return {
        "crop_rel": f"crops/{crop_name}",
        "anno_rel": f"annotated/{anno_name}",
        "mid_rel":  aligned_names[mid_idx]
    }

# ----------------------------- Main -----------------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--hours",type=int,default=6)
    ap.add_argument("--step-min",type=int,default=12)
    ap.add_argument("--out",type=str,default="detections")
    args=ap.parse_args()

    out_dir=pathlib.Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print("=== DETECTION START ===")
    fetched = fetch_window(hours_back=args.hours, step_min=args.step_min, root="frames")
    print(f"[fetch] got {len(fetched)} new file(s)")

    # Load series
    series_c2=load_series(pathlib.Path("frames")/"C2")
    series_c3=load_series(pathlib.Path("frames")/"C3")
    print(f"[DEBUG] C2 frames: {len(series_c2)}, C3 frames: {len(series_c3)}")

    detectors_stats={}
    all_hits: List[Dict[str,Any]]=[]

    def process(det:str, series):
        if not series:
            detectors_stats[det]={"frames":0,"tracks":0,"last_frame_name":"","last_frame_iso":"","last_frame_size":[0,0]}
            return []

        names=[n for n,_ in series]; images=[im for _,im in series]
        base=images[0]
        aligned=[(names[0], base)]
        for n,im in zip(names[1:], images[1:]):
            aligned.append((n, stabilize(base, im)))
        names=[n for n,_ in aligned]; images=[im for _,im in aligned]
        h, w = images[0].shape[:2]

        static_mask=build_static_mask(images[-min(8,len(images)):])
        pts=find_moving_points(aligned, static_mask=static_mask)
        guarded=[(t,x,y,a) for (t,x,y,a) in pts if radial_guard_ok(x,y,w,h,OCCULTER_RADIUS_FRACTION,MAX_EDGE_RADIUS_FRACTION)]
        tracks=link_tracks(guarded, min_len=3)

        detectors_stats[det]={"frames":len(series),"tracks":len(tracks),
                              "last_frame_name":names[-1], "last_frame_iso":parse_frame_iso(names[-1]) or "",
                              "last_frame_size":[int(w),int(h)]}

        # Build candidates + crops
        cands=[]
        crop_paths=[]
        extras=[]
        for i,tr in enumerate(tracks, start=1):
            pos=[]
            for (t,x,y,a) in tr:
                iso=parse_frame_iso(names[t]) or ""
                pos.append({"frame": names[t], "time_utc": iso, "x": float(x), "y": float(y)})

            paths = crop_for_track(det, names, images, tr)
            cands.append({
                "detector": det,
                "series_mid_frame": names[len(names)//2],
                "track_index": i,
                "positions": pos,
                "image_size": [int(w),int(h)],
                "origin": "upper_left",
                "crop_path": paths["crop_rel"],
                "annotated_path": paths["anno_rel"],
                "original_mid_path": paths["mid_rel"],
            })
            crop_paths.append(str(pathlib.Path("detections")/paths["crop_rel"]))
            extras.append(paths)

        # AI classify in batch
        if crop_paths:
            ai = classify_crop_batch(crop_paths)
            for cand, aires in zip(cands, ai):
                cand["ai_label"] = aires.get("label","unknown")
                cand["ai_score"] = float(aires.get("score",0.0))
        return cands

    all_hits.extend(process("C2", series_c2))
    all_hits.extend(process("C3", series_c3))

    # Summary + write files
    ts_iso=datetime.utcnow().isoformat(timespec="seconds")+"Z"
    summary={
        "timestamp_utc": ts_iso,
        "hours_back": args.hours,
        "step_min": args.step_min,
        "detectors": detectors_stats,
        "fetched_new_frames": len(fetched),
        "errors": [],
        "auto_selected_count": sum(1 for c in all_hits if c.get("ai_label")=="comet" and c.get("ai_score",0)>=AI_MIN_SCORE),
        "candidates_in_report": len(all_hits),
        "name": "latest_status.json",
        "generated_at": ts_iso
    }

    # Write status + candidates
    (out_dir/"latest_status.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if all_hits:
        ts_name=datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        with open(out_dir/f"candidates_{ts_name}.json","w",encoding="utf-8") as f: json.dump(all_hits, f, indent=2)
        # Always refresh candidates_latest.json for the UI
        with open(out_dir/"candidates_latest.json","w",encoding="utf-8") as f: json.dump(all_hits, f, indent=2)
    else:
        # ensure an empty file exists so UI can load "[]"
        (out_dir/"candidates_latest.json").write_text("[]\n", encoding="utf-8")

    print(f"=== DONE ===  Cands:{len(all_hits)}  AI-Comets≥{AI_MIN_SCORE}:{sum(1 for c in all_hits if c.get('ai_label')=='comet' and c.get('ai_score',0)>=AI_MIN_SCORE)}")
if __name__=="__main__":
    main()
