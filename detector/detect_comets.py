# detector/detect_comets.py
from __future__ import annotations
import os, re, math, csv, json, argparse, pathlib, shutil, io
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any, Optional

import cv2, numpy as np
from PIL import Image, ImageSequence  # extra fallback only
from fetch_lasco import fetch_window   # our robust fetcher

OCCULTER_RADIUS_FRACTION = float(os.getenv("OCCULTER_RADIUS_FRACTION", "0.18"))
MAX_EDGE_RADIUS_FRACTION = float(os.getenv("MAX_EDGE_RADIUS_FRACTION", "0.98"))
DEBUG_OVERLAYS          = os.getenv("DETECTOR_DEBUG", "0") == "1"

def ensure_dir(p: pathlib.Path): p.parent.mkdir(parents=True, exist_ok=True)
def save_png(path: pathlib.Path, img: np.ndarray): ensure_dir(path); cv2.imwrite(str(path), img)

def load_series(folder: pathlib.Path) -> List[Tuple[str, np.ndarray]]:
    pairs=[]; 
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

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--hours",type=int,default=6)
    ap.add_argument("--step-min",type=int,default=12)
    ap.add_argument("--out",type=str,default="detections")
    args=ap.parse_args()

    out_dir=pathlib.Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print("=== DETECTION START ===")
    # Always try to fetch fresh frames
    fetched = fetch_window(hours_back=args.hours, step_min=args.step_min, root="frames")
    print(f"[fetch] got {len(fetched)} new file(s)")

    # If still nothing, final hard fallback: split the live GIFs directly here too
    # (protects against import mismatches)
    total_after = (len(load_series(pathlib.Path("frames")/"C2")) +
                   len(load_series(pathlib.Path("frames")/"C3")))
    if total_after == 0:
        try:
            from fetch_lasco import _fallback_gif  # type: ignore
            got = _fallback_gif("C2", max(6, args.hours*2), pathlib.Path("frames")) + \
                  _fallback_gif("C3", max(6, args.hours*2), pathlib.Path("frames"))
            print(f"[gif-fallback] extracted {len(got)} frames from live GIFs")
        except Exception as e:
            print(f"[gif-fallback] failed: {e}")

    # Load series now
    series_c2=load_series(pathlib.Path("frames")/"C2")
    series_c3=load_series(pathlib.Path("frames")/"C3")
    print(f"[DEBUG] C2 frames: {len(series_c2)}, C3 frames: {len(series_c3)}")

    detectors_stats={}
    results={"C2":[],"C3":[]}

    def process(det:str, series):
        hits=[]
        if not series:
            detectors_stats[det]={"frames":0,"tracks":0,"last_frame_name":"","last_frame_iso":"","last_frame_size":[0,0]}
            return hits
        names=[n for n,_ in series]; images=[im for _,im in series]
        base=images[0]
        aligned=[(names[0], base)]
        for n,im in zip(names[1:], images[1:]): aligned.append((n, stabilize(base, im)))
        names=[n for n,_ in aligned]; images=[im for _,im in aligned]
        w,h=images[0].shape[1], images[0].shape[0]
        static_mask=build_static_mask(images[-min(8,len(images)):])
        pts=find_moving_points(aligned, static_mask=static_mask)
        guarded=[(t,x,y,a) for (t,x,y,a) in pts if radial_guard_ok(x,y,w,h,OCCULTER_RADIUS_FRACTION,MAX_EDGE_RADIUS_FRACTION)]
        tracks=link_tracks(guarded, min_len=3)
        detectors_stats[det]={"frames":len(series),"tracks":len(tracks),
                              "last_frame_name":names[-1], "last_frame_iso":parse_frame_iso(names[-1]) or "",
                              "last_frame_size":[int(w),int(h)]}
        mid_name, mid_img = aligned[len(aligned)//2]
        for i,tr in enumerate(tracks):
            # minimal output with positions for now
            positions=[]
            for (t,x,y,a) in tr:
                iso=parse_frame_iso(names[t]) or ""
                positions.append({"frame": names[t], "time_utc": iso, "x": float(x), "y": float(y)})
            hits.append({
                "detector": det,
                "series_mid_frame": mid_name,
                "track_index": i+1,
                "positions": positions,
                "image_size": [int(w),int(h)],
                "origin": "upper_left",
            })
        return hits

    results["C2"]=process("C2", series_c2)
    results["C3"]=process("C3", series_c3)

    all_hits=results["C2"]+results["C3"]
    ts_iso=datetime.utcnow().isoformat(timespec="seconds")+"Z"
    summary={
        "timestamp_utc": ts_iso,
        "hours_back": args.hours,
        "step_min": args.step_min,
        "detectors": detectors_stats,
        "fetched_new_frames": len(fetched),
        "errors": [],
        "auto_selected_count": 0,
        "candidates_in_report": len(all_hits),
        "name": "latest_status.json",
        "generated_at": ts_iso
    }

    ensure_dir(out_dir/"latest_status.json")
    with open(out_dir/"latest_status.json","w") as f: json.dump(summary, f, indent=2)
    if all_hits:
        ts_name=datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        with open(out_dir/f"candidates_{ts_name}.json","w") as f: json.dump(all_hits, f, indent=2)

    print(f"=== DONE ===  C2:{len(results['C2'])}  C3:{len(results['C3'])}  Candidates:{len(all_hits)}  Tracks:{sum(len(results[k]) for k in results)}")
if __name__=="__main__":
    main()
