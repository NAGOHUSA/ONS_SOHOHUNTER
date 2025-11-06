# detector/detect_comets.py
from __future__ import annotations
import os, re, math, json, argparse, pathlib, shutil
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

import cv2, numpy as np

# Optional GIF writer
try:
    import imageio
except Exception:
    imageio = None

# Local modules in repo
from fetch_lasco import fetch_window                 # pulls frames into frames/C2 and frames/C3
from ai_classifier import classify_crop_batch        # returns [{"label":..., "score":...}, ...]

# ----------------------------- Tunables / env -----------------------------
OCCULTER_RADIUS_FRACTION = float(os.getenv("OCCULTER_RADIUS_FRACTION", "0.18"))
MAX_EDGE_RADIUS_FRACTION = float(os.getenv("MAX_EDGE_RADIUS_FRACTION", "0.98"))

CROP_SIZE_C2 = int(os.getenv("CROP_SIZE_C2", "96"))
CROP_SIZE_C3 = int(os.getenv("CROP_SIZE_C3", "128"))
CROP_PAD     = int(os.getenv("CROP_PAD", "10"))     # extra pixels around bbox
AI_MIN_SCORE = float(os.getenv("AI_MIN_SCORE", "0.55"))
DEBUG_OVERLAYS = os.getenv("DETECTOR_DEBUG", "0") == "1"

GIF_FPS = int(os.getenv("CROP_GIF_FPS", "5"))
MP4_FPS = int(os.getenv("CROP_MP4_FPS", "8"))

# ----------------------------- Helpers -----------------------------
def ensure_dir(p: pathlib.Path): p.mkdir(parents=True, exist_ok=True)

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

def radial_guard_ok(x,y,w,h,rmin_frac,rmax_frac):
    cx,cy=w/2.0,h/2.0; r=math.hypot(x-cx,y-cy); rmax=min(cx,cy); frac=r/max(1e-6,rmax)
    return (rmin_frac<=frac<=rmax_frac)

def to_bgr(img_gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

def write_gif(frames_gray: List[np.ndarray], out_path: pathlib.Path, fps:int=5):
    if imageio is None:
        return False
    try:
        imgs = [imageio.core.util.Array(cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)) for f in frames_gray]
        imageio.mimsave(str(out_path), imgs, fps=fps, loop=0)
        return True
    except Exception as e:
        print(f"[warn] GIF save failed: {e}")
        return False

def write_mp4(frames_gray: List[np.ndarray], out_path: pathlib.Path, fps:int=8):
    try:
        h, w = frames_gray[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h), isColor=False)
        for f in frames_gray:
            vw.write(f)
        vw.release()
        return True
    except Exception as e:
        print(f"[warn] MP4 save failed: {e}")
        return False

def crop_for_track(
    det:str,
    aligned_names:List[str],
    aligned_imgs:List[np.ndarray],
    tr:List[Tuple[int,float,float,float]]
)->Dict[str,str]:
    """
    Build a single crop box that contains the whole track; then
    (1) save a mid-frame crop (gray)
    (2) save an annotated crop (drawn ON the crop)
    (3) save a GIF/MP4 timelapse of cropped frames
    (4) ensure the ORIGINAL mid-frame file exists under detections/originals/
    """
    # Compute bbox covering the entire track
    xs=[x for (t,x,y,a) in tr]; ys=[y for (t,x,y,a) in tr]
    # Use the aligned mid-frame index consistently
    mid_idx = len(aligned_imgs)//2
    mid_name = aligned_names[mid_idx]
    mid_img  = aligned_imgs[mid_idx]

    # Track-wide bbox with padding
    x0=int(max(0, min(xs) - CROP_PAD)); y0=int(max(0, min(ys) - CROP_PAD))
    x1=int(min(mid_img.shape[1]-1, max(xs) + CROP_PAD)); y1=int(min(mid_img.shape[0]-1, max(ys) + CROP_PAD))

    # Select output dirs
    crops_dir = pathlib.Path("detections")/"crops"
    anno_dir  = pathlib.Path("detections")/"annotated"
    orig_dir  = pathlib.Path("detections")/"originals"
    gifs_dir  = pathlib.Path("detections")/"gifs"
    mp4_dir   = pathlib.Path("detections")/"mp4"
    for d in (crops_dir, anno_dir, orig_dir, gifs_dir, mp4_dir): ensure_dir(d)

    # Create a cropped sequence across ALL aligned frames using the SAME bbox
    # (So the GIF/MP4 shows motion)
    cropped_seq = []
    for img in aligned_imgs:
        crop = img[y0:y1+1, x0:x1+1].copy()
        cropped_seq.append(crop)

    # Normalize crop size per detector for the representative stills (do NOT resize the sequence to keep aspect stable)
    rep_crop = cropped_seq[mid_idx].copy()
    target = CROP_SIZE_C2 if det=="C2" else CROP_SIZE_C3
    if max(rep_crop.shape[:2])>0 and max(rep_crop.shape[:2])!=target:
        s = float(target)/float(max(rep_crop.shape[:2]))
        rep_crop_resized = cv2.resize(rep_crop, (int(rep_crop.shape[1]*s), int(rep_crop.shape[0]*s)), interpolation=cv2.INTER_AREA)
    else:
        rep_crop_resized = rep_crop

    # Build names
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base=f"{det}_{ts}_trk"
    crop_name = f"{base}.png"
    anno_name = f"{base}_annotated.png"
    gif_name  = f"{base}.gif"
    mp4_name  = f"{base}.mp4"

    # (1) Save representative crop (gray)
    cv2.imwrite(str(crops_dir/crop_name), rep_crop_resized)

    # (2) Save annotated crop: draw the per-frame positions re-mapped into crop coordinates ON THE CROP
    ann_crop = to_bgr(rep_crop_resized)
    # we need to map original (x,y) → crop coords, and if we resized, map scale too
    s = 1.0
    if rep_crop_resized.shape[:2] != rep_crop.shape[:2]:
        s = float(rep_crop_resized.shape[1]) / float(rep_crop.shape[1])  # scale by width
    # draw small trail using positions, mapped into crop space (use mid frame alignment for readability)
    for (t,x,y,a) in tr:
        # map to crop-local
        x_local = (x - x0) * s
        y_local = (y - y0) * s
        cv2.circle(ann_crop, (int(round(x_local)), int(round(y_local))), 2, (0,255,255), -1)
    # draw bbox outline (this is the crop boundary, so subtle)
    cv2.rectangle(ann_crop, (1,1), (ann_crop.shape[1]-2, ann_crop.shape[0]-2), (0,200,0), 1)
    cv2.imwrite(str(anno_dir/anno_name), ann_crop)

    # (3) Save GIF/MP4 for the cropped sequence
    # Make a normalized 8-bit sequence for writing
    seq8 = [cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) for f in cropped_seq]
    gif_ok = False
    if imageio is not None and len(seq8) >= 2:
        gif_ok = write_gif(seq8, gifs_dir/gif_name, fps=GIF_FPS)
    mp4_ok = False
    if len(seq8) >= 2:
        mp4_ok = write_mp4(seq8, mp4_dir/mp4_name, fps=MP4_FPS)

    # (4) Copy ORIGINAL mid-frame into detections/originals/ (fixes 404)
    src = pathlib.Path("frames")/det/mid_name
    dst = orig_dir/mid_name
    try:
        if src.exists():
            if not dst.exists():
                shutil.copyfile(src, dst)
        else:
            # fallback: write the aligned mid image to keep link alive
            cv2.imwrite(str(dst), mid_img)
    except Exception as e:
        print(f"[warn] could not place original mid-frame: {e}")

    return {
        "crop_rel": f"crops/{crop_name}",
        "anno_rel": f"annotated/{anno_name}",       # annotated CROP (drawn on crop area)
        "orig_rel": f"originals/{mid_name}",        # original mid-frame (full)
        "gif_rel":  (f"gifs/{gif_name}" if gif_ok else ""),
        "mp4_rel":  (f"mp4/{mp4_name}" if mp4_ok else ""),
        "mid_name": mid_name,
        "bbox": [int(x0), int(y0), int(x1), int(y1)]
    }

# ----------------------------- Main -----------------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--hours",type=int,default=6)
    ap.add_argument("--step-min",type=int,default=12)
    ap.add_argument("--out",type=str,default="detections")
    args=ap.parse_args()

    out_dir=pathlib.Path(args.out); ensure_dir(out_dir)

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

        cands=[]
        crop_paths=[]
        for i,tr in enumerate(tracks, start=1):
            pos=[]
            for (t,x,y,a) in tr:
                iso=parse_frame_iso(names[t]) or ""
                pos.append({"frame": names[t], "time_utc": iso, "x": float(x), "y": float(y)})

            paths = crop_for_track(det, names, images, tr)
            cand = {
                "detector": det,
                "series_mid_frame": paths["mid_name"],
                "track_index": i,
                "positions": pos,
                "image_size": [int(w),int(h)],
                "origin": "upper_left",
                "crop_path": paths["crop_rel"],
                "annotated_path": paths["anno_rel"],       # now points to annotated CROP
                "original_mid_path": paths["orig_rel"],    # full original mid-frame
            }
            if paths.get("gif_rel"): cand["crop_gif_path"] = paths["gif_rel"]
            if paths.get("mp4_rel"): cand["crop_mp4_path"] = paths["mp4_rel"]

            cands.append(cand)
            crop_paths.append(str(pathlib.Path("detections")/paths["crop_rel"]))
        # AI classify in batch
        if crop_paths:
            ai = classify_crop_batch(crop_paths)
            for cand, aires in zip(cands, ai):
                cand["ai_label"] = aires.get("label","unknown")
                cand["ai_score"] = float(aires.get("score",0.0))
        return cands

    all_hits.extend(process("C2", series_c2))
    all_hits.extend(process("C3", series_c3))

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
        with open(out_dir/"candidates_latest.json","w",encoding="utf-8") as f: json.dump(all_hits, f, indent=2)
    else:
        (out_dir/"candidates_latest.json").write_text("[]\n", encoding="utf-8")

    print(f"=== DONE ===  Cands:{len(all_hits)}  AI-Comets≥{AI_MIN_SCORE}:{sum(1 for c in all_hits if c.get('ai_label')=='comet' and c.get('ai_score',0)>=AI_MIN_SCORE)}")

if __name__=="__main__":
    main()
