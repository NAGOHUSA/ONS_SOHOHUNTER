# detector/fetch_lasco.py
import os, re, io, pathlib, requests
from typing import List
from datetime import datetime, timedelta, timezone
from PIL import Image, ImageSequence  # Pillow is in requirements

BASE = "https://soho.nascom.nasa.gov"
LATEST = f"{BASE}/data/LATEST"
INDEX = {"C2": f"{LATEST}/latest-lascoC2.html", "C3": f"{LATEST}/latest-lascoC3.html"}
HEADERS = {"User-Agent": "ONS_SOHOHUNTER/1.4 (+https://github.com/NAGOHUSA/ONS_SOHOHUNTER)"}

HREF_IMG = re.compile(r'''href\s*=\s*["']([^"']+\.(?:png|jpe?g|gif))["']''', re.IGNORECASE)
SRC_IMG  = re.compile(r'''src\s*=\s*["']([^"']+\.(?:png|jpe?g|gif))["']''', re.IGNORECASE)
DIR_JPG  = re.compile(r'''href\s*=\s*["']([^"']+\.jpe?g)["']''', re.IGNORECASE)

def _abs(u:str)->str:
    if u.startswith(("http://","https://")): return u
    if u.startswith("/"): return f"{BASE}{u}"
    return f"{LATEST}/{u}"

def _dl(u:str)->bytes:
    r=requests.get(u,headers=HEADERS,timeout=60); r.raise_for_status(); return r.content

def _save(data:bytes,dst:pathlib.Path)->None:
    dst.parent.mkdir(parents=True,exist_ok=True)
    with open(dst,"wb") as f: f.write(data)

def _norm(det:str,name_or_url:str)->str:
    name=os.path.basename(name_or_url.split("?")[0])
    return name if name.upper().startswith(det+"_") else f"{det}_{name}"

def _belongs(u:str,det:str)->bool:
    s=u.lower(); d=det.lower()
    return (f"/{d}/" in s) or (f"_{d}_" in s) or (d in s)

def _parse_latest(det:str, html:str)->List[str]:
    links=[m.group(1) for m in HREF_IMG.finditer(html)]+[m.group(1) for m in SRC_IMG.finditer(html)]
    links=[_abs(u) for u in links if _belongs(u,det)]
    seen=set(); out=[]
    for u in links:
        if u not in seen: seen.add(u); out.append(u)
    out.sort(key=lambda u:(("_1024" in u.lower())*2)+(u.lower().endswith(".jpg")), reverse=True)
    return out

def _dir_url(det:str, day:datetime)->str:
    return f"{BASE}/data/REPROCESSING/Completed/{day:%Y}/{det.lower()}/{day:%Y%m%d}/"

def _parse_dir(html:str)->List[str]:
    seen=set(); out=[]
    for m in DIR_JPG.finditer(html):
        u=m.group(1)
        if u not in seen: seen.add(u); out.append(u)
    out.sort(key=lambda u:("_1024" in u.lower(), u.lower().endswith(".jpg")), reverse=True)
    return out

def _fallback_dir(det:str, max_count:int)->List[str]:
    out=[]; now=datetime.now(timezone.utc)
    for day in (now, now-timedelta(days=1)):
        url=_dir_url(det,day)
        try:
            r=requests.get(url,headers=HEADERS,timeout=60)
            print(f"[fetch:{det}] DIR {url} -> {r.status_code}")
            if r.status_code!=200: continue
            files=_parse_dir(r.text)
            out.extend([url+f for f in files])
            if len(out)>=max_count: break
        except Exception as e:
            print(f"[fetch:{det}] DIR error: {e}")
    return out[:max_count]

def _fallback_gif(det:str, max_count:int, root:pathlib.Path)->List[str]:
    url=f"{LATEST}/current_{det.lower()}.gif"
    saved=[]; det_root=(pathlib.Path(root)/det); det_root.mkdir(parents=True,exist_ok=True)
    try:
        data=_dl(url)
        im=Image.open(io.BytesIO(data))
        ts=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        for i,frame in enumerate(ImageSequence.Iterator(im)):
            if i>=max_count: break
            fr=frame.convert("L") if frame.mode not in ("L","LA") else frame
            name=f"{det}_{ts}_{det.lower()}_{i:03d}.png"
            fr.save(det_root/name, format="PNG")
            saved.append(str(det_root/name))
        if saved: print(f"[fetch:{det}] GIF fallback extracted {len(saved)} from {url}")
    except Exception as e:
        print(f"[fetch:{det}] GIF fallback failed: {e}")
    return saved

def fetch_latest_from_page(det:str, max_count:int, root:pathlib.Path)->List[str]:
    saved=[]; det_root=(pathlib.Path(root)/det); det_root.mkdir(parents=True,exist_ok=True)
    # 1) LATEST
    try:
        url=INDEX[det]; r=requests.get(url,headers=HEADERS,timeout=60)
        print(f"[fetch:{det}] LATEST {url} -> {r.status_code}")
        links=_parse_latest(det,r.text)[:max_count] if r.status_code==200 else []
    except Exception as e:
        print(f"[fetch:{det}] LATEST error: {e}"); links=[]
    for img in links:
        try:
            fname=_norm(det,img); dst=det_root/fname
            if dst.exists() and dst.stat().st_size>0: continue
            _save(_dl(img), dst); saved.append(str(dst))
            print(f"[fetch:{det}] saved {fname} (LATEST)")
        except Exception as e:
            print(f"[fetch:{det}] failed {img} -> {e}")

    # 2) DIR
    if not saved:
        print(f"[fetch:{det}] LATEST empty; trying DIR fallback…")
        for img in _fallback_dir(det, max_count):
            try:
                fname=_norm(det,img); dst=det_root/fname
                if dst.exists() and dst.stat().st_size>0: continue
                _save(_dl(img), dst); saved.append(str(dst))
                print(f"[fetch:{det}] saved {fname} (DIR)")
            except Exception as e:
                print(f"[fetch:{det}] failed {img} -> {e}")

    # 3) GIF
    if not saved:
        print(f"[fetch:{det}] DIR empty; extracting current_{det.lower()}.gif …")
        saved=_fallback_gif(det, max_count, root)

    return saved

def fetch_window(hours_back:int=6, step_min:int=12, root:str="frames")->List[str]:
    rootp=pathlib.Path(root)
    est=max(2, min(48, (hours_back*60)//max(1,step_min)))
    all=[]
    for det in ("C2","C3"):
        got=fetch_latest_from_page(det, est, rootp)
        all.extend(got)
    print(f"[fetch] summary: saved {len(all)} file(s) into {root}/")
    return all
