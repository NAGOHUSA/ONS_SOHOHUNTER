# detector/fetch_lasco.py
import os
import re
import io
import pathlib
import requests
from typing import List
from datetime import datetime, timedelta, timezone
from PIL import Image, ImageSequence  # Pillow already in requirements

BASE = "https://soho.nascom.nasa.gov"
LATEST = f"{BASE}/data/LATEST"
INDEX = {
    "C2": f"{LATEST}/latest-lascoC2.html",
    "C3": f"{LATEST}/latest-lascoC3.html",
}

HEADERS = {
    "User-Agent": "ONS_SOHOHUNTER/1.4 (+https://github.com/NAGOHUSA/ONS_SOHOHUNTER)"
}

# Match href/src image links
HREF_IMG = re.compile(r'''href\s*=\s*["']([^"']+\.(?:png|jpe?g|gif))["']''', re.IGNORECASE)
SRC_IMG  = re.compile(r'''src\s*=\s*["']([^"']+\.(?:png|jpe?g|gif))["']''', re.IGNORECASE)
DIR_JPG  = re.compile(r'''href\s*=\s*["']([^"']+\.jpe?g)["']''', re.IGNORECASE)

def _abs_url(url: str) -> str:
    if url.startswith(("http://","https://")): return url
    if url.startswith("/"): return f"{BASE}{url}"
    return f"{LATEST}/{url}"

def _belongs_to_detector(u: str, det: str) -> bool:
    s = u.lower(); d = det.lower()
    return (f"/{d}/" in s) or (f"_{d}_" in s) or (d in s)

def _download(url: str) -> bytes:
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.content

def _save_bytes(data: bytes, dst: pathlib.Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        f.write(data)

def _normalize_name(det: str, url_or_name: str) -> str:
    name = os.path.basename(url_or_name.split("?")[0])
    if not name.upper().startswith(det + "_"):
        name = f"{det}_{name}"
    return name

def _parse_index_for_images(det: str, html: str) -> List[str]:
    hrefs = [m.group(1) for m in HREF_IMG.finditer(html)]
    srcs  = [m.group(1) for m in SRC_IMG.finditer(html)]
    links = hrefs + srcs
    links = [_abs_url(u) for u in links if _belongs_to_detector(u, det)]
    seen = set(); uniq = []
    for u in links:
        if u not in seen:
            seen.add(u); uniq.append(u)
    # Prefer 1024, then others
    uniq.sort(key=lambda u: (("_1024" in u.lower())*2) + (u.lower().endswith(".jpg")), reverse=True)
    return uniq

def _dir_url(det: str, day: datetime) -> str:
    yyyy = day.strftime("%Y")
    ymd  = day.strftime("%Y%m%d")
    cam = det.lower()
    return f"{BASE}/data/REPROCESSING/Completed/{yyyy}/{cam}/{ymd}/"

def _parse_dir_for_jpgs(html: str) -> List[str]:
    out, seen = [], set()
    for m in DIR_JPG.finditer(html):
        u = m.group(1)
        if u not in seen:
            seen.add(u); out.append(u)
    # prefer *_1024*.jpg first
    out.sort(key=lambda u: ("_1024" in u.lower(), u.lower().endswith(".jpg")), reverse=True)
    return out

def _fallback_dir(det: str, max_count: int) -> List[str]:
    now = datetime.now(timezone.utc)
    out: List[str] = []
    for day in (now, now - timedelta(days=1)):
        url = _dir_url(det, day)
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            if r.status_code != 200:
                print(f"[fetch:{det}] dir {url} -> {r.status_code}")
                continue
            files = _parse_dir_for_jpgs(r.text)
            abs_urls = [url + f for f in files]
            out.extend(abs_urls)
            if len(out) >= max_count:
                break
        except Exception as e:
            print(f"[fetch:{det}] dir fallback error {url}: {e}")
    return out[:max_count]

def _gif_fallback(det: str, max_count: int, root: pathlib.Path) -> List[str]:
    """Download current_c2.gif/current_c3.gif and split into PNGs."""
    gif_url = f"{LATEST}/current_{det.lower()}.gif"
    det_root = root / det
    det_root.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []
    try:
        data = _download(gif_url)
        im = Image.open(io.BytesIO(data))
        # use UTC timestamp base for names
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        count = 0
        for i, frame in enumerate(ImageSequence.Iterator(im)):
            if count >= max_count: break
            # ensure RGB then save as PNG
            fr = frame.convert("L") if frame.mode not in ("L","LA") else frame
            # synthesize lasco-like name: YYYYMMDD_HHMMSS_c2_1024.jpg -> but we save PNG
            name = f"{det}_{ts}_{det.lower()}_{i:03d}.png"
            dst = det_root / name
            fr.save(dst, format="PNG")
            saved.append(str(dst))
            count += 1
        if saved:
            print(f"[fetch:{det}] GIF fallback extracted {len(saved)} frame(s) from {gif_url}")
    except Exception as e:
        print(f"[fetch:{det}] GIF fallback failed ({gif_url}): {e}")
    return saved

def fetch_latest_from_page(det: str, max_count: int, root: pathlib.Path) -> List[str]:
    saved: List[str] = []
    # 1) Try LATEST index page
    try:
        url = INDEX[det]
        r = requests.get(url, headers=HEADERS, timeout=60)
        print(f"[fetch:{det}] LATEST {url} -> {r.status_code}")
        if r.status_code == 200:
            links = _parse_index_for_images(det, r.text)[:max_count]
        else:
            links = []
    except Exception as e:
        print(f"[fetch:{det}] LATEST fetch failed: {e}")
        links = []

    det_root = root / det
    det_root.mkdir(parents=True, exist_ok=True)

    for img_url in links:
        try:
            fname = _normalize_name(det, img_url)
            dst = det_root / fname
            if dst.exists() and dst.stat().st_size > 0:
                continue
            _save_bytes(_download(img_url), dst)
            saved.append(str(dst))
            print(f"[fetch:{det}] saved {fname} (LATEST)")
        except Exception as e:
            print(f"[fetch:{det}] failed {img_url} -> {e}")

    # 2) Directory listing fallback if nothing saved
    if not saved:
        print(f"[fetch:{det}] LATEST yielded 0 files; trying directory fallback…")
        try:
            links = _fallback_dir(det, max_count)
            for img_url in links:
                try:
                    fname = _normalize_name(det, img_url)
                    dst = det_root / fname
                    if dst.exists() and dst.stat().st_size > 0:
                        continue
                    _save_bytes(_download(img_url), dst)
                    saved.append(str(dst))
                    print(f"[fetch:{det}] saved {fname} (DIR)")
                except Exception as e:
                    print(f"[fetch:{det}] failed {img_url} -> {e}")
        except Exception as e:
            print(f"[fetch:{det}] dir fallback outer error: {e}")

    # 3) Live GIF fallback if still nothing
    if not saved:
        print(f"[fetch:{det}] Directory also empty; extracting frames from current_{det.lower()}.gif …")
        saved = _gif_fallback(det, max_count, root)

    return saved

def fetch_window(hours_back: int = 6, step_min: int = 12, root: str = "frames") -> List[str]:
    """
    Pull recent images for both C2 and C3.
    Priority: LATEST page -> REPROCESSING/Completed dir listing -> current_{c2|c3}.gif frames.
    """
    root_path = pathlib.Path(root)
    est = max(2, min(48, (hours_back * 60) // max(1, step_min)))

    saved_all: List[str] = []
    for det in ("C2", "C3"):
        got = fetch_latest_from_page(det, est, root_path)
        saved_all.extend(got)

    print(f"[fetch] summary: saved {len(saved_all)} file(s) into {root}/")
    return saved_all
