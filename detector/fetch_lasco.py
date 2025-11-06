# detector/fetch_lasco.py
import os
import re
import pathlib
import requests
from typing import List, Optional
from datetime import datetime, timedelta, timezone

BASE = "https://soho.nascom.nasa.gov"
LATEST = f"{BASE}/data/LATEST"
INDEX = {
    "C2": f"{LATEST}/latest-lascoC2.html",
    "C3": f"{LATEST}/latest-lascoC3.html",
}

HEADERS = {
    "User-Agent": "ONS_SOHOHUNTER/1.3 (+https://github.com/NAGOHUSA/ONS_SOHOHUNTER)"
}

# Match href/src image links
HREF_IMG = re.compile(r'''href\s*=\s*["']([^"']+\.(?:png|jpe?g|gif))["']''', re.IGNORECASE)
SRC_IMG  = re.compile(r'''src\s*=\s*["']([^"']+\.(?:png|jpe?g|gif))["']''', re.IGNORECASE)

# Fallback directory listing: anchor links to *_1024.jpg or similar
DIR_JPG = re.compile(r'''href\s*=\s*["']([^"']+\.jpe?g)["']''', re.IGNORECASE)

def _score_url(u: str) -> int:
    s = u.lower()
    score = 0
    if "_1024" in s: score += 10
    if "_512" in s:  score -= 2
    if "/reprocessing/completed/" in s: score += 3
    if "/lasco" in s: score += 1
    return score

def _abs_url(url: str) -> str:
    if url.startswith(("http://","https://")):
        return url
    if url.startswith("/"):
        return f"{BASE}{url}"
    return f"{LATEST}/{url}"

def _belongs_to_detector(u: str, det: str) -> bool:
    s = u.lower(); d = det.lower()
    # accept if path includes c2/c3 or filename contains it, otherwise last resort: 'c2' substring
    return (f"/{d}/" in s) or (f"_{d}_" in s) or (f"{d}" in s)

def _download(url: str) -> bytes:
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.content

def _save_bytes(data: bytes, dst: pathlib.Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        f.write(data)

def _normalize_name(det: str, url: str) -> str:
    name = os.path.basename(url.split("?")[0])
    if not name.upper().startswith(det + "_"):
        name = f"{det}_{name}"
    return name

def _parse_index_for_images(det: str, html: str) -> List[str]:
    hrefs = [m.group(1) for m in HREF_IMG.finditer(html)]
    srcs  = [m.group(1) for m in SRC_IMG.finditer(html)]
    all_links = hrefs + srcs
    norm = [_abs_url(u) for u in all_links]
    norm = [u for u in norm if _belongs_to_detector(u, det)]
    seen = set(); uniq = []
    for u in norm:
        if u not in seen:
            seen.add(u); uniq.append(u)
    uniq.sort(key=_score_url, reverse=True)
    return uniq

# ---------- fallback to directory tree for today & yesterday ----------
def _dir_url(det: str, day: datetime) -> str:
    cam = det.lower()  # "c2" or "c3"
    yyyy = day.strftime("%Y")
    ymd  = day.strftime("%Y%m%d")
    return f"{BASE}/data/REPROCESSING/Completed/{yyyy}/{cam}/{ymd}/"

def _parse_dir_for_jpgs(html: str) -> List[str]:
    links = []
    for m in DIR_JPG.finditer(html):
        links.append(m.group(1))
    # Dedup, keep order
    seen = set(); out = []
    for u in links:
        if u not in seen:
            seen.add(u); out.append(u)
    return out

def _fallback_fetch(det: str, max_count: int) -> List[str]:
    # Try today first, then yesterday
    now = datetime.now(timezone.utc)
    days = [now, now - timedelta(days=1)]
    out = []
    for day in days:
        try:
            url = _dir_url(det, day)
            r = requests.get(url, headers=HEADERS, timeout=60)
            if r.status_code != 200:  # try next day
                continue
            files = _parse_dir_for_jpgs(r.text)
            # prefer 1024, then others; also newest later in listing typically, so reverse
            files.sort(key=lambda u: (("_1024" in u.lower())*2) + (u.lower().endswith(".jpg")), reverse=True)
            # make absolute urls
            abs_urls = [url + f for f in files if f.lower().endswith((".jpg",".jpeg"))]
            out.extend(abs_urls)
            if len(out) >= max_count:
                break
        except Exception:
            continue
    # keep at most max_count
    return out[:max_count]

# ---------- public API ----------
def fetch_latest_from_page(det: str, max_count: int, root: pathlib.Path) -> List[str]:
    url = INDEX[det]
    saved: List[str] = []
    try:
        html = requests.get(url, headers=HEADERS, timeout=60).text
        links = _parse_index_for_images(det, html)
    except Exception as e:
        print(f"[fetch:{det}] index fetch failed: {e}")
        links = []

    # If LATEST page yielded nothing, fall back to REPROCESSING/Completed tree
    if not links:
        print(f"[fetch:{det}] LATEST page had no usable image links; trying directory fallback…")
        links = _fallback_fetch(det, max_count)

    if max_count > 0:
        links = links[:max_count]

    det_root = root / det
    det_root.mkdir(parents=True, exist_ok=True)

    for img_url in links:
        try:
            fname = _normalize_name(det, img_url)
            dst = det_root / fname
            if dst.exists() and dst.stat().st_size > 0:
                continue
            data = _download(img_url)
            _save_bytes(data, dst)
            saved.append(str(dst))
            print(f"[fetch:{det}] saved {fname}")
        except Exception as e:
            print(f"[fetch:{det}] failed {img_url} -> {e}")

    return saved

def fetch_window(hours_back: int = 6, step_min: int = 12, root: str = "frames") -> List[str]:
    """
    Pull recent images for both C2 and C3. Prefer the NASA 'LATEST' index pages.
    If those stop exposing direct image links, fall back to scraping the
    REPROCESSING/Completed directory for today & yesterday.
    """
    root_path = pathlib.Path(root)
    est = max(2, min(48, (hours_back * 60) // max(1, step_min)))

    saved_all: List[str] = []
    for det in ("C2", "C3"):
        saved = fetch_latest_from_page(det, est, root_path)
        saved_all.extend(saved)

    print(f"[fetch] summary: saved {len(saved_all)} file(s)")
    return saved_all
