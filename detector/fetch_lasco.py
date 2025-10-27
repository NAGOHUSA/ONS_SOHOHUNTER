import os
import re
import pathlib
import requests
from typing import List, Optional
from datetime import datetime

BASE = "https://soho.nascom.nasa.gov"
LATEST = f"{BASE}/data/LATEST"
INDEX = {
    "C2": f"{LATEST}/latest-lascoC2.html",
    "C3": f"{LATEST}/latest-lascoC3.html",
}

HEADERS = {
    "User-Agent": "ONS_SOHOHUNTER/1.2 (+https://github.com/NAGOHUSA/ONS_SOHOHUNTER)"
}

# Match href/src image links (we'll do two passes: href first, then img src)
HREF_IMG = re.compile(r'''href\s*=\s*["']([^"']+\.(?:png|jpe?g|gif))["']''', re.IGNORECASE)
SRC_IMG  = re.compile(r'''src\s*=\s*["']([^"']+\.(?:png|jpe?g|gif))["']''', re.IGNORECASE)

# Prefer full-res "_1024" over thumbnails "_512"
def _score_url(u: str) -> int:
    s = u.lower()
    score = 0
    if "_1024" in s: score += 10
    if "_512" in s:  score -= 2
    if "/reprocessing/completed/" in s: score += 3
    if "/lasco" in s: score += 1
    return score

def _abs_url(url: str) -> str:
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if url.startswith("/"):
        return f"{BASE}{url}"
    # most links on the page are relative to /data/LATEST/
    return f"{LATEST}/{url}"

def _belongs_to_detector(u: str, det: str) -> bool:
    s = u.lower()
    det_l = det.lower()
    # Accept if path includes /c2/ or /c3/, or filename includes _c2_/_c3_.
    if f"/{det_l}/" in s or f"_{det_l}_" in s:
        return True
    # Also accept LASCO pages that don’t explicit include c2/c3 in path but filename shows it.
    return det_l in s

def _download(url: str) -> bytes:
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.content

def _save_bytes(data: bytes, dst: pathlib.Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        f.write(data)

def _normalize_name(det: str, url: str) -> str:
    """Keep original filename; prefix with det if needed."""
    name = os.path.basename(url.split("?")[0])
    if not name.upper().startswith(det + "_"):
        name = f"{det}_{name}"
    return name

def _parse_index_for_images(det: str, html: str) -> List[str]:
    # First prefer anchor hrefs (tend to be 1024) then img src (often 512)
    hrefs = [m.group(1) for m in HREF_IMG.finditer(html)]
    srcs  = [m.group(1) for m in SRC_IMG.finditer(html)]
    all_links = hrefs + srcs

    # Normalize absolute URLs and filter by detector
    norm = [_abs_url(u) for u in all_links]
    norm = [u for u in norm if _belongs_to_detector(u, det)]

    # Dedup while preserving first occurrence
    seen = set()
    uniq = []
    for u in norm:
        if u not in seen:
            seen.add(u)
            uniq.append(u)

    # Sort by our quality score so _1024 show first
    uniq.sort(key=_score_url, reverse=True)
    return uniq

def fetch_latest_from_page(det: str, max_count: int, root: pathlib.Path) -> List[str]:
    url = INDEX[det]
    try:
        html = requests.get(url, headers=HEADERS, timeout=60).text
    except Exception as e:
        print(f"[fetch:{det}] index fetch failed: {e}")
        return []

    links = _parse_index_for_images(det, html)
    if max_count > 0:
        links = links[:max_count]

    saved: List[str] = []
    det_root = root / det
    det_root.mkdir(parents=True, exist_ok=True)

    for img_url in links:
        try:
            fname = _normalize_name(det, img_url)
            dst = det_root / fname
            # Skip if already present
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
    Pull recent images for both C2 and C3 directly from the NASA 'LATEST' index pages.
    We approximate desired count from hours_back/step_min and fetch that many best links,
    preferring full-res (_1024) where available.
    """
    root_path = pathlib.Path(root)
    est = max(2, min(48, (hours_back * 60) // max(1, step_min)))

    saved_all: List[str] = []
    for det in ("C2", "C3"):
        saved = fetch_latest_from_page(det, est, root_path)
        saved_all.extend(saved)

    print(f"[fetch] summary: saved {len(saved_all)} file(s)")
    return saved_all
