# detector/fetch_lasco.py
import os
import re
import pathlib
import requests
from typing import List

BASE = "https://soho.nascom.nasa.gov"
LATEST = f"{BASE}/data/LATEST"
INDEX = {
    "C2": f"{LATEST}/latest-lascoC2.html",
    "C3": f"{LATEST}/latest-lascoC3.html",
}

HEADERS = {
    "User-Agent": "ONS_SOHOHUNTER/1.0 (+https://github.com/NAGOHUSA/ONS_SOHOHUNTER)"
}

# We’ll accept common raster formats you’re likely to want to analyze/preview.
IMG_EXTS = (".png", ".jpg", ".jpeg", ".gif")

HREF_OR_SRC_IMG = re.compile(
    r'''(?:href|src)\s*=\s*["']([^"']+\.(?:png|jpg|jpeg|gif))["']''',
    re.IGNORECASE
)


def _abs_url(url: str) -> str:
    """Normalize to an absolute URL under the /data/LATEST/ base if needed."""
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if url.startswith("/"):
        return f"{BASE}{url}"
    # relative path on the LATEST page
    return f"{LATEST}/{url}"


def _want(url: str) -> bool:
    """Filter thumbnails or irrelevant assets if present."""
    u = url.lower()
    if not u.endswith(IMG_EXTS):
        return False
    # Skip obvious thumbnails if present on the page.
    if "/thumb" in u or "_thumb" in u:
        return False
    return True


def _download(url: str, dst_path: pathlib.Path) -> bool:
    """Download URL into dst_path. Returns True if saved (new), False if already exists."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists() and dst_path.stat().st_size > 0:
        return False
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    with open(dst_path, "wb") as f:
        f.write(r.content)
    return True


def fetch_latest_from_latest_page(detector: str, count: int, root: pathlib.Path) -> List[str]:
    """
    Fetch up to `count` newest images for given detector ('C2' or 'C3') from the
    SOHO/NASA LATEST index page and save under frames/<detector>/.
    Returns list of saved file paths (strings).
    """
    detector = detector.upper()
    if detector not in INDEX:
        raise ValueError("detector must be 'C2' or 'C3'")

    idx_url = INDEX[detector]
    try:
        html = requests.get(idx_url, headers=HEADERS, timeout=60).text
    except Exception:
        return []

    # Find candidate image links on the page
    all_links = [_abs_url(m.group(1)) for m in HREF_OR_SRC_IMG.finditer(html)]
    # Filter and de-dup while preserving order (newest first: the page usually lists newest near the top)
    seen = set()
    candidates: List[str] = []
    for u in all_links:
        if not _want(u):
            continue
        # keep only links that look like they belong to LASCO and the detector page (we’re already on C2/C3 page)
        if "lasco" not in u.lower():
            continue
        if u in seen:
            continue
        seen.add(u)
        candidates.append(u)

    # Limit to requested count
    if count > 0:
        candidates = candidates[:count]

    saved: List[str] = []
    det_root = root / detector
    for u in candidates:
        fname = os.path.basename(u.split("?")[0])
        # Prefix filename with detector if not already present (helps uniqueness)
        if not fname.upper().startswith(detector + "_"):
            fname = f"{detector}_{fname}"
        dst = det_root / fname
        try:
            _download(u, dst)
            saved.append(str(dst))
        except Exception:
            # Skip silently on per-file failures; page sometimes has transient links.
            pass

    return saved


def fetch_window(hours_back: int = 6, step_min: int = 12, root: str = "frames") -> List[str]:
    """
    Compatibility shim used by the pipeline.
    We approximate how many images to fetch from the LATEST pages based on hours_back/step_min,
    then pull that many *latest* images for each of C2 and C3.

    Returns list of absolute file paths saved.
    """
    root_path = pathlib.Path(root)
    # Rough estimate: number of frames you *would* want in this window.
    # Clamp to a sane range the LATEST pages typically expose.
    est = max(2, min(48, (hours_back * 60) // max(1, step_min)))

    saved_all: List[str] = []
    for det in ("C2", "C3"):
        saved_all.extend(fetch_latest_from_latest_page(det, est, root_path))

    return saved_all
