# detector/fetch_lasco.py
import os
import re
import pathlib
import requests
from typing import List, Tuple, Optional
from datetime import datetime
from io import BytesIO

try:
    from PIL import Image  # for GIF->PNG conversion if needed
except Exception:
    Image = None  # we'll skip conversion if Pillow isn't available

BASE = "https://soho.nascom.nasa.gov"
LATEST = f"{BASE}/data/LATEST"
INDEX = {
    "C2": f"{LATEST}/latest-lascoC2.html",
    "C3": f"{LATEST}/latest-lascoC3.html",
}
CURRENT = {
    "C2": [f"{LATEST}/current_c2.jpg", f"{LATEST}/current_c2.gif", f"{LATEST}/current_c2.png"],
    "C3": [f"{LATEST}/current_c3.jpg", f"{LATEST}/current_c3.gif", f"{LATEST}/current_c3.png"],
}

HEADERS = {
    "User-Agent": "ONS_SOHOHUNTER/1.1 (+https://github.com/NAGOHUSA/ONS_SOHOHUNTER)"
}

IMG_EXTS = (".png", ".jpg", ".jpeg", ".gif")
HREF_OR_SRC_IMG = re.compile(r'''(?:href|src)\s*=\s*["']([^"']+\.(?:png|jpg|jpeg|gif))["']''', re.IGNORECASE)


def _log(msg: str):
    print(f"[fetch] {msg}")


def _abs_url(url: str) -> str:
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if url.startswith("/"):
        return f"{BASE}{url}"
    return f"{LATEST}/{url}"


def _want(url: str) -> bool:
    u = url.lower()
    if not u.endswith(IMG_EXTS):
        return False
    if "/thumb" in u or "_thumb" in u:
        return False
    # prefer lasco C2/C3 paths/images
    if "lasco" not in u:
        return False
    return True


def _timestamp_from_headers(headers) -> Optional[str]:
    # e.g., "Mon, 27 Oct 2025 15:04:00 GMT"
    lm = headers.get("Last-Modified")
    if not lm:
        return None
    try:
        dt = datetime.strptime(lm, "%a, %d %b %Y %H:%M:%S %Z")
        return dt.strftime("%Y%m%d_%H%M%S")
    except Exception:
        return None


def _save_bytes(content: bytes, dst: pathlib.Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        f.write(content)


def _download_bin(url: str) -> Tuple[bytes, dict]:
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.content, r.headers


def _save_current(det: str, root: pathlib.Path) -> Optional[str]:
    """
    Try current_c2/c3 endpoints first. If GIF, convert to PNG (when Pillow available).
    Returns saved path or None.
    """
    for url in CURRENT[det]:
        try:
            data, headers = _download_bin(url)
        except Exception as e:
            _log(f"{det}: current fetch failed {url} -> {e}")
            continue

        ts = _timestamp_from_headers(headers) or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        ext = os.path.splitext(url.split("?")[0])[1].lower()
        fname_base = f"{det}_{ts}"

        det_dir = root / det
        det_dir.mkdir(parents=True, exist_ok=True)

        if ext == ".gif":
            if Image:
                try:
                    im = Image.open(BytesIO(data))
                    # take first frame if animated; convert to L (grayscale) for consistency
                    im.seek(0)
                    if im.mode not in ("L", "LA"):
                        im = im.convert("L")
                    dst = det_dir / f"{fname_base}.png"
                    im.save(dst)
                    _log(f"{det}: saved converted PNG from GIF -> {dst.name}")
                    return str(dst)
                except Exception as e:
                    _log(f"{det}: GIF conversion failed: {e}")
                    # if conversion fails, just write the GIF
                    dst = det_dir / f"{fname_base}.gif"
                    _save_bytes(data, dst)
                    _log(f"{det}: saved GIF as-is -> {dst.name}")
                    return str(dst)
            else:
                dst = det_dir / f"{fname_base}.gif"
                _save_bytes(data, dst)
                _log(f"{det}: saved GIF (no Pillow) -> {dst.name}")
                return str(dst)
        else:
            # jpg/png
            dst = det_dir / f"{fname_base}{ext}"
            _save_bytes(data, dst)
            _log(f"{det}: saved current -> {dst.name}")
            return str(dst)

    return None


def _from_latest_page(det: str, count: int, root: pathlib.Path) -> List[str]:
    url = INDEX[det]
    try:
        html = requests.get(url, headers=HEADERS, timeout=60).text
    except Exception as e:
        _log(f"{det}: latest page fetch failed {url} -> {e}")
        return []

    links = []
    for m in HREF_OR_SRC_IMG.finditer(html):
        u = _abs_url(m.group(1))
        if _want(u):
            links.append(u)

    # de-dup preserving order
    seen = set()
    ordered = []
    for u in links:
        if u in seen:
            continue
        seen.add(u)
        ordered.append(u)

    if count > 0:
        ordered = ordered[:count]

    saved: List[str] = []
    det_dir = root / det
    det_dir.mkdir(parents=True, exist_ok=True)

    for u in ordered:
        fname = os.path.basename(u.split("?")[0])
        # prepend det if not present
        if not fname.upper().startswith(det + "_"):
            fname = f"{det}_{fname}"
        dst = det_dir / fname
        if dst.exists() and dst.stat().st_size > 0:
            continue
        try:
            data, _ = _download_bin(u)
            _save_bytes(data, dst)
            saved.append(str(dst))
            _log(f"{det}: saved from latest page -> {dst.name}")
        except Exception as e:
            _log(f"{det}: failed downloading {u} -> {e}")

    return saved


def fetch_window(hours_back: int = 6, step_min: int = 12, root: str = "frames") -> List[str]:
    """
    Pull newest images for both C2 and C3 detectors.
    Strategy:
      1) Try the 'current_c2/c3' endpoints to guarantee at least one frame.
      2) Also parse the latest index pages to grab several more recent frames.
    Returns absolute file paths saved (new downloads only).
    """
    root_path = pathlib.Path(root)
    est = max(2, min(48, (hours_back * 60) // max(1, step_min)))

    saved_all: List[str] = []

    for det in ("C2", "C3"):
        # Always try to get at least one "current" frame
        first = _save_current(det, root_path)
        if first:
            saved_all.append(first)

        # Then fetch a handful more from the listing page
        more = _from_latest_page(det, est, root_path)
        saved_all.extend(more)

    _log(f"summary: saved {len(saved_all)} new file(s)")
    return saved_all
