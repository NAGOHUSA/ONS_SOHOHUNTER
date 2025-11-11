# detector/fetch_ccor1.py
import os, pathlib, requests, json, datetime, argparse
from astropy.io import fits
import cv2, numpy as np

BASE_URL = "https://www.swpc.noaa.gov/content/ccor1-latest-running-difference.fits"
# fallback list of the last N images (NOAA also provides a JSON index)
INDEX_URL = "https://services.swpc.noaa.gov/products/solar-images/ccor1/index.json"

def download_file(url: str, dst: pathlib.Path):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(r.content)

def fits_to_png(fits_path: pathlib.Path, png_path: pathlib.Path):
    with fits.open(fits_path) as hdul:
        data = hdul[0].data.astype(np.float32)
        # simple stretch
        data = np.nan_to_num(data)
        norm = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(str(png_path), norm.astype(np.uint8))

def fetch_window(hours_back: int = 6, step_min: int = 12, root: str = "frames") -> list[str]:
    """
    Pull the *most recent* CCOR-1 running-difference FITS files that are
    newer than `hours_back` ago and spaced at least `step_min` minutes.
    Returns list of downloaded PNG filenames (relative to repo root).
    """
    out_dir = pathlib.Path(root) / "CCOR1"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Get index of the last 48 h (more than enough)
    idx = requests.get(INDEX_URL, timeout=15).json()
    now = datetime.datetime.utcnow()
    cutoff = now - datetime.timedelta(hours=hours_back)

    candidates = []
    for entry in idx:
        ts = datetime.datetime.strptime(entry["time_tag"], "%Y-%m-%dT%H:%M:%SZ")
        if ts < cutoff:
            continue
        candidates.append((ts, entry["url"]))

    # sort newest first
    candidates.sort(reverse=True)

    downloaded = []
    last_ts = None
    for ts, url in candidates:
        if last_ts is not None:
            delta = (last_ts - ts).total_seconds() / 60
            if delta < step_min:
                continue
        # download
        fits_name = url.split("/")[-1]
        fits_path = out_dir / fits_name
        png_name = fits_name.replace(".fits", ".png")
        png_path = out_dir / png_name

        if not fits_path.exists():
            try:
                download_file(url, fits_path)
                fits_to_png(fits_path, png_path)
            except Exception as e:
                print(f"[CCOR1] skip {url}: {e}")
                continue

        downloaded.append(str(png_path))
        last_ts = ts

    return downloaded
