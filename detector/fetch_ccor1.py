# detector/fetch_ccor1.py
"""
Fetch CCOR-1 (GOES-19) running-difference coronagraph images.
- Public FITS files from NOAA SWPC
- Converts to 8-bit PNG (same format as LASCO)
- Respects --hours and --step-min
- Saves to frames/CCOR1/
"""

import os
import pathlib
import requests
import json
import datetime
import argparse
from typing import List

import cv2
import numpy as np
from astropy.io import fits


# ----------------------------------------------------------------------
# Configuration (can be overridden with env vars if needed later)
# ----------------------------------------------------------------------
BASE_INDEX_URL = os.getenv(
    "CCOR1_INDEX_URL",
    "https://services.swpc.noaa.gov/products/solar-images/ccor1/index.json"
)
# Direct latest file (fallback)
LATEST_FITS_URL = "https://www.swpc.noaa.gov/content/ccor1-latest-running-difference.fits"


def download_file(url: str, dst: pathlib.Path) -> None:
    """Download with error handling."""
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(r.content)
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}")


def fits_to_png(fits_path: pathlib.Path, png_path: pathlib.Path) -> None:
    """Convert FITS → normalized 8-bit PNG."""
    try:
        with fits.open(fits_path) as hdul:
            data = hdul[0].data
            if data is None:
                raise ValueError("Empty FITS data")

            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            norm = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
            cv2.imwrite(str(png_path), norm.astype(np.uint8))
    except Exception as e:
        raise RuntimeError(f"FITS → PNG failed for {fits_path}: {e}")


def fetch_window(
    hours_back: int = 6,
    step_min: int = 12,
    root: str = "frames"
) -> List[str]:
    """
    Download CCOR-1 running-difference images newer than `hours_back` ago,
    spaced at least `step_min` minutes apart.

    Returns list of downloaded PNG paths (relative to repo root).
    """
    out_dir = pathlib.Path(root) / "CCOR1"
    out_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.datetime.utcnow()
    cutoff = now - datetime.timedelta(hours=hours_back)

    # ------------------------------------------------------------------
    # 1. Get index of recent files
    # ------------------------------------------------------------------
    try:
        index = requests.get(BASE_INDEX_URL, timeout=15).json()
    except Exception as e:
        print(f"[CCOR1] Index fetch failed ({e}), falling back to latest only.")
        index = []

    candidates = []
    for entry in index:
        try:
            ts_str = entry.get("time_tag")
            url = entry.get("url")
            if not ts_str or not url:
                continue
            ts = datetime.datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ")
            if ts >= cutoff:
                candidates.append((ts, url))
        except Exception:
            continue

    # Add latest as fallback
    candidates.append((now, LATEST_FITS_URL))
    candidates.sort(reverse=True)  # newest first

    downloaded_pngs = []
    last_ts = None

    for ts, url in candidates:
        # Enforce spacing
        if last_ts is not None:
            delta_min = (last_ts - ts).total_seconds() / 60.0
            if delta_min < step_min:
                continue

        fits_name = url.split("/")[-1].split("?")[0]
        if not fits_name.endswith(".fits"):
            fits_name = f"{ts.strftime('%Y%m%d_%H%M%S')}.fits"
        fits_path = out_dir / fits_name
        png_name = fits_name.replace(".fits", ".png")
        png_path = out_dir / png_name

        # Skip if already have PNG
        if png_path.exists():
            downloaded_pngs.append(str(png_path))
            last_ts = ts
            continue

        # Download + convert
        try:
            download_file(url, fits_path)
            fits_to_png(fits_path, png_path)
            downloaded_pngs.append(str(png_path))
            print(f"[CCOR1] Saved {png_path.name}")
            last_ts = ts
        except Exception as e:
            print(f"[CCOR1] Skip {url}: {e}")
            if fits_path.exists():
                fits_path.unlink(missing_ok=True)
            continue

    return downloaded_pngs


# ----------------------------------------------------------------------
# CLI for local testing
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=6)
    parser.add_argument("--step-min", type=int, default=12)
    args = parser.parse_args()

    files = fetch_window(hours_back=args.hours, step_min=args.step_min)
    print(f"Downloaded {len(files)} CCOR-1 images.")
