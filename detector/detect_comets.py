from datetime import datetime, timedelta
from pathlib import Path
import requests

FRAMES_DIR = Path("frames")

def fetch_goes_frames(hours: int, step_min: int):
    now = datetime.utcnow()
    start = now - timedelta(hours=hours)
    frames = []
    timestamps = []
    downloaded = 0

    sub_dir = FRAMES_DIR / "goes_ccor1"
    sub_dir.mkdir(parents=True, exist_ok=True)

    # Use NOAA JSON index
    try:
        index_url = "https://services.swpc.noaa.gov/json/goes/primary/"
        resp = requests.get(index_url, timeout=10)
        if resp.status_code != 200:
            raise Exception("Index failed")
        data = resp.json()

        recent = []
        for entry in data:
            if 'ccor1' not in entry.get('products', []):
                continue
            try:
                ts = datetime.fromisoformat(entry['time_tag'].rstrip('Z'))
                if ts >= start:
                    recent.append((ts, entry))
            except:
                continue

        recent.sort(key=lambda x: x[0])
        recent = recent[-20:]

        for ts, entry in recent:
            date_str = ts.strftime("%Y%m%d")
            time_str = ts.strftime("%H%M%S")
            iso = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
            filename = f"goes19_ccor1_{date_str}_{time_str}.png"
            url = f"https://www.ngdc.noaa.gov/stp/satellite/goes/dataaccess.html/goes19/ccor1/{ts.year}/{ts.month:02d}/{ts.day:02d}/{filename}"
            path = sub_dir / filename

            if path.exists():
                frames.append(str(path))
                timestamps.append(iso)
                downloaded += 1
                continue

            try:
                img_resp = requests.get(url, timeout=12)
                if img_resp.status_code == 200:
                    path.write_bytes(img_resp.content)
                    frames.append(str(path))
                    timestamps.append(iso)
                    downloaded += 1
            except:
                pass
    except Exception as e:
        print(f"GOES fetch failed: {e}")

    return frames, timestamps, downloaded
