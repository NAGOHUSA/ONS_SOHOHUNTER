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
    t = start.replace(minute=0, second=0, microsecond=0)

    while t < now:
        t += timedelta(minutes=step_min)
        if t > now:
            break

        date_str = t.strftime("%Y%m%d")
        time_str = t.strftime("%H%M")
        iso = t.strftime("%Y-%m-%dT%H:%M:00Z")
        url = f"https://www.swpc.noaa.gov/content/goes19-ccor1/{date_str}/{date_str}_{time_str}_ccor1.png"
        path = sub_dir / f"goes_ccor1_{date_str}_{time_str}.png"

        if path.exists():
            frames.append(str(path))
            timestamps.append(iso)
            downloaded += 1
            continue

        try:
            resp = requests.get(url, timeout=12)
            if resp.status_code == 200:
                path.write_bytes(resp.content)
                frames.append(str(path))
                timestamps.append(iso)
                downloaded += 1
        except Exception:
            pass

    return frames, timestamps, downloaded
