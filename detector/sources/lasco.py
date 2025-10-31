from datetime import datetime, timedelta
from pathlib import Path
import requests

FRAMES_DIR = Path("frames")

def fetch_lasco_frames(hours: int, step_min: int):
    now = datetime.utcnow()
    start = now - timedelta(hours=hours)
    all_frames = []
    all_timestamps = []
    downloaded = 0

    for sub in ("c2", "c3"):
        url_tmpl = (
            "https://soho.nascom.nasa.gov/data/REPROCESSING/Completed/"
            "{year}/{sub}/{date}/{date}_{time}_{sub}_1024.jpg"
        )
        t = start.replace(minute=0, second=0, microsecond=0)
        sub_dir = FRAMES_DIR / f"lasco_{sub}"
        sub_dir.mkdir(parents=True, exist_ok=True)

        while t < now:
            t += timedelta(minutes=step_min)
            if t > now:
                break

            date_str = t.strftime("%Y%m%d")
            time_str = t.strftime("%H%M")
            year = t.year
            iso = t.strftime("%Y-%m-%dT%H:%M:00Z")
            url = url_tmpl.format(year=year, sub=sub, date=date_str, time=time_str)
            path = sub_dir / f"lasco_{sub}_{date_str}_{time_str}.jpg"

            if path.exists():
                all_frames.append(str(path))
                all_timestamps.append(iso)
                downloaded += 1
                continue

            try:
                resp = requests.get(url, timeout=12)
                if resp.status_code == 404:
                    continue
                if resp.status_code == 200:
                    path.write_bytes(resp.content)
                    all_frames.append(str(path))
                    all_timestamps.append(iso)
                    downloaded += 1
            except Exception:
                pass

    return all_frames, all_timestamps, downloaded
