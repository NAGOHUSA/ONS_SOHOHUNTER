from datetime import datetime, timedelta
from pathlib import Path
import requests

FRAMES_DIR = Path("frames")

def fetch_lasco_frames(hours, step_min):
    now = datetime.utcnow()
    start = now - timedelta(hours=hours)
    frames = {"lasco_c2": [], "lasco_c3": []}
    timestamps = {"lasco_c2": [], "lasco_c3": []}
    downloaded = {"lasco_c2": 0, "lasco_c3": 0}

    for sub in ("c2", "c3"):
        url_tmpl = f"https://soho.nascom.nasa.gov/data/REPROCESSING/Completed/{{year}}/{sub}/{{date}}/{{date}}_{{time}}_{sub}_1024.jpg"
        t = start.replace(minute=0, second=0, microsecond=0)
        sub_dir = FRAMES_DIR / f"lasco_{sub}"
        sub_dir.mkdir(parents=True, exist_ok=True)

        while t < now:
            t += timedelta(minutes=step_min)
            if t > now: break
            date_str = t.strftime("%Y%m%d")
            time_str = t.strftime("%H%M")
            year = t.year
            iso = t.strftime("%Y-%m-%dT%H:%M:00Z")
            url = url_tmpl.format(year=year, date=date_str, time=time_str)
            path = sub_dir / f"lasco_{sub}_{date_str}_{time_str}.jpg"

            if path.exists():
                frames[f"lasco_{sub}"].append(str(path))
                timestamps[f"lasco_{sub}"].append(iso)
                downloaded[f"lasco_{sub}"] += 1
                continue

            try:
                resp = requests.get(url, timeout=12)
                if resp.status_code == 404:
                    continue
                if resp.status_code == 200:
                    path.write_bytes(resp.content)
                    frames[f"lasco_{sub}"].append(str(path))
                    timestamps[f"lasco_{sub}"].append(iso)
                    downloaded[f"lasco_{sub}"] += 1
            except: pass

    # Flatten for tracker
    all_frames = frames["lasco_c2"] + frames["lasco_c3"]
    all_ts = timestamps["lasco_c2"] + timestamps["lasco_c3"]
    all_dl = downloaded["lasco_c2"] + downloaded["lasco_c3"]
    return all_frames, all_ts, all_dl
