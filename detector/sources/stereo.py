from datetime import datetime, timedelta
from pathlib import Path
import requests
from bs4 import BeautifulSoup

FRAMES_DIR = Path("frames")

def fetch_stereo_frames(hours, step_min):
    now = datetime.utcnow()
    start = now - timedelta(hours=hours)
    frames = []
    timestamps = []
    downloaded = 0

    for sub in ("cor1", "cor2"):
        sub_dir = FRAMES_DIR / f"stereo_{sub}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        base = "https://stereo-ssc.nascom.nasa.gov/cgi-bin/images"
        params = {
            "telescope": f"{sub}_ahead",
            "start_time": start.strftime("%Y-%m-%d %H:%M"),
            "end_time": now.strftime("%Y-%m-%d %H:%M")
        }
        try:
            resp = requests.get(base, params=params, timeout=20)
            soup = BeautifulSoup(resp.text, 'html.parser')
            links = [a['href'] for a in soup.find_all('a', href=True) if f"{sub}_ahead" in a['href']]
            for link in links[:10]:
                url = f"https://stereo-ssc.nascom.nasa.gov{link}"
                name = Path(link).name
                path = sub_dir / name
                if not path.exists():
                    img_data = requests.get(url).content
                    path.write_bytes(img_data)
                    downloaded += 1
                frames.append(str(path))
                # Approximate timestamp from filename
                ts_str = name.split("_")[1:3]
                iso = f"2025-{ts_str[0][:2]}-{ts_str[0][2:4]}T{ts_str[1][:2]}:{ts_str[1][2:]}:00Z"
                timestamps.append(iso)
        except: pass

    return frames, timestamps, downloaded
