import pathlib, datetime as dt, requests

HV = "https://api.helioviewer.org/v1"
SOURCES = [
    {"observatory":"SOHO","instrument":"LASCO","detector":"C2","measurement":"white-light"},
    {"observatory":"SOHO","instrument":"LASCO","detector":"C3","measurement":"white-light"},
]

def fetch_image(when_utc: dt.datetime, dst_path: pathlib.Path, source: dict) -> str | None:
    params = {
        "date": when_utc.strftime("%Y-%m-%dT%H:%M:%S"),
        "observatory": source["observatory"],
        "instrument": source["instrument"],
        "detector": source["detector"],
        "measurement": source["measurement"],
        "format": "png",
    }
    r = requests.get(f"{HV}/getClosestImage/", params=params, timeout=60)
    r.raise_for_status()
    j = r.json()
    url = j.get("url")
    if not url:
        return None
    img = requests.get(url, timeout=60)
    img.raise_for_status()
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "wb") as f:
        f.write(img.content)
    return str(dst_path)

def fetch_window(hours_back=6, step_min=12, root="frames") -> list[str]:
    root = pathlib.Path(root)
    now = dt.datetime.utcnow().replace(second=0, microsecond=0)
    times = [now - dt.timedelta(minutes=m) for m in range(0, hours_back*60, step_min)]
    saved: list[str] = []
    for src in SOURCES:
        det = src["detector"]
        for t in times:
            name = f"{det}_{t.strftime('%Y%m%d_%H%M')}.png"
            path = root / det / name
            if path.exists():
                continue
            try:
                p = fetch_image(t, path, src)
                if p:
                    saved.append(p)
            except Exception:
                # Missing frames happen; skip quietly
                pass
    return saved
