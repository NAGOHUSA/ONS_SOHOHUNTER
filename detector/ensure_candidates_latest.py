#!/usr/bin/env python3
"""
Write detections/candidates_latest.json every run.

- If the most recent candidates_*.json exists, copy its JSON array into candidates_latest.json
- If none exist, write an empty array []

This avoids front-end "nothing to load" states on quiet runs.
"""
from __future__ import annotations
import json, re
from pathlib import Path

DETECTIONS = Path(__file__).resolve().parent.parent / "detections"
DETECTIONS.mkdir(exist_ok=True)

def newest_candidates_json():
    pats = list(DETECTIONS.glob("candidates_*.json"))
    if not pats: return None
    # Sort descending by name (timestamp in filename) then by mtime as fallback
    pats.sort(key=lambda p: (p.name, p.stat().st_mtime), reverse=True)
    return pats[0]

def read_json_array(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list): return data
        if isinstance(data, dict) and isinstance(data.get("candidates"), list):
            return data["candidates"]
    except Exception:
        pass
    return []

def main():
    latest = newest_candidates_json()
    out = DETECTIONS / "candidates_latest.json"
    if latest is None:
        # write empty array, so file always exists
        out.write_text("[]\n", encoding="utf-8")
        print("Wrote empty candidates_latest.json")
        return
    arr = read_json_array(latest)
    with out.open("w", encoding="utf-8") as f:
        json.dump(arr, f, indent=2)
        f.write("\n")
    print(f"Updated candidates_latest.json from {latest.name} (n={len(arr)})")

if __name__ == "__main__":
    main()
