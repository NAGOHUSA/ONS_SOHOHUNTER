import cv2, json, os, math, pathlib, argparse, numpy as np
from datetime import datetime
from typing import List, Tuple
from fetch_lasco import fetch_window
from pathlib import Path
import re

# ---------- helpers (draw + sheets + annotate) ----------
def draw_tracks_overlay(base_img: np.ndarray, tracks, out_path: Path, radius=3, thickness=1):
    vis = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    # draw faint grid/crosshair for context
    h, w = base_img.shape[:2]
    cv2.line(vis, (w//2, 0), (w//2, h), (40,40,40), 1)
    cv2.line(vis, (0, h//2), (w, h//2), (40,40,40), 1)
    for idx, tr in enumerate(tracks, 1):
        pts = [(int(x), int(y)) for (_,x,y,_) in tr]
        for p in pts:
            cv2.circle(vis, p, radius, (0,255,255), -1)
        for i in range(1, len(pts)):
            cv2.line(vis, pts[i-1], pts[i], (0,200,255), thickness)
        if pts:
            cv2.putText(vis, f"#{idx}", pts[-1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,215,255), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)

def contact_sheet(images: list[np.ndarray], cols=4, pad=4):
    if not images:
        return None
    h, w = images[0].shape[:2]
    rows = int(np.ceil(len(images)/cols))
    sheet = np.full((rows*h + (rows+1)*pad, cols*w + (cols+1)*pad), 10, dtype=np.uint8)
    idx = 0
    y = pad
    for r in range(rows):
        x = pad
        for c in range(cols):
            if idx < len(images):
                img = images[idx]
                if img.shape != (h,w):
                    img = cv2.resize(img, (w,h))
                sheet[y:]()
