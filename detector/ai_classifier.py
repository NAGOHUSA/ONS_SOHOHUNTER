# detector/ai_classifier.py
# Trainable stub classifier for SOHO crops with hard guards against overlays.
# If detector/classifier/model.npz exists, we use linear weights; otherwise a heuristic.
# Returns: list of {"label": "...", "score": float}

import os
import cv2
import numpy as np
import re
from typing import List, Dict, Tuple

MODEL_DIR = os.path.join(os.path.dirname(__file__), "classifier")
MODEL_PATH = os.path.join(MODEL_DIR, "model.npz")

# -------- Overlay / Artifact tests --------

def _has_timestamp_overlay(gray: np.ndarray) -> bool:
    """Detect very bright, high-contrast text blocks in corners/edges."""
    if gray is None or gray.size == 0:
        return False
    h, w = gray.shape
    if h < 32 or w < 32:
        return False

    cw = max(120, w // 8)
    ch = max(80,  h // 8)
    edge = max(22, min(h, w) // 40)

    regions = [
        gray[0:ch, 0:cw],
        gray[0:ch, w-cw:w],
        gray[h-ch:h, 0:cw],
        gray[h-ch:h, w-cw:w],
        gray[0:edge, :],
        gray[h-edge:h, :],
        gray[:, 0:edge],
        gray[:, w-edge:w],
    ]

    for reg in regions:
        if reg.size == 0:
            continue
        m = float(np.mean(reg))
        mx = int(np.max(reg))
        sd = float(np.std(reg))
        bright_ratio = float(np.count_nonzero(reg > 220)) / float(reg.size)
        if (m > 170 and mx > 240 and sd > 55) or (mx > 246 and bright_ratio > 0.12):
            return True
    return False

def _is_edge_artifact(gray: np.ndarray) -> bool:
    """Edge-only frames or boundary glow."""
    if gray is None or gray.size == 0:
        return False
    h, w = gray.shape
    if h < 24 or w < 24:
        return False

    ch0, ch1 = h // 3, 2 * h // 3
    cw0, cw1 = w // 3, 2 * w // 3
    center = gray[ch0:ch1, cw0:cw1]
    if center.size == 0:
        return True

    center_mean = float(np.mean(center))
    center_max  = int(np.max(center))

    m = max(6, min(h, w) // 12)
    edges = [
        gray[0:m, :], gray[h-m:h, :], gray[:, 0:m], gray[:, w-m:w]
    ]
    edge_mean = float(np.mean(np.concatenate([e.flatten() for e in edges if e.size])))

    if edge_mean > center_mean * 1.5 and center_max < 150:
        return True
    return False

def _near_border(gray: np.ndarray, border_px: int = 12) -> bool:
    """Brightest pixel too close to border."""
    if gray is None or gray.size == 0:
        return False
    h, w = gray.shape
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    x, y = maxLoc
    return (x < border_px or y < border_px or x > w - 1 - border_px or y > h - 1 - border_px)

# -------- Features / Scoring --------

def _extract_features(gray: np.ndarray) -> Tuple[np.ndarray, bool]:
    if gray is None or gray.size == 0:
        return np.zeros(6, np.float32), False

    overlay = _has_timestamp_overlay(gray)
    edge_art = _is_edge_artifact(gray)

    g = gray.astype(np.float32)
    mean = float(np.mean(g))
    std  = float(np.std(g))
    lap  = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    edge_var = float(np.var(lap))
    hi   = float(np.percentile(g, 95))
    lo   = float(np.percentile(g, 5))
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
    p = hist / (np.sum(hist) + 1e-6)
    entropy = float(-np.sum(p * np.log2(p + 1e-9)))

    feats = np.array([mean, std, edge_var, hi, lo, entropy], dtype=np.float32)

    if overlay or edge_art or _near_border(gray):
        return feats, True
    return feats, False

def _score_linear(feats: np.ndarray, w: np.ndarray, b: float) -> float:
    s = float(np.dot(feats, w) + b)
    return 1.0 / (1.0 + np.exp(-s))

def _load_model():
    if os.path.exists(MODEL_PATH):
        data = np.load(MODEL_PATH)
        return data["w"].astype(np.float32), float(data["b"])
    return None, None

def _heuristic_score(feats: np.ndarray) -> float:
    mean, std, edge_var, hi, lo, entropy = feats.tolist()
    std_n  = min(std / 40.0, 1.0)
    edge_n = min(edge_var / 400.0, 1.0)
    span_n = min(max((hi - lo) / 255.0, 0.0), 1.0)
    ent_n  = min(entropy / 5.0, 1.0)
    score = 0.15 * std_n + 0.45 * edge_n + 0.30 * span_n + 0.10 * (1.0 - abs(ent_n - 0.5) * 2.0)
    return float(max(0.0, min(1.0, score)))

# -------- Public API --------

def classify_crop_batch(crop_paths: List[str]) -> List[Dict[str, float]]:
    w, b = _load_model()
    out = []
    for p in crop_paths:
        try:
            im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if im is None:
                out.append({"label": "unknown", "score": 0.0})
                continue

            h, ww = im.shape[:2]
            if max(h, ww) > 256:
                sc = 256.0 / max(h, ww)
                im = cv2.resize(im, (int(ww * sc), int(h * sc)), interpolation=cv2.INTER_AREA)

            feats, looks_like_overlay = _extract_features(im)

            if looks_like_overlay:
                out.append({"label": "not_comet", "score": 0.01})
                continue

            s = _score_linear(feats, w, b) if w is not None else _heuristic_score(feats)
            out.append({"label": "comet" if s >= 0.5 else "not_comet", "score": float(s)})

        except Exception:
            out.append({"label": "unknown", "score": 0.0})
    return out
