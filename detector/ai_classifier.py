# detector/ai_classifier.py
# Minimal, trainable stub classifier for SOHO crop patches.
# - If detector/classifier/model.npz exists, loads linear weights.
# - Else falls back to a simple heuristic.
# - NOW WITH TIMESTAMP OVERLAY DETECTION
# Returns: list of {"label": "...", "score": float in [0,1]}.

import os
import cv2
import numpy as np
from typing import List, Dict

MODEL_DIR = os.path.join(os.path.dirname(__file__), "classifier")
MODEL_PATH = os.path.join(MODEL_DIR, "model.npz")


def _has_timestamp_overlay(gray: np.ndarray) -> bool:
    """
    Detect if image has timestamp overlay (bright text in corners).
    Timestamps are typically very bright, high-contrast regions in corners.
    """
    if gray.size == 0:
        return False
    
    h, w = gray.shape
    if h < 16 or w < 16:
        return False
    
    # Check all four corners for bright, high-contrast regions
    corner_size_h = min(h // 6, 80)
    corner_size_w = min(w // 6, 120)
    
    corners = [
        gray[0:corner_size_h, 0:corner_size_w],                    # top-left
        gray[0:corner_size_h, -corner_size_w:],                   # top-right
        gray[-corner_size_h:, 0:corner_size_w],                   # bottom-left
        gray[-corner_size_h:, -corner_size_w:]                    # bottom-right
    ]
    
    for corner in corners:
        if corner.size == 0:
            continue
        
        brightness = np.mean(corner)
        max_val = np.max(corner)
        std_val = np.std(corner)
        
        # Timestamps characteristics:
        # - Very bright (mean > 180, max near 255)
        # - High local contrast/std (text edges)
        # - Concentrated bright pixels
        bright_pixel_ratio = np.sum(corner > 220) / corner.size
        
        if (brightness > 180 and max_val > 240 and std_val > 60) or \
           (max_val > 245 and bright_pixel_ratio > 0.15):
            return True
    
    return False


def _is_edge_artifact(gray: np.ndarray) -> bool:
    """
    Detect if image is primarily an edge artifact or frame boundary issue.
    Real comets should have signal in the center region.
    """
    if gray.size == 0:
        return False
    
    h, w = gray.shape
    if h < 20 or w < 20:
        return False
    
    # Check center region - real comets should have signal here
    center_h_start = h // 3
    center_h_end = 2 * h // 3
    center_w_start = w // 3
    center_w_end = 2 * w // 3
    
    center = gray[center_h_start:center_h_end, center_w_start:center_w_end]
    if center.size == 0:
        return True
    
    center_brightness = np.mean(center)
    center_max = np.max(center)
    
    # Check edges
    edge_top = gray[0:h//10, :]
    edge_bottom = gray[-h//10:, :]
    edge_left = gray[:, 0:w//10]
    edge_right = gray[:, -w//10:]
    
    edge_brightness = np.mean([
        np.mean(edge_top) if edge_top.size > 0 else 0,
        np.mean(edge_bottom) if edge_bottom.size > 0 else 0,
        np.mean(edge_left) if edge_left.size > 0 else 0,
        np.mean(edge_right) if edge_right.size > 0 else 0
    ])
    
    # If edges are much brighter than center, likely an artifact
    if edge_brightness > center_brightness * 1.5 and center_max < 150:
        return True
    
    return False


def _extract_features(gray: np.ndarray) -> np.ndarray:
    """Compute simple features: mean, std, edge_var, high_pct, low_pct, entropy-ish."""
    
    # Early rejection for timestamp overlays
    if _has_timestamp_overlay(gray):
        # Return features that will score very low
        return np.array([30.0, 3.0, 5.0, 50.0, 20.0, 1.5], dtype=np.float32)
    
    # Early rejection for edge artifacts
    if _is_edge_artifact(gray):
        # Return features that will score very low
        return np.array([35.0, 4.0, 8.0, 55.0, 25.0, 1.8], dtype=np.float32)
    
    g = gray.astype(np.float32)
    mean = float(np.mean(g))
    std = float(np.std(g))
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    edge_var = float(np.var(lap))
    hi = float(np.percentile(g, 95))
    lo = float(np.percentile(g, 5))
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
    p = hist / (np.sum(hist) + 1e-6)
    entropy = float(-np.sum(p * np.log2(p + 1e-9)))
    return np.array([mean, std, edge_var, hi, lo, entropy], dtype=np.float32)


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
    std_n = min(std / 40.0, 1.0)
    edge_n = min(edge_var / 400.0, 1.0)
    span_n = min(max((hi - lo) / 255.0, 0.0), 1.0)
    ent_n = min(entropy / 5.0, 1.0)
    score = 0.15 * std_n + 0.45 * edge_n + 0.30 * span_n + 0.10 * (1.0 - abs(ent_n - 0.5) * 2.0)
    return float(max(0.0, min(1.0, score)))


def classify_crop_batch(crop_paths: List[str]) -> List[Dict[str, float]]:
    w, b = _load_model()
    out = []
    for p in crop_paths:
        try:
            im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if im is None:
                out.append({"label": "unknown", "score": 0.0})
                continue
            h, w0 = im.shape[:2]
            if max(h, w0) > 256:
                scale = 256.0 / max(h, w0)
                im = cv2.resize(im, (int(w0 * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            feats = _extract_features(im)
            s = _score_linear(feats, w, b) if w is not None else _heuristic_score(feats)
            label = "comet" if s >= 0.5 else "not_comet"
            out.append({"label": label, "score": float(s)})
        except Exception:
            out.append({"label": "unknown", "score": 0.0})
    return out
