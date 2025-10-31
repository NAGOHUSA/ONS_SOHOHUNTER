# detector/ai_classifier.py
# Minimal, trainable stub classifier for SOHO crop patches.
# - If detector/classifier/model.npz exists, loads linear weights.
# - Else falls back to a simple heuristic.
# Returns: list of {"label": "...", "score": float in [0,1]}.

import os
import cv2
import numpy as np
from typing import List, Dict, Tuple

MODEL_DIR = os.path.join(os.path.dirname(__file__), "classifier")
MODEL_PATH = os.path.join(MODEL_DIR, "model.npz")


def _extract_features(gray: np.ndarray) -> np.ndarray:
    """Compute simple features: mean, std, edge_var, high_pct, low_pct, entropy-ish."""
    g = gray.astype(np.float32)
    mean = float(np.mean(g))
    std = float(np.std(g))
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    edge_var = float(np.var(lap))
    hi = float(np.percentile(g, 95))
    lo = float(np.percentile(g, 5))
    # entropy-ish: histogram spread
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
    p = hist / (np.sum(hist) + 1e-6)
    entropy = float(-np.sum(p * np.log2(p + 1e-9)))
    return np.array([mean, std, edge_var, hi, lo, entropy], dtype=np.float32)


def _score_linear(feats: np.ndarray, w: np.ndarray, b: float) -> float:
    s = float(np.dot(feats, w) + b)
    # squash to [0,1] with logistic
    return 1.0 / (1.0 + np.exp(-s))


def _load_model() -> Tuple[np.ndarray, float]:
    if os.path.exists(MODEL_PATH):
        data = np.load(MODEL_PATH)
        return data["w"].astype(np.float32), float(data["b"])
    return None, None


def _heuristic_score(feats: np.ndarray) -> float:
    # Simple rule of thumb: comets often appear as compact/high-contrast streaks.
    mean, std, edge_var, hi, lo, entropy = feats.tolist()
    # Normalize roughly
    std_n = min(std / 40.0, 1.0)
    edge_n = min(edge_var / 400.0, 1.0)
    span_n = min(max((hi - lo) / 255.0, 0.0), 1.0)
    ent_n = min(entropy / 5.0, 1.0)
    # Weighted blend
    score = 0.15 * std_n + 0.45 * edge_n + 0.30 * span_n + 0.10 * (1.0 - abs(ent_n - 0.5) * 2.0)
    return float(max(0.0, min(1.0, score)))


def classify_crop_batch(crop_paths: List[str]) -> List[Dict[str, float]]:
    # Load model if present
    w, b = _load_model()
    out: List[Dict[str, float]] = []
    for p in crop_paths:
        try:
            im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if im is None:
                out.append({"label": "unknown", "score": 0.0})
                continue
            # resize lightly for stability
            h, w0 = im.shape[:2]
            if max(h, w0) > 256:
                scale = 256.0 / max(h, w0)
                im = cv2.resize(im, (int(w0 * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            feats = _extract_features(im)
            if w is not None:
                s = _score_linear(feats, w, b)
            else:
                s = _heuristic_score(feats)

            label = "comet" if s >= 0.5 else "not_comet"
            out.append({"label": label, "score": float(s)})
        except Exception:
            out.append({"label": "unknown", "score": 0.0})
    return out
