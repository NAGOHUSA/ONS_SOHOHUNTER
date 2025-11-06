# detector/train_classifier.py
import os, csv, numpy as np, cv2

# Reuse your existing feature extractor + model paths
from ai_classifier import _extract_features, MODEL_DIR, MODEL_PATH

CSV_PATH = os.path.join(MODEL_DIR, "training_labels.csv")

def load_dataset():
    X, y = [], []
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"Training CSV not found: {CSV_PATH}")

    with open(CSV_PATH, "r", newline="") as f:
        rdr = csv.DictReader(f)
        for i, row in enumerate(rdr, start=2):  # start=2 includes header line
            if not row:
                print(f"[skip@line{i}] empty row")
                continue
            p = (row.get("path") or "").strip()
            lab = (row.get("label") or "").strip().lower()
            if not p or not lab:
                print(f"[skip@line{i}] missing path/label")
                continue
            if not os.path.exists(p):
                print(f"[skip@line{i}] not found: {p}")
                continue

            im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if im is None:
                print(f"[skip@line{i}] cannot read: {p}")
                continue

            # normalize size a bit for stable features
            h, w = im.shape[:2]
            if max(h, w) > 256:
                s = 256.0 / max(h, w)
                im = cv2.resize(im, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

            feats, _ = _extract_features(im)
            X.append(feats)
            y.append(1.0 if lab == "comet" else 0.0)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    if X.size == 0:
        raise SystemExit("No valid samples after cleaning. Check CSV paths + images exist.")
    return X, y

def fit_linear(X, y, l2=1e-3):
    nfeat = X.shape[1]
    A = X.T @ X + l2 * np.eye(nfeat, dtype=np.float32)
    b = X.T @ y
    w = np.linalg.solve(A, b)
    yhat = X @ w
    b0 = float(np.mean(y - yhat))
    return w.astype(np.float32), b0

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    X, y = load_dataset()
    if len(X) < 4:
        raise SystemExit(f"Need at least 4 valid rows to train (have {len(X)}).")

    w, b0 = fit_linear(X, y)
    np.savez(MODEL_PATH, w=w, b=b0)
    print(f"Saved model to {MODEL_PATH}")

    # quick sanity metric with logistic link
    yhat = 1.0 / (1.0 + np.exp(-(X @ w + b0)))
    acc = np.mean((yhat >= 0.5) == (y >= 0.5))
    print(f"Train logistic-approx accuracy: {acc:.3f}")

if __name__ == "__main__":
    main()
