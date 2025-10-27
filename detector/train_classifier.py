
# detector/train_classifier.py
import os, csv, numpy as np, cv2
from ai_classifier import _extract_features, MODEL_DIR, MODEL_PATH

CSV_PATH = os.path.join(MODEL_DIR, "training_labels.csv")

def load_dataset():
    X, y = [], []
    with open(CSV_PATH, "r", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            p = row["path"].strip()
            lab = row["label"].strip().lower()
            im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if im is None:
                print(f"[skip] cannot read {p}")
                continue
            # light resize for stability
            h,w = im.shape[:2]
            if max(h,w) > 256:
                s = 256.0 / max(h,w)
                im = cv2.resize(im, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
            feats = _extract_features(im)
            X.append(feats)
            y.append(1.0 if lab == "comet" else 0.0)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def fit_linear(X, y, l2=1e-3):
    # Solve ridge regression (linear -> logistic later in ai_classifier)
    # We approximate by fitting linear scores to labels in [0,1].
    nfeat = X.shape[1]
    A = X.T @ X + l2 * np.eye(nfeat, dtype=np.float32)
    b = X.T @ y
    w = np.linalg.solve(A, b)
    # Bias (intercept)
    mu = np.mean(X, axis=0)
    yhat = X @ w
    b0 = float(np.mean(y - yhat))
    return w.astype(np.float32), b0

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"Training CSV not found: {CSV_PATH}")
    X, y = load_dataset()
    if len(X) < 4:
        raise SystemExit("Not enough samples to train. Need at least 4 rows.")
    w, b0 = fit_linear(X, y)
    np.savez(MODEL_PATH, w=w, b=b0)
    print(f"Saved model to {MODEL_PATH}")
    # Quick sanity check
    yhat = 1.0 / (1.0 + np.exp(-(X @ w + b0)))
    acc = np.mean((yhat >= 0.5) == (y >= 0.5))
    print(f"Train logistic-approx accuracy: {acc:.3f}")

if __name__ == "__main__":
    main()
