# scripts/make_label_csv.py
import glob, os, csv

def main():
    paths = sorted(glob.glob("detections/crops/*.png"))
    os.makedirs("detector/classifier", exist_ok=True)
    csv_path = "detector/classifier/training_labels.csv"

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label"])
        w.writeheader()
        for p in paths:
            w.writerow({"path": p, "label": "not_comet"})  # flip true comets to "comet"

    print(f"Wrote {csv_path} with {len(paths)} rows.")

if __name__ == "__main__":
    main()
