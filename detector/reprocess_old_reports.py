#!/usr/bin/env python3
"""
Re-process all old candidates_*.json files:
- Load each report
- For every track, load the crop image
- Run AI classification (same as live detector)
- Add ai_label + ai_score
- Save updated JSON (same filename)
"""

import json
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# --------------------------------------------------------------
# CONFIG – MUST MATCH detect_comets.py
# --------------------------------------------------------------
DETECTIONS_DIR = Path("detections")
CROPS_SUBDIR = DETECTIONS_DIR / "crops"
IMG_SIZE = {"C2": (512, 512), "C3": (1024, 1024)}
USE_AI = os.getenv("USE_AI_CLASSIFIER", "1") == "1"
AI_VETO = os.getenv("AI_VETO_ENABLED", "1") == "1"
AI_VETO_LABEL = os.getenv("AI_VETO_LABEL", "not_comet")
AI_VETO_MAX = float(os.getenv("AI_VETO_SCORE_MAX", "0.9"))

# --------------------------------------------------------------
# AI CLASSIFIER (EXACT SAME AS IN detect_comets.py)
# --------------------------------------------------------------
def classify_crop(crop):
    if not USE_AI:
        return {"label": "unknown", "score": 0.0}

    if crop.size == 0:
        return {"label": "not_comet", "score": 0.0}

    h, w = crop.shape
    center = crop[h//4:3*h//4, w//4:3*w//4]
    if center.size == 0:
        return {"label": "not_comet", "score": 0.0}

    brightness = np.mean(center)
    contrast = np.std(center)
    score = min(0.99, (brightness / 255) * 0.6 + (contrast / 50) * 0.4)
    label = "comet" if score > 0.6 else "not_comet"
    return {"label": label, "score": round(score, 3)}

# --------------------------------------------------------------
# MAIN REPROCESSOR
# --------------------------------------------------------------
def main():
    report_files = sorted(DETECTIONS_DIR.glob("candidates_*.json"), reverse=True)
    if not report_files:
        print("No old reports found in detections/")
        return

    print(f"Found {len(report_files)} reports to reprocess...")

    updated_count = 0
    comet_added_count = 0

    for report_path in report_files:
        print(f"\nProcessing {report_path.name}...", end="")
        try:
            with open(report_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f" Failed to load: {e}")
            continue

        modified = False
        comets_in_this = False

        for cand in data:
            # Skip if already has ai_label
            if cand.get("ai_label"):
                if cand["ai_label"] == "comet":
                    comets_in_this = True
                continue

            crop_path = Path(cand.get("crop_path", ""))
            if not crop_path.is_absolute():
                crop_path = DETECTIONS_DIR / crop_path

            if not crop_path.exists():
                print(f"\n  Missing crop: {crop_path}")
                continue

            crop = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
            if crop is None:
                continue

            ai = classify_crop(crop)

            # VETO logic (same as live)
            if AI_VETO and ai["label"] == AI_VETO_LABEL and ai["score"] > AI_VETO_MAX:
                cand["ai_label"] = "vetoed"
                cand["ai_score"] = ai["score"]
                modified = True
                continue

            cand["ai_label"] = ai["label"]
            cand["ai_score"] = ai["score"]
            modified = True
            if ai["label"] == "comet":
                comets_in_this = True

        if modified:
            # Backup old file
            backup = report_path.with_suffix(".json.bak")
            report_path.rename(backup)

            # Write new
            with open(report_path, "w") as f:
                json.dump(data, f, indent=2)

            print(f" Updated ({'comet' if comets_in_this else 'no comet'})")
            updated_count += 1
            if comets_in_this:
                comet_added_count += 1
        else:
            print(" No changes")

    print(f"\nDone! Updated {updated_count} reports.")
    print(f"   {comet_added_count} now have at least one AI 'comet' → will appear in Potential Hits")

if __name__ == "__main__":
    main()
