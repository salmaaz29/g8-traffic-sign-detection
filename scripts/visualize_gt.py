"""Vérifie les annotations : mosaïque de 10 images avec leurs boîtes.
Usage : python scripts/visualize_gt.py [--raw /kaggle/input]  ->  reports/gt_check.png"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.gtsdb import find_gt, list_images, parse_gt
from src.utils import ROOT, load_config

cfg = load_config()
ap = argparse.ArgumentParser()
ap.add_argument("--raw", default=None)
args = ap.parse_args()

gt_path = find_gt(args.raw or ROOT / cfg["paths"]["raw"])
raw = gt_path.parent
ann = parse_gt(gt_path)
files = list_images(raw)
print(f"gt.txt : {gt_path}")
print(f"{len(files)} images, {len(ann)} images annotées, {sum(len(v) for v in ann.values())} panneaux")

random.seed(0)
chosen = random.sample(sorted(k for k in ann if k in files), 10)
tiles = []
for stem in chosen:
    img = cv2.imread(str(raw / files[stem]))
    for x1, y1, x2, y2, c in ann[stem]:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img, str(c), (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    cv2.putText(img, files[stem], (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
    tiles.append(cv2.resize(img, (680, 400)))

rows = [np.hstack(tiles[i:i + 2]) for i in range(0, 10, 2)]
out = ROOT / cfg["paths"]["reports"] / "gt_check.png"
out.parent.mkdir(exist_ok=True)
cv2.imwrite(str(out), np.vstack(rows))
print("Mosaïque enregistrée :", out)
