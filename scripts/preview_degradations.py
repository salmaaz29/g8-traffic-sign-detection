"""Mosaïque de contrôle : une image, chaque dégradation (lignes) x niveaux 0 à 5 (colonnes).
Usage : python scripts/preview_degradations.py --data-dir data/processed/gtsdb_4cls
        -> reports/degradations_preview.png"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.degradations.degrade import KINDS, apply, make_seed
from src.utils import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=str(ROOT / "data/processed/gtsdb_4cls"))
ap.add_argument("--split", default="test")
ap.add_argument("--image", default=None, help="nom du fichier, ex: 00012.png (défaut : 1re image avec panneau)")
ap.add_argument("--out", default=str(ROOT / "reports/degradations_preview.png"))
a = ap.parse_args()

data = Path(a.data_dir)
coco = json.loads((data / "coco" / f"{a.split}.json").read_text())
boxes_by = {}
for an in coco["annotations"]:
    x, y, w, h = an["bbox"]
    boxes_by.setdefault(an["image_id"], []).append((x, y, x + w, y + h))
im = next(i for i in coco["images"] if (i["file_name"] == a.image if a.image else i["id"] in boxes_by))
img = cv2.imread(str(data / "images" / a.split / im["file_name"]))
boxes = boxes_by.get(im["id"], [])

tw, th = 340, 200
rows = []
for kind in KINDS:
    tiles = []
    for level in range(6):
        out = apply(img, kind, level, seed=make_seed(im["file_name"], kind, level), boxes=boxes)
        t = cv2.resize(out, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.putText(t, f"{kind} {level}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(t)
    rows.append(np.hstack(tiles))
Path(a.out).parent.mkdir(exist_ok=True)
cv2.imwrite(a.out, np.vstack(rows))
print("Mosaïque enregistrée :", a.out, "| image :", im["file_name"])
