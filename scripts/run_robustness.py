"""
Balaye toutes les dégradations x niveaux (+ image propre) pour un modèle, avec ou sans prétraitement,
et ajoute une ligne par combinaison dans un CSV. Les combinaisons déjà présentes sont sautées
(on peut donc relancer après une coupure de session Kaggle).

Usage :
  python scripts/run_robustness.py --model yolo11 --weights weights/yolo11_best.pt \
      --data-dir data/processed/gtsdb_4cls --preprocess none clahe --out results/yolo11.csv
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.degradations.degrade import KINDS
from src.evaluation.evaluate import append_csv, evaluate, get_preprocess, load_detector
from src.utils import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=["yolo11", "rtdetr", "frcnn"])
ap.add_argument("--weights", required=True)
ap.add_argument("--data-dir", default=str(ROOT / "data/processed/gtsdb_4cls"))
ap.add_argument("--split", default="test")
ap.add_argument("--preprocess", nargs="+", default=["none"])
ap.add_argument("--kinds", nargs="+", default=KINDS)
ap.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3, 4, 5])
ap.add_argument("--imgsz", type=int, default=1280)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--out", required=True)
a = ap.parse_args()

done = set()
if Path(a.out).exists():
    with open(a.out, newline="", encoding="utf-8") as f:
        done = {(r["model"], r["preprocess"], r["degradation"], int(r["level"])) for r in csv.DictReader(f)}

data = Path(a.data_dir)
det = load_detector(a.model, a.weights, a.imgsz)
jobs = [("clean", 0)] + [(k, l) for k in a.kinds for l in a.levels]
for pre in a.preprocess:
    pp = get_preprocess(pre)
    for kind, level in jobs:
        if (a.model, pre, kind, level) in done:
            continue
        res = evaluate(det, data / "images" / a.split, data / "coco" / f"{a.split}.json",
                       degradation=None if kind == "clean" else kind, level=level,
                       preprocess=pp, limit=a.limit, progress=False)
        append_csv(a.out, dict(res, model=a.model, preprocess=pre, degradation=kind, level=level))
        print(f"{a.model:7s} {pre:8s} {kind:10s} niv {level}  mAP50={res['map50']:.3f}  "
              f"mAP50-95={res['map50_95']:.3f}  P={res['precision']:.3f}  R={res['recall']:.3f}", flush=True)
print("Terminé ->", a.out)
