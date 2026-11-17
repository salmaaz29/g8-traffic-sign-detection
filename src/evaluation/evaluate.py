"""
Évaluation COMMUNE à tous les modèles (YOLO11, RT-DETR, Faster R-CNN).

Même code, mêmes seuils, mêmes images, mêmes dégradations pour tous -> comparaison équitable.

Métriques :
  - map50, map50_95 : pycocotools, détections avec un seuil de confiance très bas (0.001)
  - precision, recall : à un seuil de confiance d'exploitation (0.25) et IoU 0.5, classe respectée
  - mean_conf       : confiance moyenne des détections gardées à 0.25
  - ms_per_image, fps : temps de Detector.predict() seul (hors dégradation et prétraitement)

Prérequis côté détecteur : predict(image_bgr, conf) -> [{"box": (x1,y1,x2,y2), "score", "class_id"}]
avec class_id de 0 à N-1 (catégorie COCO correspondante = class_id + 1).

Usage :
  python -m src.evaluation.evaluate --model yolo11 --weights weights/yolo11_best.pt \
      --data-dir data/processed/gtsdb_4cls --split test
  python -m src.evaluation.evaluate ... --degradation rain --level 3 --preprocess clahe
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.degradations.degrade import apply as degrade_apply
from src.degradations.degrade import make_seed
from src.utils import ROOT


# ----------------------------------------------------------------------------- outils
def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _operating_point(gt_by_img, dets_by_img, conf_thr, iou_thr):
    """Précision / rappel / confiance moyenne au seuil conf_thr (appariement glouton, même classe)."""
    tp = fp = n_gt = 0
    scores, tp_scores = [], []
    for img_id, gts in gt_by_img.items():
        n_gt += len(gts)
        dets = sorted((d for d in dets_by_img.get(img_id, []) if d["score"] >= conf_thr),
                      key=lambda d: -d["score"])
        used = [False] * len(gts)
        for d in dets:
            scores.append(d["score"])
            best, best_iou = -1, iou_thr
            for j, g in enumerate(gts):
                if used[j] or g["class_id"] != d["class_id"]:
                    continue
                v = _iou(d["box"], g["box"])
                if v >= best_iou:
                    best, best_iou = j, v
            if best >= 0:
                used[best] = True
                tp += 1
                tp_scores.append(d["score"])
            else:
                fp += 1
    fn = n_gt - tp
    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / n_gt if n_gt else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "n_gt": n_gt,
        "n_det": len(scores),
        "mean_conf": float(np.mean(scores)) if scores else 0.0,
        "mean_conf_tp": float(np.mean(tp_scores)) if tp_scores else 0.0,
    }


def _coco_map(coco: dict, images: list, dets_json: list):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not dets_json:
        return 0.0, 0.0, {c["name"]: 0.0 for c in coco["categories"]}
    ids = [im["id"] for im in images]
    idset = set(ids)
    gt = COCO()
    gt.dataset = {"images": images, "categories": coco["categories"],
                  "annotations": [a for a in coco["annotations"] if a["image_id"] in idset]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        dt = gt.loadRes(dets_json)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    names = {c["id"]: c["name"] for c in coco["categories"]}
    per_class = {}
    for k, cid in enumerate(ev.params.catIds):
        p = ev.eval["precision"][0, :, k, 0, 2]        # IoU 0.5, toutes tailles, maxDets 100
        p = p[p > -1]
        per_class[names[cid]] = float(p.mean()) if p.size else float("nan")
    return max(0.0, float(ev.stats[1])), max(0.0, float(ev.stats[0])), per_class


# ----------------------------------------------------------------------- évaluation
def evaluate(detector, images_dir, coco_json, degradation=None, level=0, preprocess=None,
             conf_eval=0.001, conf_op=0.25, iou_thr=0.5, limit=None, warmup=3, progress=True):
    """Évalue `detector` sur les images de `images_dir` annotées dans `coco_json`.

    degradation / level : dégradation appliquée en mémoire (reproductible, rien n'est écrit sur disque)
    preprocess          : fonction image_bgr -> image_bgr appliquée APRÈS la dégradation (ou None)
    """
    coco = json.loads(Path(coco_json).read_text())
    images = sorted(coco["images"], key=lambda i: i["id"])
    if limit:
        images = images[:limit]
    n_cls = len(coco["categories"])

    gt_by_img = {im["id"]: [] for im in images}
    for a in coco["annotations"]:
        if a["image_id"] in gt_by_img:
            x, y, w, h = a["bbox"]
            gt_by_img[a["image_id"]].append({"box": (x, y, x + w, y + h),
                                             "class_id": a["category_id"] - 1})

    dets_json, dets_by_img, times = [], defaultdict(list), []
    for k, im in enumerate(tqdm(images, disable=not progress, desc="eval")):
        img = cv2.imread(str(Path(images_dir) / im["file_name"]))
        if img is None:
            raise FileNotFoundError(Path(images_dir) / im["file_name"])
        if degradation and level > 0:
            boxes = [g["box"] for g in gt_by_img[im["id"]]]
            img = degrade_apply(img, degradation, level,
                                seed=make_seed(im["file_name"], degradation, level), boxes=boxes)
        if preprocess is not None:
            img = preprocess(img)

        if k == 0:                                    # échauffement GPU (non chronométré)
            for _ in range(warmup):
                detector.predict(img, conf=conf_eval)
        t0 = time.perf_counter()
        dets = detector.predict(img, conf=conf_eval)
        times.append(time.perf_counter() - t0)

        for d in dets:
            if not 0 <= d["class_id"] < n_cls:
                continue
            x1, y1, x2, y2 = d["box"]
            dets_by_img[im["id"]].append({"box": (x1, y1, x2, y2), "score": d["score"],
                                          "class_id": d["class_id"]})
            dets_json.append({"image_id": im["id"], "category_id": d["class_id"] + 1,
                              "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(d["score"])})

    map50, map50_95, per_class = _coco_map(coco, images, dets_json)
    op = _operating_point(gt_by_img, dets_by_img, conf_op, iou_thr)
    ms = 1000 * float(np.mean(times))
    return {"n_images": len(images), "map50": map50, "map50_95": map50_95,
            "precision": op["precision"], "recall": op["recall"],
            "mean_conf": op["mean_conf"], "mean_conf_tp": op["mean_conf_tp"],
            "tp": op["tp"], "fp": op["fp"], "fn": op["fn"], "n_gt": op["n_gt"], "n_det": op["n_det"],
            "ms_per_image": ms, "fps": 1000.0 / ms if ms > 0 else 0.0,
            "per_class_ap50": per_class}


# ------------------------------------------------------------------- aides CLI / CSV
def load_detector(model: str, weights: str, imgsz: int = 1280):
    if model in ("yolo11", "rtdetr"):
        from src.models.yolo import UltralyticsDetector
        return UltralyticsDetector(weights, name=model, kind="rtdetr" if model == "rtdetr" else "yolo",
                                   imgsz=imgsz)
    if model == "frcnn":
        from src.models.frcnn import FasterRCNNDetector   # écrit par la personne B
        return FasterRCNNDetector(weights)
    raise ValueError(f"Modèle inconnu : {model} (yolo11 | rtdetr | frcnn)")


def get_preprocess(name):
    if name in (None, "", "none"):
        return None
    from src.preprocessing.preprocess import preprocess    # écrit par la personne B
    return lambda im: preprocess(im, name)


CSV_FIELDS = ["model", "preprocess", "degradation", "level", "n_images", "map50", "map50_95",
              "precision", "recall", "mean_conf", "mean_conf_tp", "tp", "fp", "fn", "n_gt", "n_det",
              "ms_per_image", "fps", "per_class_ap50"]


def append_csv(path, row: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    row = dict(row, per_class_ap50=json.dumps(row.get("per_class_ap50", {})))
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["yolo11", "rtdetr", "frcnn"])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-dir", default=str(ROOT / "data/processed/gtsdb_4cls"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--degradation", default=None)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--preprocess", default="none")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None, help="fichier CSV où ajouter le résultat")
    a = ap.parse_args()

    data = Path(a.data_dir)
    det = load_detector(a.model, a.weights, a.imgsz)
    res = evaluate(det, data / "images" / a.split, data / "coco" / f"{a.split}.json",
                   degradation=a.degradation, level=a.level, preprocess=get_preprocess(a.preprocess),
                   limit=a.limit)
    print(f"\n{a.model} | prétraitement={a.preprocess} | dégradation={a.degradation or 'aucune'} "
          f"niveau={a.level} | {res['n_images']} images")
    for k in ["map50", "map50_95", "precision", "recall", "mean_conf", "ms_per_image", "fps"]:
        print(f"  {k:13s}: {res[k]:.3f}")
    print("  AP50 par classe :", {k: round(v, 3) for k, v in res["per_class_ap50"].items()})
    if a.out:
        append_csv(a.out, dict(res, model=a.model, preprocess=a.preprocess,
                               degradation=a.degradation or "clean", level=a.level))


if __name__ == "__main__":
    main()
