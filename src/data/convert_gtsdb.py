"""
Conversion GTSDB -> YOLO (YOLO11, RT-DETR) + COCO JSON (Faster R-CNN)
avec split FIXE (sauvegardé dans splits/split.json).

Deux cas gérés automatiquement :
  - dataset complet (900 images annotées) : train/val = images 0..599, test = 600..899
  - TrainIJCNN2013 seul (600 images)      : split 70 / 15 / 15 par blocs d'images consécutives

Usage :
    python -m src.data.convert_gtsdb --mode 4cls --raw /kaggle/input
"""
import argparse
import json
import random
from pathlib import Path

import cv2
from tqdm import tqdm

from src.data.classes import get_class_names, map_class
from src.data.gtsdb import find_gt, list_images, parse_gt
from src.utils import ROOT, load_config


def _chunks(names, block):
    return [names[i:i + block] for i in range(0, len(names), block)]


def _flat(blocks):
    return sorted(n for b in blocks for n in b)


def make_split(names, last_idx, val_ratio, block, seed,
               val_ratio_no_test=0.15, test_ratio_no_test=0.15):
    names = sorted(names)  # stems : "00000", "00001", ...
    trainval = [n for n in names if int(n) <= last_idx]
    test = [n for n in names if int(n) > last_idx]
    rng = random.Random(seed)

    # blocs d'images consécutives : on évite que des images quasi identiques
    # (même séquence) se retrouvent dans deux sous-ensembles différents
    if test:  # dataset complet : le test officiel existe
        blocks = _chunks(trainval, block)
        rng.shuffle(blocks)
        n_val = round(len(blocks) * val_ratio)
        return {"train": _flat(blocks[n_val:]), "val": _flat(blocks[:n_val]),
                "test": sorted(test)}

    # pas de test officiel annoté (TrainIJCNN2013 seul) : split 3 voies
    blocks = _chunks(names, block)
    rng.shuffle(blocks)
    n_val = round(len(blocks) * val_ratio_no_test)
    n_test = round(len(blocks) * test_ratio_no_test)
    return {"train": _flat(blocks[n_val + n_test:]),
            "val": _flat(blocks[:n_val]),
            "test": _flat(blocks[n_val:n_val + n_test])}


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=cfg["dataset"]["mode"], choices=["4cls", "43cls"])
    ap.add_argument("--raw", default=None, help="dossier du dataset (ex: /kaggle/input)")
    ap.add_argument("--processed", default=None, help="dossier de sortie")
    args = ap.parse_args()
    mode = args.mode

    gt_path = find_gt(args.raw or ROOT / cfg["paths"]["raw"])
    raw = gt_path.parent
    print("gt.txt trouvé :", gt_path)
    out = Path(args.processed or ROOT / cfg["paths"]["processed"]) / f"gtsdb_{mode}"
    split_path = ROOT / cfg["paths"]["split"]

    files = list_images(raw)          # {stem: nom_fichier}
    assert files, f"Aucune image (.ppm/.png/.jpg) dans {raw}"
    print(f"{len(files)} images trouvées")
    names = list(files)
    ann = parse_gt(gt_path)
    class_names = get_class_names(mode)

    # --- split fixe : créé une fois, puis relu (sauf s'il n'a pas de test) ---
    split = None
    if split_path.exists():
        split = json.loads(split_path.read_text())
        if not split.get("test"):
            print("Ancien split sans images de test -> recréé.")
            split = None
        else:
            print(f"Split existant relu : {split_path}")
    if split is None:
        d = cfg["dataset"]
        split = make_split(names, d["train_val_last_index"], d["val_ratio"],
                           d["block_size"], cfg["seed"],
                           d.get("val_ratio_no_test", 0.15), d.get("test_ratio_no_test", 0.15))
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text(json.dumps(split, indent=1))
        print(f"Split créé : {split_path}")
    print({k: len(v) for k, v in split.items()})
    assert not (set(split["train"]) & set(split["val"]) | set(split["train"]) & set(split["test"])
                | set(split["val"]) & set(split["test"])), "Fuite entre sous-ensembles !"

    coco_dir = out / "coco"
    coco_dir.mkdir(parents=True, exist_ok=True)

    for sp, split_files in split.items():
        img_dir = out / "images" / sp
        lbl_dir = out / "labels" / sp
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        # supprime les fichiers d'un ancien split (évite les images dans le mauvais dossier)
        keep = set(split_files)
        for f in list(img_dir.glob("*.png")) + list(lbl_dir.glob("*.txt")):
            if f.stem not in keep:
                f.unlink()

        coco = {
            "images": [], "annotations": [],
            # id = index + 1 (0 est réservé au fond pour Faster R-CNN)
            "categories": [{"id": i + 1, "name": n} for i, n in enumerate(class_names)],
        }
        ann_id = 1
        for img_id, stem in enumerate(tqdm(split_files, desc=sp), start=1):
            img = cv2.imread(str(raw / files[stem]))
            h, w = img.shape[:2]
            dst = img_dir / f"{stem}.png"
            if not dst.exists():
                cv2.imwrite(str(dst), img)  # PNG : sans perte (Ultralytics ne lit pas .ppm)

            coco["images"].append({"id": img_id, "file_name": f"{stem}.png",
                                   "width": w, "height": h})
            lines = []
            for x1, y1, x2, y2, c in ann.get(stem, []):
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                bw, bh = x2 - x1, y2 - y1
                if bw <= 1 or bh <= 1:
                    continue
                k = map_class(c, mode)
                lines.append(f"{k} {(x1 + bw / 2) / w:.6f} {(y1 + bh / 2) / h:.6f} "
                             f"{bw / w:.6f} {bh / h:.6f}")
                coco["annotations"].append({
                    "id": ann_id, "image_id": img_id, "category_id": k + 1,
                    "bbox": [x1, y1, bw, bh], "area": bw * bh, "iscrowd": 0,
                    "orig_class": c,
                })
                ann_id += 1
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines))

        (coco_dir / f"{sp}.json").write_text(json.dumps(coco))

    yaml_txt = f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
    yaml_txt += "".join(f"  {i}: {n}\n" for i, n in enumerate(class_names))
    (out / "dataset.yaml").write_text(yaml_txt)
    print(f"Terminé -> {out}\nYOLO : {out/'dataset.yaml'}\nCOCO : {coco_dir}")


if __name__ == "__main__":
    main()
