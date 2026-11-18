"""Entraînement de Faster R-CNN ResNet50-FPN sur GTSDB (4 classes).

Test rapide (2 epochs, 40 images) :
  python -m src.models.train_frcnn --data-dir data/processed/gtsdb_4cls --epochs 2 --limit 40
Entraînement complet (Kaggle) :
  python -m src.models.train_frcnn --data-dir data/processed/gtsdb_4cls --epochs 30 --out weights

Sauvegarde : <out>/frcnn_best.pth  = STATE_DICT du modèle (meilleur mAP@0.5:0.95 sur val)
             <out>/frcnn_last.pth  = state_dict de la dernière epoch
"""
from __future__ import annotations

import argparse
import contextlib
import io
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.models.frcnn import build_model
from src.models.frcnn_dataset import GTSDBFrcnnDataset, collate_fn


@torch.inference_mode()
def validate(model, loader, coco_json, device):
    """mAP@0.5 et mAP@0.5:0.95 sur val (pycocotools, seuil très bas comme evaluate.py)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    model.eval()
    model.roi_heads.score_thresh = 0.001
    dets, ids = [], []
    for images, targets in loader:
        outs = model([im.to(device) for im in images])
        for t, o in zip(targets, outs):
            iid = int(t["image_id"])
            ids.append(iid)
            for b, s, l in zip(o["boxes"].cpu().tolist(), o["scores"].cpu().tolist(),
                               o["labels"].cpu().tolist()):
                dets.append({"image_id": iid, "category_id": int(l),
                             "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": s})
    if not dets:
        return 0.0, 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(coco_json))
        ev = COCOeval(gt, gt.loadRes(dets), "bbox")
        ev.params.imgIds = ids
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[1]), float(ev.stats[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="weights")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--limit", type=int, default=None, help="nb d'images (test rapide)")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, out = Path(a.data_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    train_ds = GTSDBFrcnnDataset(data / "images/train", data / "coco/train.json", train=True, limit=a.limit)
    val_ds = GTSDBFrcnnDataset(data / "images/val", data / "coco/val.json", train=False, limit=a.limit)
    train_dl = DataLoader(train_ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                          collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=a.workers, collate_fn=collate_fn)

    model = build_model(pretrained=True).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=a.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best = -1.0
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, run = time.time(), 0.0
        for k, (images, targets) in enumerate(train_dl):
            images = [im.to(device) for im in images]
            targets = [{n: v.to(device) for n, v in t.items()} for t in targets]
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = sum(model(images, targets).values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item()
        sched.step()
        m50, m5095 = validate(model, val_dl, data / "coco/val.json", device)
        print(f"epoch {ep:3d}/{a.epochs} | loss {run / len(train_dl):.4f} | "
              f"val mAP50 {m50:.3f} | mAP50-95 {m5095:.3f} | {time.time() - t0:.0f}s", flush=True)
        torch.save(model.state_dict(), out / "frcnn_last.pth")
        if m5095 > best:
            best = m5095
            torch.save(model.state_dict(), out / "frcnn_best.pth")
            print(f"  -> frcnn_best.pth sauvegardé (mAP50-95 = {best:.3f})")


if __name__ == "__main__":
    main()
