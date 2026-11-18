"""Dataset PyTorch pour Faster R-CNN, lit les JSON COCO de gtsdb_4cls.

Les category_id du JSON valent déjà 1..4 : c'est exactement la convention torchvision (0 = fond).
Augmentations : photométriques uniquement (pas de flip : il change le sens de certains panneaux,
pas de recadrage : les panneaux sont petits). Aucune dégradation météo ici, pour garder une
comparaison équitable avec YOLO11 / RT-DETR entraînés sans.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class GTSDBFrcnnDataset(Dataset):
    def __init__(self, images_dir, coco_json, train: bool = False, limit: int | None = None):
        coco = json.loads(Path(coco_json).read_text())
        self.images_dir = Path(images_dir)
        self.images = sorted(coco["images"], key=lambda i: i["id"])
        if limit:
            self.images = self.images[:limit]
        self.anns = defaultdict(list)
        for a in coco["annotations"]:
            self.anns[a["image_id"]].append(a)
        self.train = train

    def __len__(self):
        return len(self.images)

    @staticmethod
    def _augment(img: np.ndarray) -> np.ndarray:
        rng = np.random
        f = img.astype(np.float32)
        f = f * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25)            # contraste / luminosité
        if rng.rand() < 0.3:                                             # gamma (clair / sombre)
            f = 255.0 * (np.clip(f, 0, 255) / 255.0) ** rng.uniform(0.7, 1.5)
        if rng.rand() < 0.3:                                             # bruit capteur
            f += rng.normal(0, rng.uniform(1, 6), f.shape)
        out = np.clip(f, 0, 255).astype(np.uint8)
        if rng.rand() < 0.2:                                             # léger flou
            k = int(rng.choice([3, 5]))
            out = cv2.GaussianBlur(out, (k, k), 0)
        return out

    def __getitem__(self, i):
        im = self.images[i]
        img = cv2.imread(str(self.images_dir / im["file_name"]))
        if img is None:
            raise FileNotFoundError(self.images_dir / im["file_name"])
        if self.train:
            img = self._augment(img)
        img = np.ascontiguousarray(img[..., ::-1])                       # BGR -> RGB
        boxes, labels = [], []
        for a in self.anns.get(im["id"], []):
            x, y, w, h = a["bbox"]
            if w < 1 or h < 1:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(a["category_id"])
        target = {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                  "labels": torch.tensor(labels, dtype=torch.int64),
                  "image_id": torch.tensor([im["id"]])}
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))
