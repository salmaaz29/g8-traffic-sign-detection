"""Faster R-CNN (ResNet50-FPN) : construction du modèle + wrapper Detector (personne B).

Convention torchvision : label 0 = fond, 1..4 = classes. Le wrapper retire ce +1 :
class_id (0..3) = label - 1, comme YOLO11 / RT-DETR, pour que evaluate.py fonctionne tel quel.
"""
from __future__ import annotations

import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.models.base import Detector

CLASS_NAMES = ["prohibitory", "danger", "mandatory", "other"]
NUM_CLASSES = len(CLASS_NAMES) + 1          # + fond
MIN_SIZE, MAX_SIZE = 800, 1360              # taille native GTSDB : on n'écrase pas les petits panneaux


def build_model(pretrained: bool = True) -> torch.nn.Module:
    """Faster R-CNN ResNet50-FPN, tête remplacée pour 4 classes + fond."""
    weights = "DEFAULT" if pretrained else None
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=weights, weights_backbone=None if not pretrained else "DEFAULT",
        min_size=MIN_SIZE, max_size=MAX_SIZE, box_detections_per_img=100)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    return model


class FasterRCNNDetector(Detector):
    def __init__(self, weights_path: str, device: str | None = None):
        self.name = "frcnn"
        self.class_names = list(CLASS_NAMES)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = build_model(pretrained=False)
        obj = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict) and "model" in obj:        # checkpoint {"model": state_dict, ...}
            obj = obj["model"]
        self.model.load_state_dict(obj)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, image: np.ndarray, conf: float = 0.25) -> list[dict]:
        # image : BGR uint8 (H, W, 3) -> RGB float [0,1]
        rgb = np.ascontiguousarray(image[..., ::-1])
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0).to(self.device)
        self.model.roi_heads.score_thresh = float(conf)
        out = self.model([x])[0]                            # boîtes déjà en pixels de l'image reçue
        boxes = out["boxes"].cpu().numpy()
        scores = out["scores"].cpu().numpy()
        labels = out["labels"].cpu().numpy().astype(int)
        return [{"box": tuple(float(v) for v in boxes[i]), "score": float(scores[i]),
                 "class_id": int(labels[i]) - 1, "label": CLASS_NAMES[int(labels[i]) - 1]}
                for i in range(len(labels))]
