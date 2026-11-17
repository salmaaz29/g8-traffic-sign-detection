"""Wrapper Ultralytics : sert pour YOLO11 ET RT-DETR."""
import numpy as np

from src.models.base import Detector


class UltralyticsDetector(Detector):
    def __init__(self, weights: str, name: str = "yolo11", kind: str = "yolo",
                 imgsz: int = 1280, iou: float = 0.5, device: str | None = None):
        from ultralytics import RTDETR, YOLO
        self.model = (RTDETR if kind == "rtdetr" else YOLO)(weights)
        self.name, self.imgsz, self.iou, self.device = name, imgsz, iou, device
        self.class_names = list(self.model.names.values())

    def predict(self, image: np.ndarray, conf: float = 0.25) -> list[dict]:
        # Ultralytics considère un ndarray comme BGR (convention OpenCV)
        res = self.model.predict(image, conf=conf, iou=self.iou, imgsz=self.imgsz,
                                 device=self.device, verbose=False)[0]
        out = []
        for b in res.boxes:
            cid = int(b.cls)
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            out.append({"box": (x1, y1, x2, y2), "score": float(b.conf),
                        "class_id": cid, "label": self.model.names[cid]})
        return out
