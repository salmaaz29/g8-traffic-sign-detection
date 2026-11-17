"""Test rapide : GPU dispo ? YOLO11 pré-entraîné (COCO) tourne sur une image ?
Usage : python scripts/test_pretrained.py chemin/image.png"""
import sys
import time
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.yolo import UltralyticsDetector

print("PyTorch :", torch.__version__)
print("CUDA dispo :", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

img = cv2.imread(sys.argv[1])
det = UltralyticsDetector("yolo11n.pt", name="yolo11n-coco", imgsz=640)  # téléchargé auto
det.predict(img)  # warm-up
t = time.time()
dets = det.predict(img, conf=0.25)
print(f"{len(dets)} détections en {(time.time() - t) * 1000:.0f} ms")
for d in dets:
    x1, y1, x2, y2 = map(int, d["box"])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f'{d["label"]} {d["score"]:.2f}', (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.imwrite("reports/test_pretrained.png", img)
print("Résultat : reports/test_pretrained.png")
