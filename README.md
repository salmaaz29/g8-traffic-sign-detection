# G8 — Détection de panneaux : comparaison YOLO11 / Faster R-CNN / RT-DETR

## Installation
```bash
python -m venv .venv && source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```
(Si `pycocotools` échoue sous Windows : `pip install pycocotools-windows`.)

## Données (Personne A)
1. Télécharger **FullIJCNN2013.zip** (GTSDB) depuis https://benchmark.ini.rub.de/gtsdb_dataset.html
   (ou un miroir Kaggle « GTSDB »), puis le dézipper dans `data/raw/` pour obtenir
   `data/raw/FullIJCNN2013/00000.ppm ... gt.txt`.
2. Vérifier les annotations : `python scripts/visualize_gt.py` -> `reports/gt_check.png`
3. Convertir : `python -m src.data.convert_gtsdb --mode 4cls`
   -> `data/processed/gtsdb_4cls/` (dataset.yaml pour YOLO/RT-DETR, coco/*.json pour Faster R-CNN)
4. **Committer `splits/split.json`** : tout le monde utilise exactement les mêmes images.

## Environnement / GPU (Personne B)
```bash
python scripts/test_pretrained.py data/processed/gtsdb_4cls/images/val/00010.png
```
Sur Colab : Exécution > Modifier le type d'exécution > GPU T4, puis
```python
!git clone <url_du_depot> && cd g8-traffic-sign-detection && pip install -r requirements.txt
```

## Entraînement (étape 2, aperçu)
```python
from ultralytics import YOLO
YOLO("yolo11s.pt").train(data="data/processed/gtsdb_4cls/dataset.yaml", imgsz=1280, epochs=100, batch=8, seed=42)
```
