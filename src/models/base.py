"""Contrat commun : tous les modèles exposent la même méthode predict()."""
from abc import ABC, abstractmethod

import numpy as np


class Detector(ABC):
    name: str = "base"
    class_names: list[str] = []

    @abstractmethod
    def predict(self, image: np.ndarray, conf: float = 0.25) -> list[dict]:
        """
        image : tableau BGR (H, W, 3), uint8 (format OpenCV)
        retourne une liste de détections :
          [{"box": (x1, y1, x2, y2), "score": 0.87, "class_id": 2, "label": "mandatory"}, ...]
        coordonnées en pixels de l'image d'entrée.
        """
