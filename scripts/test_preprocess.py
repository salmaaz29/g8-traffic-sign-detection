"""Teste le prétraitement sur une image : dégradations x méthodes -> mosaïque avant/après.

  python scripts/test_preprocess.py --image data/.../images/test/00012.png --out reports/preprocess_preview.png
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.degradations.degrade import apply, make_seed
from src.preprocessing.preprocess import METHODS, preprocess

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True)
ap.add_argument("--out", default="reports/preprocess_preview.png")
ap.add_argument("--level", type=int, default=3)
ap.add_argument("--kinds", nargs="+", default=["night", "blur", "fog", "rain", "jpeg"])
ap.add_argument("--width", type=int, default=340)
a = ap.parse_args()

img = cv2.imread(a.image)
assert img is not None, f"image introuvable : {a.image}"


def tile(im, text):
    s = a.width / im.shape[1]
    t = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.putText(t, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    return t


rows = []
for kind in a.kinds:
    deg = apply(img, kind, a.level, seed=make_seed("test", kind, a.level))
    cells = [tile(img, "original"), tile(deg, f"{kind} niv.{a.level}")]
    for m in METHODS:
        t0 = time.perf_counter()
        out = preprocess(deg, m)
        dt = time.perf_counter() - t0
        assert out.shape == img.shape and out.dtype == np.uint8, (m, out.shape, out.dtype)
        cells.append(tile(out, f"{m} {dt:.2f}s"))
    rows.append(np.hstack(cells))
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(a.out, np.vstack(rows))
print("OK ->", a.out)
