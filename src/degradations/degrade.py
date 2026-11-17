"""
Dégradations synthétiques pour tester la robustesse des détecteurs.

    from src.degradations.degrade import apply, KINDS, make_seed
    out = apply(image_bgr, "rain", level=3, seed=make_seed("00012.png", "rain", 3))

- kind  : "rain", "fog", "night", "blur", "occlusion", "jpeg"
- level : 0 (image intacte) puis 1 (léger) -> 5 (très sévère)
- seed  : rend la dégradation REPRODUCTIBLE : tous les modèles voient exactement la même image dégradée.
- boxes : (occlusion seulement) boîtes vérité terrain [(x1, y1, x2, y2), ...] en pixels.
          Les panneaux sont alors recouverts d'une fraction précise de leur surface.

Entrée et sortie : tableau BGR uint8 (H, W, 3), même taille.
"""
from __future__ import annotations

import zlib

import cv2
import numpy as np

KINDS = ["rain", "fog", "night", "blur", "occlusion", "jpeg"]
LEVELS = (1, 2, 3, 4, 5)


def make_seed(name: str, kind: str, level: int) -> int:
    """Graine déterministe (identique d'une machine à l'autre) pour une image donnée."""
    return zlib.crc32(f"{name}|{kind}|{level}".encode()) & 0xFFFFFFFF


# ----------------------------------------------------------------------------- pluie
def _rain(img, level, rng, boxes=None):
    h, w = img.shape[:2]
    density = (3e-4, 6e-4, 1e-3, 1.5e-3, 2.2e-3)[level - 1]   # gouttes par pixel
    length = max(5, int(h * (0.02, 0.03, 0.04, 0.05, 0.06)[level - 1]))
    darken = (0.95, 0.90, 0.85, 0.80, 0.75)[level - 1]          # ciel couvert
    haze = (0.05, 0.10, 0.15, 0.20, 0.25)[level - 1]            # voile gris
    thickness = 1 if level < 4 else 2

    n = int(h * w * density)
    xs, ys = rng.integers(0, w, n), rng.integers(0, h, n)
    vals = rng.integers(170, 256, n)
    slant = np.deg2rad(rng.uniform(-12, 12))                    # même inclinaison pour toutes
    dx, dy = int(length * np.sin(slant)), int(length * np.cos(slant))

    layer = np.zeros((h, w), np.uint8)
    for x, y, v in zip(xs, ys, vals):
        cv2.line(layer, (int(x), int(y)), (int(x) + dx, int(y) + dy), int(v), thickness)
    layer = cv2.GaussianBlur(layer, (3, 3), 0)

    f = img.astype(np.float32) * darken
    f = f * (1 - haze) + 190.0 * haze
    f += layer[..., None].astype(np.float32) * 0.55
    return np.clip(f, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------------- brouillard
def _fog(img, level, rng, boxes=None):
    """Modèle de diffusion atmosphérique : I = J*t + A*(1-t), t = exp(-beta*profondeur)."""
    h, w = img.shape[:2]
    beta = (0.4, 0.8, 1.2, 1.7, 2.3)[level - 1]
    airlight = 215.0
    y = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None]      # haut = loin, bas = proche
    small = rng.random((h // 64 + 2, w // 64 + 2)).astype(np.float32) - 0.5
    noise = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)  # nappes de brume
    depth = np.clip(0.6 + 0.4 * y + 0.3 * noise, 0.05, None)
    t = np.exp(-beta * depth)[..., None]
    f = img.astype(np.float32) * t + airlight * (1 - t)
    return np.clip(f, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------------------- nuit
def _night(img, level, rng, boxes=None):
    gamma = (1.3, 1.7, 2.1, 2.6, 3.2)[level - 1]
    gain = (0.8, 0.65, 0.5, 0.4, 0.3)[level - 1]
    sigma = (2, 4, 6, 9, 12)[level - 1]                          # bruit du capteur
    f = (img.astype(np.float32) / 255.0) ** gamma * gain * 255.0
    f *= np.array([1.05, 0.98, 0.90], np.float32)               # teinte bleutée (BGR)
    f += rng.normal(0, sigma, f.shape).astype(np.float32)
    return np.clip(f, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------------ flou de bougé
def _blur(img, level, rng, boxes=None):
    k = (5, 9, 15, 23, 31)[level - 1]                            # longueur du flou en pixels
    angle = rng.uniform(-10, 10)                                 # mouvement quasi horizontal
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0
    m = cv2.getRotationMatrix2D(((k - 1) / 2, (k - 1) / 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, m, (k, k))
    kernel /= max(float(kernel.sum()), 1e-6)
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT)


# ------------------------------------------------------------------------- occultation
def _fill(out, xa, ya, xb, yb, rng):
    h, w = out.shape[:2]
    xa, xb, ya, yb = max(0, xa), min(w, xb), max(0, ya), min(h, yb)
    if xb <= xa or yb <= ya:
        return
    base = rng.integers(25, 140)                                 # gris foncé à moyen (branche, camion...)
    patch = rng.normal(base, 8, (yb - ya, xb - xa, 1)) + rng.normal(0, 3, (yb - ya, xb - xa, 3))
    out[ya:yb, xa:xb] = np.clip(patch, 0, 255).astype(np.uint8)


def _occlusion(img, level, rng, boxes=None):
    """Avec boxes : chaque panneau est recouvert sur 10/20/30/40/50 % de sa surface,
    depuis un côté choisi au hasard. Sans boxes : patchs aléatoires dans l'image."""
    out = img.copy()
    h, w = img.shape[:2]
    frac = (0.10, 0.20, 0.30, 0.40, 0.50)[level - 1]

    if boxes:
        for x1, y1, x2, y2 in boxes:
            x1, y1, x2, y2 = (int(round(v)) for v in (x1, y1, x2, y2))
            bw, bh = x2 - x1, y2 - y1
            if bw < 2 or bh < 2:
                continue
            side = int(rng.integers(0, 4))                       # 0 haut, 1 bas, 2 gauche, 3 droite
            if side < 2:
                ph = max(1, int(round(bh * frac)))
                ya, yb = (y1, y1 + ph) if side == 0 else (y2 - ph, y2)
                _fill(out, x1, ya, x2, yb, rng)
            else:
                pw = max(1, int(round(bw * frac)))
                xa, xb = (x1, x1 + pw) if side == 2 else (x2 - pw, x2)
                _fill(out, xa, y1, xb, y2, rng)
    else:
        for _ in range(level + 1):
            pw, ph = int(w * rng.uniform(0.05, 0.15)), int(h * rng.uniform(0.05, 0.15))
            x, y = int(rng.integers(0, max(1, w - pw))), int(rng.integers(0, max(1, h - ph)))
            _fill(out, x, y, x + pw, y + ph, rng)
    return out


# ----------------------------------------------------------------------------- JPEG
def _jpeg(img, level, rng, boxes=None):
    quality = (60, 40, 25, 15, 8)[level - 1]
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


_FUNCS = {"rain": _rain, "fog": _fog, "night": _night, "blur": _blur,
          "occlusion": _occlusion, "jpeg": _jpeg}


def apply(image: np.ndarray, kind: str, level: int, seed: int = 0, boxes=None) -> np.ndarray:
    """Applique la dégradation `kind` au niveau `level` (0 = image inchangée)."""
    if level == 0:
        return image.copy()
    if kind not in _FUNCS:
        raise ValueError(f"Dégradation inconnue : {kind!r}. Choix : {KINDS}")
    if level not in LEVELS:
        raise ValueError(f"level doit valoir 0 à 5, reçu {level}")
    return _FUNCS[kind](image, level, np.random.default_rng(seed), boxes)
