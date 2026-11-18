"""Prétraitement des images dégradées AVANT la détection (personne B).

    from src.preprocessing.preprocess import preprocess
    out = preprocess(image_bgr, "clahe")      # même taille, BGR uint8

Méthodes demandées par l'évaluation : "clahe", "dehaze", "denoise".
Méthodes supplémentaires (utilisables avec --preprocess) : "night", "sharpen".

  clahe   : égalisation locale du contraste (canal L de LAB). Si l'image est sombre, une correction
            gamma adaptative l'éclaircit d'abord -> aide surtout la NUIT et le brouillard léger.
  dehaze  : dark channel prior (He et al.) avec affinage par filtre guidé -> BROUILLARD.
  denoise : fastNlMeansDenoisingColored -> bruit capteur / JPEG.
  night   : gamma adaptatif + débruitage fort + CLAHE (la chaîne complète pour la nuit).
  sharpen : unsharp mask -> FLOU de bougé léger à moyen.
"""
from __future__ import annotations

import cv2
import numpy as np

METHODS = ["clahe", "dehaze", "denoise", "night", "sharpen"]


# ---------------------------------------------------------------------------- outils
def _adaptive_gamma_L(l: np.ndarray, target: float = 0.45, dark_thr: float = 0.35) -> np.ndarray:
    """Éclaircit le canal L seulement si l'image est sombre (luminance moyenne < dark_thr)."""
    m = float(l.mean()) / 255.0
    if m >= dark_thr:
        return l
    gamma = float(np.clip(np.log(target) / np.log(max(m, 1e-3)), 0.3, 1.0))
    lut = (255.0 * (np.arange(256) / 255.0) ** gamma).astype(np.uint8)
    return cv2.LUT(l, lut)


def _clahe(img, clip=3.0, tile=8, gamma=True):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    if gamma:
        l = _adaptive_gamma_L(l)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _denoise(img, h=6, h_color=6):
    return cv2.fastNlMeansDenoisingColored(img, None, h, h_color, 5, 15)


def _guided_filter(guide, src, r, eps):
    k = (2 * r + 1, 2 * r + 1)
    mean = lambda x: cv2.boxFilter(x, -1, k)
    mg, ms = mean(guide), mean(src)
    cov = mean(guide * src) - mg * ms
    var = mean(guide * guide) - mg * mg
    a = cov / (var + eps)
    b = ms - a * mg
    return mean(a) * guide + mean(b)


def _dehaze(img, omega=0.85, t0=0.1, patch=15):
    """Dark channel prior : J = (I - A) / max(t, t0) + A."""
    f = img.astype(np.float32) / 255.0
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    dark = cv2.erode(f.min(axis=2), ker)
    n = max(1, int(dark.size * 0.001))                         # lumière atmosphérique A
    idx = np.argpartition(dark.ravel(), -n)[-n:]
    cand = f.reshape(-1, 3)[idx]
    A = np.clip(cand[cand.sum(axis=1).argmax()], 0.3, 0.98)
    t = 1.0 - omega * cv2.erode((f / A).min(axis=2), ker)      # transmission brute
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    t = np.clip(_guided_filter(gray, t, r=40, eps=1e-3), t0, 1.0)
    j = (f - A) / t[..., None] + A
    return np.clip(j * 255.0, 0, 255).astype(np.uint8)


def _night(img):
    """Éclaircir (gamma adaptatif) -> débruiter (le bruit est amplifié par l'éclaircissement) -> CLAHE."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    lifted = cv2.cvtColor(cv2.merge([_adaptive_gamma_L(l), a, b]), cv2.COLOR_LAB2BGR)
    return _clahe(_denoise(lifted, h=15, h_color=15), clip=2.0, gamma=False)


def _sharpen(img, sigma=2.0, amount=1.5):
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)


# ---------------------------------------------------------------------------- API
def preprocess(image: np.ndarray, method: str) -> np.ndarray:
    """image BGR uint8 (H, W, 3) -> image BGR uint8 de LA MÊME TAILLE."""
    if method in (None, "", "none"):
        return image
    if method == "clahe":
        out = _clahe(image)
    elif method == "dehaze":
        out = _dehaze(image)
    elif method == "denoise":
        out = _denoise(image)
    elif method == "night":
        out = _night(image)
    elif method == "sharpen":
        out = _sharpen(image)
    else:
        raise ValueError(f"Méthode inconnue : {method!r}. Choix : {METHODS}")
    return out
