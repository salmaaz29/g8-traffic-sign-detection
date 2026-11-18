"""Application Streamlit G8 : comparaison de détecteurs de panneaux (YOLO11, RT-DETR, Faster R-CNN)
sur des scènes de conduite, avec dégradations (pluie, brouillard, nuit, flou, occultation, JPEG)
et prétraitement.

Lancer :  streamlit run app.py
Poids attendus dans weights/ : yolo11_best.pt, rtdetr_best.pt, frcnn_best.pth
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.degradations.degrade import KINDS, make_seed          # noqa: E402
from src.degradations.degrade import apply as degrade_apply    # noqa: E402
from src.preprocessing.preprocess import METHODS, preprocess   # noqa: E402

MODELS = {"YOLO11": ("yolo11", "yolo11_best.pt"),
          "RT-DETR": ("rtdetr", "rtdetr_best.pt"),
          "Faster R-CNN": ("frcnn", "frcnn_best.pth")}
COLORS = {"prohibitory": (40, 40, 230), "danger": (0, 165, 255),      # BGR
          "mandatory": (230, 120, 20), "other": (60, 180, 60)}
DEG_LABELS = {"rain": "Pluie", "fog": "Brouillard", "night": "Nuit", "blur": "Flou de bougé",
              "occlusion": "Occultation", "jpeg": "Compression JPEG"}

# compatibilité : anciennes versions de Streamlit -> use_container_width, récentes -> width="stretch"
_v = tuple(int(x) for x in st.__version__.split(".")[:2])
STRETCH = {"width": "stretch"} if _v >= (1, 52) else {"use_container_width": True}

st.set_page_config(page_title="G8 - Détection de panneaux", layout="wide")


# ------------------------------------------------------------------------------ modèles
@st.cache_resource(show_spinner="Chargement du modèle…")
def get_detector(key: str, path: str):
    if key in ("yolo11", "rtdetr"):
        from src.models.yolo import UltralyticsDetector
        det = UltralyticsDetector(path, name=key, kind="rtdetr" if key == "rtdetr" else "yolo")
    else:
        from src.models.frcnn import FasterRCNNDetector
        det = FasterRCNNDetector(path)
    det.predict(np.zeros((800, 1360, 3), np.uint8), conf=0.25)       # échauffement (non chronométré)
    return det


# ----------------------------------------------------------------------------- traitement
def make_input(img, name, kind, level, pp):
    """Dégradation (reproductible) puis prétraitement -> image envoyée aux modèles."""
    out = img
    if kind != "none" and level > 0:
        out = degrade_apply(img, kind, level, seed=make_seed(name, kind, level))
    if pp != "none":
        out = preprocess(out, pp)
    return out


def draw(img, dets):
    out = img.copy()
    h, w = out.shape[:2]
    th, fs = max(2, w // 700), max(0.5, w / 1900)
    for d in dets:
        x1, y1, x2, y2 = (int(round(v)) for v in d["box"])
        col = COLORS.get(d.get("label", ""), (200, 200, 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), col, th)
        txt = f'{d.get("label", d["class_id"])} {d["score"]:.2f}'
        (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        y0 = max(tht + 4, y1)
        cv2.rectangle(out, (x1, y0 - tht - 4), (x1 + tw + 4, y0), col, -1)
        cv2.putText(out, txt, (x1 + 2, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def infer(det, img, conf):
    t0 = time.perf_counter()
    dets = det.predict(img, conf=conf)
    return dets, 1000 * (time.perf_counter() - t0)


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def show_comparison(img, name, detectors, kind, level, pp, conf):
    proc = make_input(img, name, kind, level, pp)
    c1, c2 = st.columns(2)
    c1.image(rgb(img), caption="Image d'origine", **STRETCH)
    c2.image(rgb(proc), caption=f"Image envoyée aux modèles (dégradation : {DEG_LABELS.get(kind, 'aucune')}"
                                f"{f' niv. {level}' if kind != 'none' and level else ''} | prétraitement : {pp})",
             **STRETCH)
    st.subheader("Détections")
    cols = st.columns(len(detectors))
    rows, details = [], {}
    for col, (label, det) in zip(cols, detectors.items()):
        dets, ms = infer(det, proc, conf)
        col.image(rgb(draw(proc, dets)), caption=f"{label} : {len(dets)} détection(s)", **STRETCH)
        rows.append({"Modèle": label, "Détections": len(dets),
                     "Confiance moyenne": round(float(np.mean([d["score"] for d in dets])), 3) if dets else 0.0,
                     "Temps (ms)": round(ms, 1), "FPS": round(1000 / ms, 1) if ms > 0 else 0.0})
        details[label] = dets
    st.dataframe(pd.DataFrame(rows), **STRETCH, hide_index=True)
    with st.expander("Détail des détections (classe, score, boîte)"):
        for label, dets in details.items():
            st.markdown(f"**{label}**")
            st.dataframe(pd.DataFrame([{"classe": d.get("label", d["class_id"]), "score": round(d["score"], 3),
                                        "x1": round(d["box"][0]), "y1": round(d["box"][1]),
                                        "x2": round(d["box"][2]), "y2": round(d["box"][3])} for d in dets]),
                         **STRETCH, hide_index=True)


# ------------------------------------------------------------------------------- barre latérale
st.title("Détection de panneaux : comparaison de modèles de deep learning")
sb = st.sidebar
sb.header("Réglages")
weights_dir = Path(sb.text_input("Dossier des poids", str(ROOT / "weights")))

available = {}
for label, (key, fname) in MODELS.items():
    if (weights_dir / fname).exists():
        available[label] = (key, str(weights_dir / fname))
    else:
        sb.warning(f"{fname} introuvable dans {weights_dir}")
chosen = sb.multiselect("Modèles à comparer", list(available), default=list(available))

kind = sb.selectbox("Dégradation", ["none"] + KINDS,
                    format_func=lambda k: "Aucune" if k == "none" else DEG_LABELS[k])
level = sb.slider("Sévérité", 0, 5, 3 if kind != "none" else 0, disabled=kind == "none")
pp = sb.selectbox("Prétraitement", ["none"] + METHODS,
                  format_func=lambda m: "Aucun" if m == "none" else m)
conf = sb.slider("Seuil de confiance", 0.05, 0.95, 0.25, 0.05)
source = sb.radio("Source", ["Image", "Vidéo / scène de conduite"])

if not chosen:
    st.info("Choisis au moins un modèle dans la barre latérale (les poids doivent être dans weights/).")
    st.stop()
detectors = {label: get_detector(*available[label]) for label in chosen}

# ------------------------------------------------------------------------------------ image
if source == "Image":
    up = st.file_uploader("Charger une image de scène de conduite", type=["png", "jpg", "jpeg", "bmp"])
    samples = sorted((ROOT / "data/processed/gtsdb_4cls/images/test").glob("*.png"))
    sample = st.selectbox("…ou une image d'exemple (GTSDB test)", [None] + samples,
                          format_func=lambda p: "—" if p is None else p.name) if samples else None
    img, name = None, "image"
    if up is not None:
        img = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        name = up.name
    elif sample is not None:
        img, name = cv2.imread(str(sample)), sample.name
    if img is None:
        st.info("Charge une image pour commencer.")
    else:
        show_comparison(img, name, detectors, kind, level, pp, conf)

# ------------------------------------------------------------------------------------ vidéo
else:
    vid = st.file_uploader("Charger une vidéo de conduite", type=["mp4", "avi", "mov", "mkv"])
    if vid is None:
        st.info("Charge une vidéo pour commencer.")
        st.stop()
    tmp = Path(tempfile.gettempdir()) / f"g8_{vid.name}"
    tmp.write_bytes(vid.getvalue())
    cap = cv2.VideoCapture(str(tmp))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if n <= 0:
        st.error("Vidéo illisible.")
        st.stop()
    idx = st.slider("Image de la vidéo", 0, n - 1, 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        st.error("Image illisible.")
        st.stop()
    show_comparison(frame, f"{vid.name}#{idx}", detectors, kind, level, pp, conf)

    with st.expander("Analyser toute la vidéo (1 image sur N)"):
        stride = st.number_input("Une image sur", 1, 300, max(1, int(fps)))
        max_frames = st.number_input("Nombre maximum d'images analysées", 5, 300, 60)
        if st.button("Lancer l'analyse"):
            idxs = list(range(0, n, int(stride)))[:int(max_frames)]
            bar, stats = st.progress(0.0), {m: [] for m in detectors}
            for k, i in enumerate(idxs):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, fr = cap.read()
                if not ok:
                    continue
                proc = make_input(fr, f"{vid.name}#{i}", kind, level, pp)
                for label, det in detectors.items():
                    dets, ms = infer(det, proc, conf)
                    stats[label].append((len(dets), np.mean([d["score"] for d in dets]) if dets else np.nan, ms))
                bar.progress((k + 1) / len(idxs))
            t = [round(i / fps, 1) for i in idxs[:len(next(iter(stats.values())))]]
            st.markdown("**Nombre de détections par image analysée (axe : secondes)**")
            st.line_chart(pd.DataFrame({m: [s[0] for s in v] for m, v in stats.items()}, index=t))
            st.markdown("**Confiance moyenne**")
            st.line_chart(pd.DataFrame({m: [s[1] for s in v] for m, v in stats.items()}, index=t))
            st.dataframe(pd.DataFrame([{"Modèle": m, "Détections / image": round(np.mean([s[0] for s in v]), 2),
                                        "Confiance moyenne": round(float(np.nanmean([s[1] for s in v])), 3),
                                        "Temps moyen (ms)": round(np.mean([s[2] for s in v]), 1)}
                                       for m, v in stats.items()]), **STRETCH, hide_index=True)
    cap.release()