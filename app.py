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
CLASS_FR = {"prohibitory": "Interdiction", "danger": "Danger", "mandatory": "Obligation", "other": "Autre"}
COLORS = {"prohibitory": (48, 48, 214), "danger": (0, 160, 245),       # BGR : rouge, ambre, bleu, vert
          "mandatory": (196, 112, 24), "other": (84, 160, 40)}
DEG_LABELS = {"rain": "Pluie", "fog": "Brouillard", "night": "Nuit", "blur": "Flou de bougé",
              "occlusion": "Occultation", "jpeg": "Compression JPEG"}
PP_LABELS = {"none": "Aucun", "clahe": "CLAHE", "dehaze": "Dehaze", "denoise": "Débruitage",
             "night": "Chaîne nuit", "sharpen": "Netteté", "gamma": "Gamma",
             "clahe_soft": "CLAHE doux", "night_light": "Nuit léger", "sharpen_soft": "Netteté douce"}

st.set_page_config(page_title="G8 · Détection de panneaux", page_icon=None, layout="wide")

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Overpass:wght@400;500;600;700;800&display=swap');
:root{--ink:#16202E;--mute:#5E6B7C;--line:#E1E5EB;--bg:#F5F6F8;--navy:#1B2A49;}
.stApp{background:var(--bg);}
.stApp p,.stApp label,.stApp h1,.stApp h2,.stApp h3,.stApp li,.stApp button,.stApp input,.stApp th,.stApp td,
.stApp [data-testid="stMarkdownContainer"],.stApp [data-testid="stWidgetLabel"],.stApp [data-baseweb="tab"]{font-family:'Overpass',system-ui,sans-serif;}
.block-container{padding-top:2.2rem;padding-bottom:3rem;max-width:1440px;}
#MainMenu,footer{visibility:hidden;}
section[data-testid="stSidebar"]{background:#fff;border-right:1px solid var(--line);}
section[data-testid="stSidebar"] .block-container{padding-top:1.6rem;}
.brand{font-weight:800;font-size:1.25rem;letter-spacing:-.01em;color:var(--ink);line-height:1.1;}
.brand small{display:block;font-weight:500;font-size:.8rem;color:var(--mute);margin-top:.25rem;letter-spacing:0;}
.sb{font-weight:700;font-size:.95rem;color:var(--ink);margin:1.3rem 0 .15rem;padding-top:.9rem;border-top:1px solid var(--line);}
.sbn{font-size:.8rem;color:var(--mute);margin:-.1rem 0 .5rem;}
.ttl{font-weight:800;font-size:2.15rem;letter-spacing:-.025em;color:var(--ink);line-height:1.1;margin:0;}
.sub{color:var(--mute);font-size:1.02rem;margin:.45rem 0 1.1rem;max-width:62ch;}
.chip{display:inline-block;background:#fff;border:1px solid var(--line);border-radius:999px;padding:.22rem .8rem;margin:0 .4rem .4rem 0;font-size:.86rem;color:var(--ink);}
.chip i{font-style:normal;color:var(--mute);margin-right:.35rem;}
.legend{display:flex;flex-wrap:wrap;gap:.35rem .9rem;margin-top:.3rem;}
.legend span{font-size:.84rem;color:var(--ink);display:inline-flex;align-items:center;gap:.4rem;}
.legend em{width:.7rem;height:.7rem;border-radius:2px;display:inline-block;}
div[data-testid="stVerticalBlockBorderWrapper"]{background:#fff;border-color:var(--line)!important;border-radius:12px!important;}
.mh{display:flex;align-items:center;gap:.6rem;margin-bottom:.55rem;flex-wrap:wrap;}
.mh .bar{width:.28rem;height:1.35rem;border-radius:2px;display:inline-block;}
.mh .mn{font-weight:700;font-size:1.12rem;color:var(--ink);}
.mh .bd{font-size:.78rem;font-weight:600;color:var(--navy);border:1px solid var(--navy);border-radius:999px;padding:.02rem .55rem;}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:.6rem;margin-top:.7rem;padding-top:.7rem;border-top:1px solid var(--line);}
.stats span{display:block;font-size:.76rem;color:var(--mute);}
.stats b{font-size:1.4rem;font-weight:700;color:var(--ink);font-variant-numeric:tabular-nums;line-height:1.15;}
.stats small{font-size:.8rem;font-weight:500;color:var(--mute);margin-left:.15rem;}
.empty{border:1.5px dashed #C4CBD6;border-radius:12px;padding:2.6rem 1.5rem;text-align:center;background:#fff;}
.empty b{display:block;font-size:1.15rem;color:var(--ink);margin-bottom:.3rem;}
.empty span{color:var(--mute);}
[data-testid="stImage"] img{border-radius:8px;width:100%;}
h3{font-weight:700!important;letter-spacing:-.01em;color:var(--ink);}
</style>"""


def H(html: str):
    st.markdown(html, unsafe_allow_html=True)


def hexcol(bgr):
    return "#%02x%02x%02x" % (bgr[2], bgr[1], bgr[0])


H(CSS)


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


@st.cache_resource
def compute_device() -> str:
    try:
        import torch
        return "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    except Exception:
        return "CPU"


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
        col = COLORS.get(d.get("label", ""), (120, 120, 120))
        cv2.rectangle(out, (x1, y1), (x2, y2), col, th)
        txt = f'{CLASS_FR.get(d.get("label", ""), d["class_id"])} {d["score"]:.2f}'
        (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        y0 = max(tht + 6, y1)
        cv2.rectangle(out, (x1, y0 - tht - 6), (x1 + tw + 6, y0), col, -1)
        cv2.putText(out, txt, (x1 + 3, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def crops(img, dets, size=96, pad=0.35, limit=10):
    tiles, caps = [], []
    h, w = img.shape[:2]
    for d in sorted(dets, key=lambda d: -d["score"])[:limit]:
        x1, y1, x2, y2 = d["box"]
        px, py = (x2 - x1) * pad, (y2 - y1) * pad
        a, b = max(0, int(x1 - px)), max(0, int(y1 - py))
        c, e = min(w, int(x2 + px)), min(h, int(y2 + py))
        if c - a < 2 or e - b < 2:
            continue
        t = cv2.resize(img[b:e, a:c], (size, int(size * (e - b) / (c - a))), interpolation=cv2.INTER_CUBIC)
        tiles.append(cv2.cvtColor(t, cv2.COLOR_BGR2RGB))
        caps.append(f'{CLASS_FR.get(d.get("label", ""), d["class_id"])} {d["score"]:.2f}')
    return tiles, caps


def infer(det, img, conf):
    t0 = time.perf_counter()
    dets = det.predict(img, conf=conf)
    return dets, 1000 * (time.perf_counter() - t0)


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def settings_chips(kind, level, pp, conf):
    cond = "Image d'origine" if kind == "none" or level == 0 else f"{DEG_LABELS[kind]}, niveau {level}"
    H(f'<span class="chip"><i>Conditions</i>{cond}</span>'
      f'<span class="chip"><i>Prétraitement</i>{PP_LABELS.get(pp, pp)}</span>'
      f'<span class="chip"><i>Seuil de confiance</i>{conf:.2f}</span>'
      f'<span class="chip"><i>Calcul</i>{compute_device()}</span>')


def show_comparison(img, name, detectors, kind, level, pp, conf):
    proc = make_input(img, name, kind, level, pp)
    settings_chips(kind, level, pp, conf)

    results = {}
    with st.spinner("Analyse en cours…"):
        for label, det in detectors.items():
            dets, ms = infer(det, proc, conf)
            results[label] = {"dets": dets, "ms": ms,
                              "conf": float(np.mean([d["score"] for d in dets])) if dets else 0.0}
    fastest = min(results, key=lambda k: results[k]["ms"]) if len(results) > 1 else None
    surest = max((k for k in results if results[k]["dets"]), key=lambda k: results[k]["conf"], default=None) \
        if len(results) > 1 else None

    cols = st.columns(len(results))
    for col, (label, r) in zip(cols, results.items()):
        with col.container(border=True):
            badges = ""
            if label == fastest:
                badges += '<span class="bd">Le plus rapide</span>'
            if label == surest:
                badges += '<span class="bd">Le plus confiant</span>'
            H(f'<div class="mh"><span class="bar" style="background:{MODEL_COLORS[label]}"></span>'
              f'<span class="mn">{label}</span>{badges}</div>')
            st.image(rgb(draw(proc, r["dets"])))
            H(f'<div class="stats"><div><span>Détections</span><b>{len(r["dets"])}</b></div>'
              f'<div><span>Confiance moyenne</span><b>{r["conf"]:.2f}</b></div>'
              f'<div><span>Temps</span><b>{r["ms"]:.0f}<small>ms</small></b></div>'
              f'<div><span>Cadence</span><b>{1000 / r["ms"] if r["ms"] > 0 else 0:.1f}<small>FPS</small></b></div></div>')

    st.write("")
    t1, t2, t3, t4 = st.tabs(["Comparatif", "Avant / après", "Panneaux détectés", "Détail des détections"])
    with t1:
        df = pd.DataFrame([{"Modèle": k, "Détections": len(r["dets"]), "Confiance moyenne": round(r["conf"], 3),
                            "Temps (ms)": round(r["ms"], 1),
                            "FPS": round(1000 / r["ms"], 1) if r["ms"] > 0 else 0.0}
                           for k, r in results.items()])
        st.dataframe(df, hide_index=True, column_config={
            "Confiance moyenne": st.column_config.ProgressColumn("Confiance moyenne", min_value=0.0,
                                                                 max_value=1.0, format="%.2f")})
        st.caption("Les temps dépendent de la machine : ne les compare pas à ceux mesurés sur Kaggle (GPU T4).")
    with t2:
        c1, c2 = st.columns(2)
        c1.image(rgb(img), caption="Image d'origine")
        c2.image(rgb(proc), caption="Image envoyée aux modèles (après dégradation et prétraitement)")
    with t3:
        any_crop = False
        for label, r in results.items():
            tiles, caps = crops(proc, r["dets"])
            if tiles:
                any_crop = True
                st.markdown(f"**{label}**")
                st.image(tiles, caption=caps, width=110)
        if not any_crop:
            st.info("Aucun panneau détecté avec ces réglages. Baisse le seuil de confiance ou réduis la dégradation.")
    with t4:
        for label, r in results.items():
            st.markdown(f"**{label}**")
            if not r["dets"]:
                st.caption("Aucune détection.")
                continue
            st.dataframe(pd.DataFrame([{"Classe": CLASS_FR.get(d.get("label", ""), d["class_id"]),
                                        "Confiance": round(d["score"], 3),
                                        "x1": round(d["box"][0]), "y1": round(d["box"][1]),
                                        "x2": round(d["box"][2]), "y2": round(d["box"][3])}
                                       for d in sorted(r["dets"], key=lambda d: -d["score"])]),
                         hide_index=True,
                         column_config={"Confiance": st.column_config.ProgressColumn(
                             "Confiance", min_value=0.0, max_value=1.0, format="%.2f")})


MODEL_COLORS = {"YOLO11": "#2F6FED", "RT-DETR": "#8B5CF6", "Faster R-CNN": "#E08A00"}

# ------------------------------------------------------------------------------- barre latérale
sb = st.sidebar
with sb:
    H('<div class="brand">Détection de panneaux<small>Comparaison de modèles · Groupe 8</small></div>')

    H('<div class="sb">Modèles</div><div class="sbn">Choisis ceux à comparer côte à côte.</div>')
    with st.expander("Dossier des poids"):
        weights_dir = Path(st.text_input("Chemin", str(ROOT / "weights"), label_visibility="collapsed"))
    available = {}
    for label, (key, fname) in MODELS.items():
        if (weights_dir / fname).exists():
            available[label] = (key, str(weights_dir / fname))
        else:
            st.warning(f"{fname} introuvable dans {weights_dir}")
    chosen = st.multiselect("Modèles", list(available), default=list(available), label_visibility="collapsed")

    H('<div class="sb">Conditions de conduite</div><div class="sbn">Dégrade l\'image avant la détection.</div>')
    kind = st.selectbox("Dégradation", ["none"] + KINDS,
                        format_func=lambda k: "Aucune" if k == "none" else DEG_LABELS[k])
    level = st.slider("Sévérité", 0, 5, 3 if kind != "none" else 0, disabled=kind == "none")

    H('<div class="sb">Prétraitement</div><div class="sbn">Corrige l\'image dégradée avant les modèles.</div>')
    pp = st.selectbox("Méthode", ["none"] + METHODS, format_func=lambda m: PP_LABELS.get(m, m),
                      label_visibility="collapsed")

    H('<div class="sb">Détection</div>')
    conf = st.slider("Seuil de confiance", 0.05, 0.95, 0.25, 0.05)

    H('<div class="sb">Classes de panneaux</div><div class="legend">' +
      "".join(f'<span><em style="background:{hexcol(COLORS[c])}"></em>{CLASS_FR[c]}</span>' for c in COLORS) +
      '</div>')

# ------------------------------------------------------------------------------------- en-tête
H('<div class="ttl">Comparer des détecteurs de panneaux</div>'
  '<div class="sub">Charge une scène de conduite, dégrade-la (pluie, nuit, flou…) et vois comment '
  'chaque modèle réagit : boîtes, scores de confiance et vitesse.</div>')

if not chosen:
    H('<div class="empty"><b>Aucun modèle sélectionné</b>'
      '<span>Choisis au moins un modèle dans la barre latérale. Les poids doivent être dans le dossier weights/.</span></div>')
    st.stop()
detectors = {label: get_detector(*available[label]) for label in chosen}

source = st.radio("Source", ["Image", "Vidéo de conduite"], horizontal=True, label_visibility="collapsed")

# ------------------------------------------------------------------------------------ image
if source == "Image":
    samples = sorted((ROOT / "data/processed/gtsdb_4cls/images/test").glob("*.png"))
    if samples:
        cu, cs = st.columns([3, 2])
    else:
        cu, cs = st.container(), None
    up = cu.file_uploader("Image de scène de conduite", type=["png", "jpg", "jpeg", "bmp"])
    sample = cs.selectbox("Ou une image d'exemple (GTSDB, jeu de test)", [None] + samples,
                          format_func=lambda p: "Aucune" if p is None else p.name) if cs is not None else None
    img, name = None, "image"
    if up is not None:
        img = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        name = up.name
    elif sample is not None:
        img, name = cv2.imread(str(sample)), sample.name
    if img is None:
        H('<div class="empty"><b>Aucune image chargée</b>'
          '<span>Dépose une image ci-dessus pour lancer la comparaison.</span></div>')
    else:
        show_comparison(img, name, detectors, kind, level, pp, conf)

# ------------------------------------------------------------------------------------ vidéo
else:
    vid = st.file_uploader("Vidéo de conduite", type=["mp4", "avi", "mov", "mkv"])
    if vid is None:
        H('<div class="empty"><b>Aucune vidéo chargée</b>'
          '<span>Dépose une vidéo ci-dessus, puis choisis une image à analyser ou lance l\'analyse complète.</span></div>')
        st.stop()
    tmp = Path(tempfile.gettempdir()) / f"g8_{vid.name}"
    tmp.write_bytes(vid.getvalue())
    cap = cv2.VideoCapture(str(tmp))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if n <= 0:
        st.error("Vidéo illisible. Essaie un fichier mp4 (H.264).")
        st.stop()
    idx = st.slider("Image de la vidéo", 0, n - 1, 0, format="%d")
    st.caption(f"Image {idx + 1} sur {n}, à {idx / fps:.1f} s")
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        st.error("Impossible de lire cette image.")
        st.stop()
    show_comparison(frame, f"{vid.name}#{idx}", detectors, kind, level, pp, conf)

    st.write("")
    with st.expander("Analyser toute la vidéo"):
        c1, c2 = st.columns(2)
        stride = c1.number_input("Analyser une image sur", 1, 300, max(1, int(fps)))
        max_frames = c2.number_input("Nombre maximum d'images", 5, 300, 60)
        if st.button("Lancer l'analyse", type="primary"):
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
            st.markdown("**Nombre de détections au fil de la vidéo** (axe : secondes)")
            st.line_chart(pd.DataFrame({m: [s[0] for s in v] for m, v in stats.items()}, index=t))
            st.markdown("**Confiance moyenne au fil de la vidéo**")
            st.line_chart(pd.DataFrame({m: [s[1] for s in v] for m, v in stats.items()}, index=t))
            st.dataframe(pd.DataFrame([{"Modèle": m, "Détections par image": round(float(np.mean([s[0] for s in v])), 2),
                                        "Confiance moyenne": round(float(np.nanmean([s[1] for s in v])), 3),
                                        "Temps moyen (ms)": round(float(np.mean([s[2] for s in v])), 1)}
                                       for m, v in stats.items()]), hide_index=True)
    cap.release()