"""
Figures et tableaux de comparaison à partir des CSV produits par run_robustness.py.

Ne dépend PAS d'OpenCV : peut se lancer sur un PC ou sur Kaggle.

Usage :
    python -m src.evaluation.plots --results-dir results --out-dir reports/comparison

Sorties (dans --out-dir) :
    robustness_<métrique>.png      : une courbe par modèle, un panneau par dégradation
    summary_<métrique>.csv         : image propre, moyenne sur les niveaux 1-5, niveaux 3 et 5
    preprocess_<métrique>_<modèle>.png et summary_preprocess_<métrique>.csv
                                     (seulement si plusieurs prétraitements sont présents)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REQUIRED = {"model", "preprocess", "degradation", "level", "map50", "map50_95", "precision", "recall"}
KINDS = ["rain", "fog", "night", "blur", "occlusion", "jpeg"]
KIND_FR = {"rain": "Pluie", "fog": "Brouillard", "night": "Nuit", "blur": "Flou de bougé",
           "occlusion": "Occultation", "jpeg": "Compression JPEG"}
LABELS = {"yolo11": "YOLO11", "rtdetr": "RT-DETR", "frcnn": "Faster R-CNN"}
COLORS = {"yolo11": "#1f77b4", "rtdetr": "#ff7f0e", "frcnn": "#2ca02c"}
METRIC_FR = {"map50": "mAP@0.5", "map50_95": "mAP@0.5:0.95", "precision": "Précision", "recall": "Rappel"}
NUMERIC = ["level", "map50", "map50_95", "precision", "recall"]


def load_results(root) -> pd.DataFrame:
    """Rassemble tous les CSV de robustesse trouvés sous `root` (les autres CSV sont ignorés)."""
    frames = []
    for p in sorted(Path(root).rglob("*.csv")):
        try:
            df = pd.read_csv(p, keep_default_na=False, na_values=[""])
        except Exception:
            continue
        if REQUIRED <= set(df.columns):
            df["source"] = str(p)
            frames.append(df)
    if not frames:
        raise SystemExit(f"Aucun CSV de robustesse trouvé sous {root}")
    df = pd.concat(frames, ignore_index=True)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["level"] = df["level"].astype(int)
    df["preprocess"] = df["preprocess"].astype(str)
    return df.drop_duplicates(["model", "preprocess", "degradation", "level"], keep="last")


def _curve(df, model, pre, kind, metric):
    """(niveaux, valeurs) avec le niveau 0 = image propre du même modèle et prétraitement."""
    base = df[(df.model == model) & (df.preprocess == pre)]
    clean = base[base.degradation == "clean"][metric]
    sub = base[base.degradation == kind].sort_values("level")
    xs = list(sub.level)
    ys = list(sub[metric])
    if len(clean):
        xs, ys = [0] + xs, [float(clean.iloc[0])] + ys
    return xs, ys


def plot_models(df, pre, metric, out):
    models = [m for m in COLORS if m in set(df.model)] + sorted(set(df.model) - set(COLORS))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for ax, kind in zip(axes.ravel(), KINDS):
        for m in models:
            xs, ys = _curve(df, m, pre, kind, metric)
            if xs:
                ax.plot(xs, ys, marker="o", color=COLORS.get(m), label=LABELS.get(m, m))
        ax.set_title(KIND_FR[kind]); ax.set_xlabel("niveau de sévérité (0 = image propre)")
        ax.set_xticks(range(6)); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel(METRIC_FR[metric]); axes[1, 0].set_ylabel(METRIC_FR[metric])
    axes[0, 0].legend(loc="lower left")
    fig.suptitle(f"Robustesse : {METRIC_FR[metric]} (prétraitement : {pre})")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def summary_models(df, pre, metric):
    rows = []
    for m in sorted(set(df.model)):
        base = df[(df.model == m) & (df.preprocess == pre)]
        clean = base[base.degradation == "clean"][metric]
        rows.append({"modèle": LABELS.get(m, m), "dégradation": "image propre", "moyenne niv. 1-5": np.nan,
                     "niveau 3": np.nan, "niveau 5": np.nan, "propre": float(clean.iloc[0]) if len(clean) else np.nan,
                     "n_niveaux": 0})
        for kind in KINDS:
            s = base[base.degradation == kind].set_index("level")[metric]
            rows.append({"modèle": LABELS.get(m, m), "dégradation": KIND_FR[kind],
                         "moyenne niv. 1-5": s.mean() if len(s) else np.nan,
                         "niveau 3": s.get(3, np.nan), "niveau 5": s.get(5, np.nan),
                         "propre": np.nan, "n_niveaux": len(s)})
    return pd.DataFrame(rows).round(3)


def plot_preprocess(df, model, metric, out):
    pres = sorted(set(df[df.model == model].preprocess), key=lambda p: (p != "none", p))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for ax, kind in zip(axes.ravel(), KINDS):
        for p in pres:
            xs, ys = _curve(df, model, p, kind, metric)
            if xs:
                ax.plot(xs, ys, marker="o", linestyle="--" if p == "none" else "-", label=p)
        ax.set_title(KIND_FR[kind]); ax.set_xticks(range(6)); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
        ax.set_xlabel("niveau de sévérité (0 = image propre)")
    axes[0, 0].legend(loc="lower left", title="prétraitement")
    fig.suptitle(f"{LABELS.get(model, model)} : effet du prétraitement sur {METRIC_FR[metric]}")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def summary_preprocess(df, metric):
    """Gain moyen (niveaux 1-5) d'un prétraitement par rapport à « none »."""
    rows = []
    for m in sorted(set(df.model)):
        base = df[(df.model == m) & (df.degradation != "clean")]
        ref = base[base.preprocess == "none"].set_index(["degradation", "level"])[metric]
        for p in sorted(set(base.preprocess) - {"none"}):
            cur = base[base.preprocess == p].set_index(["degradation", "level"])[metric]
            gain = (cur - ref).dropna()
            for kind in KINDS:
                g = gain[gain.index.get_level_values(0) == kind]
                rows.append({"modèle": LABELS.get(m, m), "prétraitement": p, "dégradation": KIND_FR[kind],
                             "gain moyen": g.mean() if len(g) else np.nan, "n": len(g)})
    return pd.DataFrame(rows).round(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="reports/comparison")
    ap.add_argument("--metrics", nargs="+", default=["map50", "map50_95", "recall"], choices=list(METRIC_FR))
    ap.add_argument("--preprocess", default="none", help="prétraitement affiché dans la comparaison des modèles")
    a = ap.parse_args()

    df = load_results(a.results_dir)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    print("Modèles :", sorted(set(df.model)), "| prétraitements :", sorted(set(df.preprocess)))
    print("Sources :", sorted(set(df.source)))

    for metric in a.metrics:
        plot_models(df, a.preprocess, metric, out / f"robustness_{metric}.png")
        s = summary_models(df, a.preprocess, metric)
        s.to_csv(out / f"summary_{metric}.csv", index=False)
        print(f"\n=== {METRIC_FR[metric]} ===\n{s.to_string(index=False)}")
        if df.preprocess.nunique() > 1:
            for m in sorted(set(df.model)):
                plot_preprocess(df, m, metric, out / f"preprocess_{metric}_{m}.png")
            sp = summary_preprocess(df, metric)
            sp.to_csv(out / f"summary_preprocess_{metric}.csv", index=False)
            print(f"\n--- gain du prétraitement ({METRIC_FR[metric]}) ---\n{sp.to_string(index=False)}")
    print("\nFigures et tableaux dans :", out)


if __name__ == "__main__":
    main()
