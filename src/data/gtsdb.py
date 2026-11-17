"""Lecture de gt.txt (format : nom.ppm;xmin;ymin;xmax;ymax;classe).
Les annotations sont indexées par le NOM SANS EXTENSION (stem), car selon le miroir
les images sont en .ppm, .png ou .jpg."""
from collections import defaultdict
from pathlib import Path

IMG_EXTS = {".ppm", ".png", ".jpg", ".jpeg"}


def find_gt(root: str | Path) -> Path:
    """Cherche gt.txt dans root (récursif). Utile sur Kaggle (/kaggle/input/...)."""
    hits = sorted(Path(root).rglob("gt.txt"))
    if not hits:
        raise FileNotFoundError(f"gt.txt introuvable dans {root}. Vérifiez le chemin du dataset.")
    return hits[0]


def parse_gt(gt_path: str | Path) -> dict[str, list[tuple[int, int, int, int, int]]]:
    ann = defaultdict(list)
    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, x1, y1, x2, y2, c = line.split(";")[:6]
            ann[Path(name).stem].append((int(x1), int(y1), int(x2), int(y2), int(c)))
    return dict(ann)


def list_images(raw_dir: str | Path) -> dict[str, str]:
    """Retourne {stem: nom_de_fichier} pour les images au même niveau que gt.txt."""
    return {p.stem: p.name for p in sorted(Path(raw_dir).iterdir())
            if p.suffix.lower() in IMG_EXTS and p.stem.isdigit()}
