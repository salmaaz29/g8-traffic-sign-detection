# Page de cadrage — Projet G8

**Sujet** : application de chargement de scènes de conduite et comparaison d'approches de Deep Learning pour la détection de panneaux (prétraitement adapté), avec visualisation des boîtes englobantes et scores de confiance, pour analyser la robustesse (occultations, dégradations, météo).

**Groupe** : <noms> — **Langage** : Python — **Encadrant** : <nom>

## Choix techniques
| Élément | Choix | Justification |
|---|---|---|
| Dataset | GTSDB (900 images 1360×800, 43 classes) | Standard de la détection de panneaux, petit, accessible |
| Classes | 4 catégories (puis 43 classes si le temps le permet) | Peu de données : meilleure convergence |
| Modèles | YOLO11 (one-stage), Faster R-CNN (two-stage), RT-DETR (transformer) | Trois familles différentes |
| Interface | Streamlit | Rapide à développer, visualisation d'images |
| Entraînement | Google Colab / Kaggle (GPU) | Pas de GPU local requis |

## Protocole
- Split fixe (splits/split.json) : 600 images train/val (80/20 par blocs de 10 images consécutives), 300 test.
- Mêmes seuils pour tous : confiance 0.25, NMS IoU 0.5, résolution 1280.
- Métriques (même code pour tous, pycocotools) : mAP@0.5, mAP@0.5:0.95, précision, rappel, FPS, confiance moyenne.
- Chaque modèle évalué avec et sans prétraitement adapté.

## Dégradations (5 niveaux de sévérité chacune)
pluie, brouillard, nuit/faible luminosité, flou de mouvement, occultation (patchs), compression JPEG.

## Prétraitement
CLAHE (faible luminosité), dehazing (brouillard), débruitage / déflouttage (pluie, flou).

## Répartition
| | A | B |
|---|---|---|
| Jour 1 | Données, dépôt, vérification des annotations | Environnement, GPU, YOLO pré-entraîné, interface Detector |
| Jour 2 | Conversion YOLO/COCO + split | Wrapper YOLO + maquette Streamlit |

## Planning (4 semaines)
S1 données + prétraitement + 1er modèle · S2 entraînements + application · S3 robustesse + application complète · S4 analyse, rapport, démo.

## Questions pour l'encadrant
Rapport et/ou soutenance (date) ? Dataset libre ? Application exécutable en local ou démo ? 4 catégories acceptées ?
