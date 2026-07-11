# 🧠 BrainScanAI — Classification d'IRM cérébrales par approche semi-supervisée

> **Projet Data Science — CurelyticsIA**  
> Détection automatique de tumeurs cérébrales sur IRM avec un budget d'annotation limité

---

## Problématique

Comment entraîner un modèle fiable de détection de tumeurs cérébrales quand on ne dispose que de **100 images labellisées** sur 1 506, pour un budget de **300 €** ?

---

## Structure du projet

```
BrainScanAI/
│
├── 01_exploration_dataset.ipynb     # Exploration et nettoyage du dataset (EDA)
├── 02_pipeline_radimagenet.ipynb    # Pipeline complet (features + clustering + classification)
└── README.md
```

---

## Dataset

| Caractéristique | Valeur |
|---|---|
| Nombre total d'images | 1 506 |
| Images labellisées | 100 (50 normal / 50 cancer) |
| Images non étiquetées | 1 406 |
| Format | JPEG 512×512 |
| Type | IRM cérébrales |

> ⚠️ **Point de vigilance détecté** : 32 images du pool labellisé étaient dupliquées dans le pool non étiqueté → fuite de données potentielle corrigée avant toute modélisation.

---

##  Pipeline

```
Dataset brut (1 506 images)
        │
        ▼
┌─────────────────────────┐
│  1. Exploration & EDA   │  → Nettoyage, dédoublonnage, analyse visuelle
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  2. Extraction features │  → RadImageNet (ResNet50, 2048 dim) + HOG
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  3. Clustering          │  → K-Means + DBSCAN → pseudo-labels
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  4. Classification CNN  │  → 3 modèles comparés
└─────────────────────────┘
        │
        ▼
   Prédictions sur
  1 374 images non
     étiquetées
```

---

##  Extraction de features — RadImageNet

J'ai utilisé **RadImageNet**, un ResNet50 pré-entraîné sur **1.35 million d'images médicales** (IRM, scanner CT, échographies), comparé aux features HOG classiques.

| Méthode | Dimension | Pertinence médicale |
|---|---|---|
| HOG (classique) | 1 568 | Faible |
| RadImageNet (ResNet50) | 2 048 | ⭐ Élevée |

---

## Résultats du Clustering

| Features | Algo | ARI ↑ | Commentaire |
|---|---|---|---|
| RadImageNet | K-Means | -0.009 | Structure non alignée avec les labels |
| RadImageNet | DBSCAN | 0.107 | 2 clusters à eps restreint |
| **HOG** | **K-Means** | **0.186 ★** | **Meilleur ARI → retenu pour pseudo-labels** |
| HOG | DBSCAN | ~0.017 | Espace trop dense |

> L'ARI est calculé uniquement sur les 100 images labellisées — les seules dont on connaît le vrai label.

---

## Comparaison des 3 modèles

| Modèle | Données d'entraînement | Accuracy | Rappel cancer ★ | F1 |
|---|---|---|---|---|
| A — Faible seul | 1 374 pseudo-labels | 0.633 | 0.467 ✗ | 0.560 |
| **B — Semi-supervisé** | **Faible → Fort (finetune)** | **0.733** | **0.800 ✓** | **0.750 ✓** |
| C — Fort seul | 70 images (vrais labels) | 0.500 | 1.000 | 0.667 |

> ★ **Le rappel sur la classe cancer est la métrique prioritaire** : un faux négatif (cancer non détecté) est bien plus grave qu'un faux positif dans un contexte de dépistage médical.

**→ Le modèle B (semi-supervisé) est le meilleur compromis** : il exploite les 1 374 images non étiquetées pour améliorer ses performances par rapport au supervisé seul.

---

## Prédictions sur le pool non étiqueté

Après validation du modèle B, prédiction sur les 1 374 images sans label :

| Label prédit | Nombre d'images |
|---|---|
| Cancer | 1 027 |
| Normal | 347 |
| Confiance moyenne | 0.680 |
| Images à faible confiance (< 0.70) | 883 |

> Les 883 images à faible confiance sont à prioriser pour une validation médicale humaine.

---

## Technologies utilisées

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.x-green)

- **PyTorch / Torchvision** — Deep learning & CNN
- **RadImageNet** — Backbone médical pré-entraîné (via HuggingFace Hub)
- **scikit-learn** — Clustering, métriques, PCA, t-SNE
- **Pandas / NumPy / Matplotlib** — Manipulation de données et visualisation
- **scikit-image (HOG)** — Features classiques

---

## Passage à l'échelle

Budget : **5 000 €** pour **4 millions d'images**

Coût autorisé par image : `5 000 € ÷ 4 000 000 = 0.00125 € / image`

**Stratégie retenue : Active Learning + inférence GPU**

1. Inférence GPU sur les 4M images (~100 €)
2. Annotation humaine ciblée sur les ~50K cas incertains (~4 000 €)
3. QA médicale sur les cas douteux (~700 €)
4. Infrastructure MLOps (~200 €)

**Total estimé : ≈ 5 000 €** ✓

---

## ⚠️ Limites

- Jeu de test de seulement **30 images** — résultats indicatifs, à confirmer sur un jeu plus large
- Dataset hétérogène : mélange de plans de coupe (axial/sagittal/coronal) et possible mélange IRM/scanner
- Pas de métadonnées DICOM disponibles pour identifier les modalités

---

## 👤 Auteur

**Fatima ADDA-REZIG**  
Data Science — CurelyticsIA
