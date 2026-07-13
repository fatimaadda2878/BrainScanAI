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
├── 01_exploration_dataset.ipynb       # Exploration et nettoyage du dataset (EDA)
├── 02_pipeline_radimagenet.ipynb      # Pipeline complet (features + clustering + classification)
├── requirements.txt                   # Dépendances Python
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

> **Point de vigilance détecté** : 32 images du pool labellisé étaient dupliquées dans le pool non étiqueté → fuite de données potentielle corrigée avant toute modélisation.

> **Dataset** : disponible sur demande auprès de CurelyticsIA. Placer le fichier `mri_dataset_brain_cancer_oc.zip` à la racine du projet avant d'exécuter les notebooks.

---

## Installation

```bash
git clone (https://github.com/fatimaadda2878/BrainScanAI)
cd BrainScanAI
pip install -r requirements.txt
```

---

## Pipeline

```
Dataset brut (1 506 images)
        │
        ▼
┌─────────────────────────┐
│  1. Exploration & EDA   │  → Nettoyage, dédoublonnage, CLAHE, analyse visuelle
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  2. Extraction features │  → RadImageNet (ResNet50, 2048 dim) + HOG
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  3. Clustering          │  → K-Means + DBSCAN → pseudo-labels filtrés
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  4. Classification CNN  │  → 3 modèles comparés + early stopping (validation)
└─────────────────────────┘
        │
        ▼
   Prédictions sur
  1 374 images non
     étiquetées
```

---

## Extraction de features — RadImageNet

J'ai utilisé **RadImageNet**, un ResNet50 pré-entraîné sur **1.35 million d'images médicales** (IRM, scanner CT, échographies), comparé aux features HOG classiques. Une égalisation des histogrammes **CLAHE** est appliquée en amont pour améliorer le contraste des images.

| Méthode | Dimension | Pertinence médicale |
|---|---|---|
| HOG (classique) | 1 568 | Faible |
| RadImageNet (ResNet50) | 2 048 | ⭐ Élevée |

> Le même preprocessing (CLAHE + 224×224 + normalisation ImageNet) est appliqué à l'identique pour l'entraînement, la validation, le test et l'inférence.

---

## Résultats du Clustering

| Features | Algo | ARI ↑ | Commentaire |
|---|---|---|---|
| RadImageNet | K-Means | -0.001 | Structure non alignée avec les labels |
| RadImageNet | DBSCAN | 0.009 | 5 clusters, trop de bruit |
| **HOG** | **K-Means** | **0.198 ★** | **Meilleur ARI → retenu pour pseudo-labels** |
| HOG | DBSCAN | 0.042 | Espace trop dense |

> Le StandardScaler et le clustering sont ajustés **uniquement sur les données d'entraînement et non étiquetées** — les 30 images de test sont strictement exclues pour éviter toute fuite de données. L'ARI est calculé sur les images de train uniquement.

### Filtrage des pseudo-labels

Un classifieur SVC calibré estime la confiance de chaque pseudo-label. Un balayage de seuils mesure le compromis entre volume conservé et fiabilité :

| Seuil | Images conservées | % du pool | Fiabilité (accord clustering/SVC) |
|---|---|---|---|
| 0.55 | 1 050 | 76.4 % | 56.9 % |
| **0.60** | **774** | **56.3 %** | **61.5 %** |
| 0.65 | 444 | 32.3 % | 67.3 % |
| 0.70 | 139 | 10.1 % | 67.6 % |
| 0.75 | 33 | 2.4 % | 72.7 % |

> **Seuil retenu : 0.60** — meilleur compromis entre volume de données (774 images) et fiabilité des pseudo-labels.

---

## Comparaison des 3 modèles

| Modèle | Données d'entraînement | Accuracy | Rappel cancer ★ | F1 |
|---|---|---|---|---|
| A — Faible seul | 774 pseudo-labels filtrés | 0.567 | 0.467 ✗ | 0.519 |
| **B — Semi-supervisé** | **Faible → Fort (finetune)** | **0.800** | **0.933 ✓** | **0.824 ✓** |
| C — Fort seul | 56 images (vrais labels) | 0.500 | 1.000 ✓ | 0.667 |

> ★ **Le rappel sur la classe cancer est la métrique prioritaire** : un faux négatif (cancer non détecté) est bien plus grave qu'un faux positif dans un contexte de dépistage médical.

**→ Le modèle B (semi-supervisé) est le meilleur compromis** : meilleure accuracy (0.800) et meilleur F1 (0.824), avec un rappel cancer de 0.933.

> L'early stopping est basé sur la **loss de validation** (14 images issues des données d'entraînement), avec restauration automatique des meilleurs poids.

---

## Prédictions sur le pool non étiqueté

Après validation du modèle B, prédiction sur les 1 374 images sans label :

| Label prédit | Nombre d'images |
|---|---|
| Cancer | 1 162 |
| Normal | 212 |
| Confiance moyenne | 0.740 |
| Images à faible confiance (< 0.70) | 322 |

> Le modèle est volontairement conservateur (biais vers la classe cancer pour maximiser le rappel). Les 322 images à faible confiance sont à prioriser pour une validation médicale humaine avant tout usage.

---

## Technologies utilisées

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.x-green)

- **PyTorch / Torchvision** — Deep learning & CNN
- **RadImageNet** — Backbone médical pré-entraîné (via HuggingFace Hub)
- **OpenCV (CLAHE)** — Égalisation adaptative des histogrammes
- **scikit-learn** — Clustering, métriques, PCA, t-SNE, SVC
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
- Jeu de validation de **14 images** — l'early stopping peut être sensible à cette petite taille
- Dataset hétérogène : mélange de plans de coupe (axial/sagittal/coronal) et possible mélange IRM/scanner
- Pas de métadonnées DICOM disponibles pour identifier les modalités
- Confiance maximale des pseudo-labels limitée à 0.84 (classifieur SVC entraîné sur peu d'images)

---

## 👤 Auteur

**Fatima ADDA-REZIG**  
Data Science — CurelyticsIA
