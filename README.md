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

>  **Point de vigilance détecté** : 32 images du pool labellisé étaient dupliquées dans le pool non étiqueté → fuite de données potentielle corrigée avant toute modélisation.

> **Dataset** : disponible sur demande auprès de CurelyticsIA. Placer le fichier `mri_dataset_brain_cancer_oc.zip` à la racine du projet avant d'exécuter les notebooks.

---

## Installation

git clone https://github.com/fatimaadda2878/BrainScanAI
```bash
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

## 🔬 Extraction de features — RadImageNet

J'ai utilisé **RadImageNet**, un ResNet50 pré-entraîné sur **1.35 million d'images médicales** (IRM, scanner CT, échographies), comparé aux features HOG classiques. Une égalisation des histogrammes **CLAHE** est appliquée en amont pour améliorer le contraste des images.

| Méthode | Dimension | Pertinence médicale |
|---|---|---|
| HOG (classique) | 1 568 | Faible |
| RadImageNet (ResNet50) | 2 048 | ⭐ Élevée |

> Le même preprocessing (CLAHE + 224×224 + normalisation ImageNet) est appliqué à l'identique pour l'entraînement, la validation, le test et l'inférence.

---

## Séparation stricte des données

Pour garantir des métriques fiables, les données sont séparées dès le départ :

- **Test (30 images)** — jamais vues pendant l'entraînement, servent uniquement à l'évaluation finale
- **Validation (14 images)** — issues des données d'entraînement, servent à l'early stopping
- **Train (56 images)** — entraînement des modèles

> Le **StandardScaler**, le **clustering** et le **mapping des pseudo-labels** sont ajustés uniquement sur les 56 images de train et le pool non étiqueté — ni les images de test, ni celles de validation ne participent à ces étapes. L'ARI est calculé sur les seules images de train.

---

## Résultats du Clustering

| Features | Algo | ARI ↑ | Commentaire |
|---|---|---|---|
| RadImageNet | K-Means | -0.010 | Structure non alignée avec les labels |
| RadImageNet | DBSCAN | — | Peu de clusters exploitables |
| **HOG** | **K-Means** | **~0.18 ★** | **Meilleur ARI → retenu pour pseudo-labels** |
| HOG | DBSCAN | — | Espace trop dense |

### Filtrage des pseudo-labels

Un classifieur SVC calibré (entraîné sur les images de train uniquement) estime la confiance de chaque pseudo-label. Un balayage de seuils mesure le compromis entre volume conservé et fiabilité (accord entre le clustering et le SVC). **Seuil retenu : 0.60**, meilleur compromis entre volume de données et fiabilité.

---

## Comparaison des 3 modèles

| Modèle | Données d'entraînement | Accuracy | Rappel cancer ★ | F1 |
|---|---|---|---|---|
| A — Faible seul | pseudo-labels filtrés | 0.567 | 0.467 | 0.519 |
| **B — Semi-supervisé** | **Faible → Fort (finetune)** | **0.767** | **0.867** | **0.788** |
| C — Fort seul | 56 images (vrais labels) | 0.800 | 0.933 | 0.824 |

> ★ **Le rappel sur la classe cancer est la métrique prioritaire** : un faux négatif (cancer non détecté) est bien plus grave qu'un faux positif dans un contexte de dépistage médical.

**Analyse** : après correction de la méthodologie (séparation stricte train/validation/test), les modèles B et C obtiennent des performances très proches. Le semi-supervisé n'apporte pas de gain systématique sur ce dataset — ce qui s'explique par la petite taille du jeu de test (30 images) et par le fait que RadImageNet, déjà performant sur images médicales, laisse peu de marge au pré-entraînement faible. L'écart entre B et C se joue sur un seul patient (sur 15), ce qui n'est pas statistiquement significatif.

> L'early stopping est basé sur la **loss de validation** (14 images), avec restauration automatique des meilleurs poids.

---

## Prédictions sur le pool non étiqueté

Après validation, prédiction du modèle sur les 1 374 images sans label :

| Label prédit | Nombre d'images |
|---|---|
| Cancer | 1 158 |
| Normal | 216 |
| Confiance moyenne | 0.716 |
| Images à faible confiance (< 0.70) | 558 |

> Le modèle est volontairement conservateur (biais vers la classe cancer pour maximiser le rappel). Les 558 images à faible confiance sont à prioriser pour une validation médicale humaine avant tout usage.

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
- Jeu de validation de **14 images** — l'early stopping est sensible à cette petite taille (le modèle C a nécessité une patience plus élevée pour converger)
- Dataset hétérogène : mélange de plans de coupe (axial/sagittal/coronal) et possible mélange IRM/scanner
- Pas de métadonnées DICOM disponibles pour identifier les modalités
- Le gain du semi-supervisé n'est pas significatif sur ce dataset — un jeu de test plus grand serait nécessaire pour conclure

---

## 👤 Auteur

**Fatima ADDA-REZIG**  
Data Science — CurelyticsIA
