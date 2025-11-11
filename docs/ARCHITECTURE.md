# Architecture du Pipeline d'Analyse de Documents

## Vue d'ensemble

Le projet est structuré en un pipeline modulaire à 4 étapes pour l'analyse de documents avec Machine Learning :
1. **Prétraitement** : Amélioration du contraste et classification du type de capture
2. **Géométrie** : Normalisation géométrique (crop, deskew, rotation)
3. **Colométrie** : Normalisation colométrique (optionnel)
4. **Features** : Extraction de features (OCR, checkboxes, etc.)

## Structure des dossiers

```
doc_analysis/
├── src/                      # Code source principal
│   ├── pipeline/             # Modules du pipeline
│   │   ├── preprocessing.py  # Prétraitement (contraste + classification)
│   │   ├── geometry.py       # Normalisation géométrie
│   │   ├── colometry.py      # Normalisation colométrie
│   │   └── features.py       # Extraction de features
│   ├── api/                  # API REST
│   │   ├── app.py            # Application FastAPI principale
│   │   └── routes.py         # Routes API
│   ├── cli/                  # Interface ligne de commande
│   │   └── main.py           # Commandes CLI avec Click
│   ├── utils/                # Utilitaires
│   │   ├── config.py         # Gestion de la configuration
│   │   └── file_handler.py   # Gestion des fichiers
│   └── models/               # Modèles ML (à implémenter)
├── data/                     # Données
│   ├── input/                # Documents d'entrée
│   ├── processed/            # Documents traités (intermédiaires)
│   └── output/               # Résultats finaux
├── tests/                    # Tests unitaires
├── docs/                     # Documentation
├── models/                   # Modèles ML sauvegardés
├── logs/                     # Fichiers de logs
├── requirements.txt          # Dépendances Python
├── setup.py                  # Configuration du package
└── README.md                 # Documentation principale
```

## Pipeline modulaire

### Prétraitement (`preprocessing.py`)

**Objectif**: Préparer les images avant les transformations géométriques
- Amélioration du contraste avec CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Classification du type de capture (SCAN vs PHOTO)
- Gestion des PDFs multi-pages (chaque page est traitée séparément)

**Classe**: `PreprocessingNormalizer`
- `process(input_path, output_path)`: Traite un document unique
- `process_batch(input_dir, output_dir)`: Traite un lot de documents (gère les PDFs multi-pages)

**Sortie**: 
- Images prétraitées au format PNG
- Fichiers JSON avec métadonnées (capture_type, capture_info, etc.)

### Normalisation Colométrie (`colometry.py`)

**Objectif**: Normaliser la structure colométrique des documents
- Analyse de la disposition en colonnes
- Détection et standardisation de la structure
- Préparation pour les étapes suivantes

**Classe**: `ColometryNormalizer`
- `process(input_path, output_path)`: Traite un document unique
- `process_batch(input_dir, output_dir)`: Traite un lot de documents

### Normalisation Géométrie (`geometry.py`)

**Objectif**: Normaliser la géométrie des documents
- Détection et crop intelligent de la page (doctr db_resnet50)
- Correction de l'inclinaison fine (deskew avec transformée de Hough)
- Détection de l'orientation (0/90/180/270) avec onnxtr
- Rotation finale pour aligner le document

**Classe**: `GeometryNormalizer`
- `process(img, output_path, capture_type, original_input_path, capture_info)`: Traite une image déjà chargée
- `process_batch(input_dir, output_dir)`: Traite un lot d'images prétraitées avec parallélisation

**Entrée**: Images prétraitées depuis l'étape preprocessing (avec métadonnées JSON)
**Sortie**: Images normalisées géométriquement avec fichiers de transformation et QA

**Optimisations de Performance** :
- **Inférence par lots (Batch Inference)** : Le crop intelligent est traité en batch pour toutes les images d'un lot en une seule passe du modèle, optimisant l'utilisation du GPU/CPU
- **Parallélisation avec ProcessPoolExecutor** : Après le crop batch, les étapes de deskew, orientation et rotation sont traitées en parallèle sur plusieurs processus workers
  - Chaque worker initialise ses propres modèles (ONNX, doctr) pour éviter les problèmes de partage entre processus
  - Le nombre de workers est configuré via `performance.max_workers` dans `config.yaml`
  - En cas d'erreur du pool de processus, le système bascule automatiquement vers un traitement séquentiel (fallback)
- **Gestion robuste des erreurs** : Si une image échoue pendant le traitement parallèle, les autres images continuent d'être traitées normalement

### Extraction de Features (`features.py`)

**Objectif**: Extraire les features des documents normalisés
- Détection de checkboxes
- OCR (reconnaissance de texte)
- Autres features à définir

**Classe**: `FeatureExtractor`
- `process(input_path, output_path)`: Traite un document unique
- `process_batch(input_dir, output_dir)`: Traite un lot de documents
- `extract_checkboxes(image_path)`: Détecte les checkboxes
- `extract_ocr(image_path)`: Extrait le texte via OCR

## Interface CLI

Le module `cli/main.py` utilise Click pour fournir une interface en ligne de commande :

- `preprocessing`: Exécute uniquement le prétraitement (contraste + classification)
- `colometry`: Exécute uniquement la normalisation colométrie
- `geometry`: Exécute uniquement la normalisation géométrie (nécessite preprocessing en entrée)
- `features`: Exécute uniquement l'extraction de features
- `pipeline`: Exécute tout le pipeline ou des étapes spécifiques (via `--stages`)

## API REST

L'API FastAPI (`api/app.py`) expose les endpoints suivants :

- `POST /api/v1/analyze`: Analyse complète d'un document (pipeline complet)
- `POST /api/v1/pipeline/colometry`: Normalisation colométrie
- `POST /api/v1/pipeline/geometry`: Normalisation géométrie
- `POST /api/v1/pipeline/features`: Extraction de features
- `GET /api/v1/pipeline/status`: Statut du pipeline

## Utilitaires

### Configuration (`utils/config.py`)

Gère la configuration via variables d'environnement :
- Chargement depuis `.env`
- Valeurs par défaut
- Configuration pour API, chemins, OCR, etc.

### Gestion des fichiers (`utils/file_handler.py`)

Fonctions utilitaires pour :
- Création de répertoires
- Listing de fichiers avec filtres d'extensions
- Génération de chemins de sortie

### Initialisation des Workers (`utils/worker_init.py`)

Module partagé pour l'initialisation des workers dans le multiprocessing :
- `create_geometry_normalizer_from_dicts()` : Crée un `GeometryNormalizer` depuis des configurations sérialisées
- `create_api_dependencies_from_config()` : Crée les dépendances API (FeatureExtractor, DocumentClassifier)
- `log_worker_initialization()` : Logging standardisé pour l'initialisation des workers
- `get_config_dicts_from_config()` : Convertit un objet `Config` en dictionnaires sérialisables

Ce module centralise la logique d'initialisation pour éviter la duplication entre les différents modules (API routes, geometry pipeline).

## Flux de données

```
Document brut (PDF/Image)
    ↓
[Prétraitement] → Amélioration contraste (CLAHE) + Classification SCAN/PHOTO
    ↓ (Images PNG + Métadonnées JSON)
[Géométrie] → Crop intelligent + Deskew + Rotation (orientation 0/90/180/270)
    ↓ (Images normalisées + Fichiers transformation + QA)
[Colométrie] → Normalisation colométrie (optionnel)
    ↓
[Features] → Features extraites (OCR, checkboxes, etc.)
```

### Détails du flux

1. **Prétraitement** :
   - Charge le document (PDF ou image)
   - Pour les PDFs multi-pages, traite chaque page séparément
   - Améliore le contraste avec CLAHE
   - Classifie le type de capture (SCAN/PHOTO)
   - Sauvegarde l'image prétraitée + métadonnées JSON

2. **Géométrie** :
   - Lit les images prétraitées et leurs métadonnées JSON
   - Utilise le `capture_type` pour décider si le crop doit être appliqué
   - **Phase 1 - Crop Batch** : Traite toutes les images en batch pour le crop intelligent (une seule passe du modèle)
   - **Phase 2 - Parallélisation** : Traite deskew, orientation et rotation en parallèle avec `ProcessPoolExecutor`
   - Applique les transformations géométriques (crop, deskew, rotation)
   - Sauvegarde les images transformées + fichiers de transformation + QA

3. **Features** :
   - Lit les images normalisées géométriquement
   - Extrait les features (OCR, checkboxes, etc.)

## Extensibilité

L'architecture permet d'ajouter facilement :
- De nouvelles étapes de pipeline
- De nouveaux types de features
- De nouveaux modèles ML
- De nouveaux endpoints API

## Modules du Pipeline

### PreprocessingNormalizer

**Responsabilités** :
- Amélioration du contraste avec CLAHE
- Classification du type de capture (SCAN/PHOTO)
- Gestion des PDFs multi-pages

**Dépendances** :
- `src.utils.image_enhancer`: Fonction `enhance_contrast_clahe()`
- `src.utils.capture_classifier`: Classe `CaptureClassifier`
- `src.utils.pdf_handler`: Fonctions `is_pdf()`, `pdf_to_images()`

### GeometryNormalizer

**Responsabilités** :
- Crop intelligent de la page (doctr db_resnet50)
- Correction de l'inclinaison fine (deskew)
- Détection et correction de l'orientation (onnxtr)

**Dépendances** :
- Images prétraitées depuis `PreprocessingNormalizer`
- Métadonnées JSON avec `capture_type` et `capture_info`

**Note** : Le module ne charge plus directement les images depuis des fichiers. Il reçoit les images déjà chargées et les métadonnées depuis l'étape preprocessing.

**Architecture de Parallélisation** :
- **Module partagé** : `src/utils/worker_init.py` centralise la logique d'initialisation des workers
- **Fonctions worker** :
  - `init_geometry_worker()` : Initialise un `GeometryNormalizer` dans chaque processus worker avec ses propres modèles
  - `process_single_image_geometry()` : Traite une seule image (deskew, orientation, rotation) dans un worker
- **Processus de traitement** :
  1. Chargement des images en batch
  2. Crop intelligent en batch (toutes les images en une passe)
  3. Distribution des images croppées aux workers via `ProcessPoolExecutor`
  4. Traitement parallèle de deskew/orientation/rotation
  5. Collecte des résultats et gestion des erreurs individuelles
- **Configuration** : Le nombre de workers est défini par `performance.max_workers` dans `config.yaml`

## Prochaines étapes de développement

1. ✅ Implémenter `PreprocessingNormalizer` (fait)
2. ✅ Implémenter `GeometryNormalizer.process()` avec nouvelle signature (fait)
3. Implémenter `ColometryNormalizer.process()`
4. Implémenter `FeatureExtractor.extract_checkboxes()`
5. Implémenter `FeatureExtractor.extract_ocr()`
6. Ajouter les modèles ML nécessaires
7. Compléter les endpoints API
8. Ajouter la gestion d'erreurs et logging
9. Ajouter des tests complets

