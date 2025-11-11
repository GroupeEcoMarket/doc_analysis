# Pipeline d'Entraînement pour la Classification de Documents

Ce module contient les scripts pour préparer les données et entraîner les modèles de classification de type de document.

## Structure

```
training/
├── __init__.py
├── create_dataset.py      # Script pour organiser les données d'entraînement
├── train_classifier.py    # Script d'entraînement principal
└── README.md              # Ce fichier
```

## Préparation des Données

### Étape 1 : Extraire les features OCR des images

Si vous avez des **images PNG/JPG** organisées par type :

```
training/
├── Type1/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── Type2/
│   └── image3.png
└── ...
```

Utilisez `prepare_training_data.py` pour extraire les features OCR :

```bash
# Utiliser les chemins par défaut depuis config.yaml
# (input: training_data/raw, output: training_data/processed)
poetry run python training/prepare_training_data.py

# Ou spécifier explicitement les répertoires
poetry run python training/prepare_training_data.py --input-dir training/ --output-dir training_data/processed
```

Ce script :
1. Scanne chaque sous-dossier (= un type de document)
2. Extrait les features OCR de chaque image avec `FeatureExtractor` **en parallèle**
3. Sauvegarde les résultats dans des fichiers JSON
4. Maintient la structure de dossiers

**Options :**
- `--input-dir PATH`: Répertoire contenant les sous-dossiers d'images par type (défaut depuis `config.yaml`: `training_data/raw`)
- `--output-dir PATH`: Répertoire de sortie pour les fichiers JSON (défaut depuis `config.yaml`: `training_data/processed`)
- `--workers N` ou `-w N`: Nombre de processus parallèles (défaut: `cpu_count() - 1`)
- `--config FILE` ou `-c FILE`: Fichier de configuration personnalisé

**Exemples :**
```bash
# Utilisation par défaut (utilise les valeurs de config.yaml)
poetry run python training/prepare_training_data.py

# Spécifier uniquement le répertoire d'entrée
poetry run python training/prepare_training_data.py --input-dir training/

# Limiter à 2 workers (pour économiser la RAM)
poetry run python training/prepare_training_data.py --workers 2

# Utiliser tous les CPUs
poetry run python training/prepare_training_data.py --workers 8

# Spécifier des répertoires personnalisés
poetry run python training/prepare_training_data.py \
    --input-dir training/ \
    --output-dir custom_output/
```

**Performances :**
- Sur un CPU 8 cœurs : **~8x plus rapide** qu'en séquentiel
- Chaque worker charge son propre modèle OCR en mémoire
- Réduire `--workers` si vous manquez de RAM (chaque worker ~2-3 GB)

**Structure générée en sortie :**
```
training_data/processed/
├── Type1/
│   ├── image1.json
│   ├── image2.json
│   └── ...
├── Type2/
│   └── image3.json
└── ...
```

### Étape 1 (alternative) : Organiser des fichiers JSON existants

Si vous avez déjà des fichiers JSON OCR, utilisez `create_dataset.py` :

**Structure attendue en entrée :**
```
raw_data/
├── doc1.json
├── doc2.json
└── ...
```

**Utilisation :**
```bash
# Organiser les données dans training_data/processed (chemin par défaut pour train_classifier.py)
poetry run python training/create_dataset.py raw_data/ training_data/processed
```

**Options :**
- `--metadata FILE`: Fichier JSON optionnel avec mapping `{filename: document_type}`

**Exemple avec metadata :**
```json
{
  "doc1.json": "Attestation_CEE",
  "doc2.json": "Attestation_CEE",
  "doc3.json": "Facture"
}
```

## Entraînement du Modèle

### 2. Entraîner le modèle avec `train_classifier.py`

Ce script :
1. Scanne le dossier de données d'entraînement
2. Vectorise chaque document avec `feature_engineering.py`
3. Entraîne un modèle ML (LightGBM, LogisticRegression, ou RandomForest)
4. Sauvegarde le modèle et un rapport de performance

**Utilisation de base :**
```bash
poetry run python training/train_classifier.py
```

**Options principales :**
- `--data-dir PATH`: Répertoire contenant les données d'entraînement (sous-dossiers par type) (défaut depuis `config.yaml`: `training_data/processed`)
- `--model-path PATH`: Chemin pour sauvegarder le modèle (défaut depuis `config.yaml`: `training_data/artifacts/document_classifier.joblib`)
- `--report-path PATH`: Chemin pour sauvegarder le rapport (défaut depuis `config.yaml`: `training_data/artifacts/training_report.json`)
- `--model-type TYPE`: Type de modèle (`lightgbm`, `logistic_regression`, `random_forest`) (défaut: `lightgbm`)
- `--test-size FLOAT`: Proportion des données pour le test (défaut: `0.2`)
- `--random-state INT`: Seed aléatoire pour la reproductibilité (défaut: `42`)
- `--embedding-model NAME`: Modèle sentence-transformers (défaut depuis `config.yaml`: `antoinelouis/french-me5-base`)
- `--min-confidence FLOAT`: Seuil de confiance minimum pour filtrer les lignes OCR (défaut depuis `config.yaml`: `0.70`)
- `--workers N` ou `-w N`: Nombre de workers parallèles pour la vectorisation (défaut: `cpu_count() - 1`)

**Exemples :**

```bash
# Entraîner avec LightGBM (par défaut, utilise les valeurs de config.yaml)
poetry run python training/train_classifier.py

# Spécifier un répertoire de données différent
poetry run python training/train_classifier.py --data-dir training_data/

# Entraîner avec LogisticRegression
poetry run python training/train_classifier.py --model-type logistic_regression

# Entraîner avec RandomForest
poetry run python training/train_classifier.py --model-type random_forest

# Personnaliser les chemins de sortie
poetry run python training/train_classifier.py \
    --model-path models/my_classifier.joblib \
    --report-path models/my_report.json

# Utiliser un modèle sentence-transformers différent
poetry run python training/train_classifier.py \
    --embedding-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# Limiter le nombre de workers pour économiser la RAM
poetry run python training/train_classifier.py --workers 2
```

## Format des Données d'Entraînement

### Format JSON OCR

Chaque fichier JSON doit contenir les résultats OCR au format suivant :

```json
{
  "ocr_lines": [
    {
      "text": "Texte de la ligne",
      "confidence": 0.95,
      "bounding_box": [x1, y1, x2, y2]
    },
    ...
  ],
  "checkboxes": [...]
}
```

Ce format correspond à la sortie de `src/pipeline/features.py`.

## Structure des Données d'Entraînement

Le script `train_classifier.py` attend une structure de dossiers où chaque sous-dossier représente un type de document. Par défaut, il cherche dans `training_data/processed` (configurable via `config.yaml` → `paths.training_processed_dir`) :

```
training_data/processed/
├── Attestation_CEE/
│   ├── doc1.json
│   ├── doc2.json
│   └── ...
├── Facture/
│   ├── doc1.json
│   └── ...
├── Contrat/
│   └── ...
└── ...
```

Le nom du sous-dossier devient automatiquement le label de classe.

## Sorties

### Modèle Sauvegardé

Le modèle est sauvegardé au format joblib avec la structure suivante :

```python
{
    'model': <modèle_entraîné>,
    'class_names': ['Attestation_CEE', 'Facture', ...]
}
```

### Rapport de Performance

Le rapport JSON contient :
- Métriques globales (accuracy, F1-score)
- Rapport de classification par classe
- Matrice de confusion
- Liste des classes
- Importance des features (pour les modèles qui le supportent : LightGBM, RandomForest)
  - Pourcentage d'importance du texte sémantique
  - Pourcentage d'importance des features de position

## Dépendances

- `scikit-learn`: Pour les modèles sklearn et les métriques
- `lightgbm`: Pour le modèle LightGBM (optionnel mais recommandé)
- `sentence-transformers`: Pour les embeddings sémantiques
- `joblib`: Pour sauvegarder/charger les modèles (inclus avec scikit-learn)

## Notes

- Le modèle LightGBM est recommandé pour de meilleures performances
- Assurez-vous d'avoir suffisamment de données par classe (minimum 10-20 documents recommandé)
- La vectorisation peut prendre du temps selon le nombre de documents
- Le modèle sauvegardé peut être utilisé directement avec `DocumentClassifier`

