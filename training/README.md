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
poetry run python training/prepare_training_data.py training/ training_data/
```

Ce script :
1. Scanne chaque sous-dossier (= un type de document)
2. Extrait les features OCR de chaque image avec `FeatureExtractor` **en parallèle**
3. Sauvegarde les résultats dans des fichiers JSON
4. Maintient la structure de dossiers

**Options :**
- `--workers N` : Nombre de processus parallèles (défaut: nombre de CPUs - 1)
- `--config FILE` : Fichier de configuration personnalisé

**Exemples :**
```bash
# Utilisation par défaut (automatique)
poetry run python training/prepare_training_data.py training/ training_data/

# Limiter à 2 workers (pour économiser la RAM)
poetry run python training/prepare_training_data.py training/ training_data/ --workers 2

# Utiliser tous les CPUs
poetry run python training/prepare_training_data.py training/ training_data/ --workers 8
```

**Performances :**
- Sur un CPU 8 cœurs : **~8x plus rapide** qu'en séquentiel
- Chaque worker charge son propre modèle OCR en mémoire
- Réduire `--workers` si vous manquez de RAM (chaque worker ~2-3 GB)

**Structure générée en sortie :**
```
training_data/
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
poetry run python training/create_dataset.py raw_data/ training_data/
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
poetry run python training/train_classifier.py training_data/
```

**Options principales :**
- `--model-path PATH`: Chemin pour sauvegarder le modèle (défaut: `models/document_classifier.joblib`)
- `--report-path PATH`: Chemin pour sauvegarder le rapport (défaut: `models/training_report.json`)
- `--model-type TYPE`: Type de modèle (`lightgbm`, `logistic_regression`, `random_forest`)
- `--test-size FLOAT`: Proportion des données pour le test (défaut: 0.2)
- `--semantic-model NAME`: Modèle sentence-transformers (défaut: `paraphrase-multilingual-MiniLM-L12-v2`)
- `--min-confidence FLOAT`: Seuil de confiance OCR (défaut: 0.70)

**Exemples :**

```bash
# Entraîner avec LightGBM (par défaut)
poetry run python training/train_classifier.py training_data/

# Entraîner avec LogisticRegression
poetry run python training/train_classifier.py training_data/ --model-type logistic_regression

# Entraîner avec RandomForest
poetry run python training/train_classifier.py training_data/ --model-type random_forest

# Personnaliser les chemins de sortie
poetry run python training/train_classifier.py training_data/ \
    --model-path models/my_classifier.joblib \
    --report-path models/my_report.json

# Utiliser un modèle sentence-transformers différent
poetry run python training/train_classifier.py training_data/ \
    --semantic-model sentence-transformers/paraphrase-multilingual-mpnet-base-v2
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

Le script `train_classifier.py` attend une structure de dossiers où chaque sous-dossier représente un type de document :

```
training_data/
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

