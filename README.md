# Document Analysis Pipeline

Pipeline d'analyse de documents avec Machine Learning pour la normalisation et l'extraction de features.

## Architecture

Le projet est organisé en plusieurs modules :

- **Pipeline** : Étapes de traitement (prétraitement, colométrie, géométrie, features)
- **API** : Interface REST pour l'analyse de documents
- **CLI** : Interface en ligne de commande pour exécuter les étapes séparément
- **Utils** : Utilitaires et configuration

### Flux du Pipeline

```
Document brut (PDF/Image)
    ↓
[Prétraitement] → Amélioration contraste + Classification SCAN/PHOTO
    ↓
[Géométrie] → Crop, Deskew, Rotation (orientation 0/90/180/270)
    ↓
[Features] → Extraction de features (OCR, checkboxes, etc.)
```

Voir [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) pour plus de détails.

## Installation

### 1. Créer un environnement virtuel

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

**Note sur les PDFs** : Le pipeline utilise PyMuPDF par défaut pour convertir les PDFs (aucune dépendance externe requise). Si vous préférez utiliser `pdf2image`, vous devrez installer poppler :
- **Windows** : Télécharger [poppler](https://github.com/oschwartz10612/poppler-windows/releases/) et l'ajouter au PATH
- **Linux** : `sudo apt-get install poppler-utils`
- **Mac** : `brew install poppler`

### 3. Configuration

#### Variables d'environnement

Copier `env.example` vers `.env` et configurer les variables d'environnement si nécessaire :

```bash
# Windows
copy env.example .env

# Linux/Mac
cp env.example .env
```

#### Fichier de configuration du pipeline

Le fichier `config.yaml` à la racine du projet contient tous les seuils et paramètres du pipeline :

```yaml
geometry:
  deskew:
    enabled: true
    min_confidence: 0.20  # Ne pas deskewer si confiance < 20%
    max_angle: 15.0
    min_angle: 0.5
  orientation:
    min_confidence: 0.70
  crop:
    min_area_ratio: 0.85
```

**Documentation complète** : Voir [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) pour tous les paramètres disponibles et des exemples d'utilisation.

**Exemple important** - Ajuster le seuil de deskew :
```yaml
# Ne deskewer que si la confiance est > 40%
geometry:
  deskew:
    min_confidence: 0.40
```

## Utilisation

### Test avec un PDF

Pour tester le pipeline avec un PDF :

1. **Placer votre PDF dans `data/input/`** :
```bash
copy votre_document.pdf data\input\
```

2. **Exécuter le pipeline complet** :
```bash
python -m src.cli.main pipeline --input data/input --output data/output
```

3. **Ou exécuter les étapes séparément** :
```bash
# Étape 1: Prétraitement (amélioration contraste + classification)
python -m src.cli.main preprocessing --input data/input --output data/processed/preprocessing

# Étape 2: Normalisation géométrique (crop, deskew, rotation)
python -m src.cli.main geometry --input data/processed/preprocessing --output data/output/geometry
```

4. **Ou utiliser le script de test** :
```bash
python test_pdf_example.py
```

**Note** : Pour les PDFs multi-pages, chaque page sera traitée et sauvegardée avec le suffixe `_page1.png`, `_page2.png`, etc.

Voir `docs/TESTING.md` pour plus de détails.

### Ligne de commande

Exécuter une étape spécifique du pipeline :

```bash
# Étape 1: Prétraitement (amélioration contraste + classification SCAN/PHOTO)
python -m src.cli.main preprocessing --input data/input/ --output data/processed/preprocessing/

# Étape 2: Normalisation géométrie (crop, deskew, rotation)
python -m src.cli.main geometry --input data/processed/preprocessing/ --output data/output/geometry/

# Normalisation colométrie (optionnel)
python -m src.cli.main colometry --input data/input/ --output data/processed/colometry/

# Extraction de features
python -m src.cli.main features --input data/output/geometry/ --output data/output/

# Exécuter tout le pipeline (prétraitement → géométrie → features)
python -m src.cli.main pipeline --input data/input/ --output data/output/

# Exécuter des étapes spécifiques du pipeline
python -m src.cli.main pipeline --input data/input/ --output data/output/ --stages preprocessing --stages geometry
```

**Important** : L'étape `geometry` attend maintenant la sortie de `preprocessing` en entrée. Les images doivent être prétraitées avec leurs métadonnées JSON correspondantes.

### API

Démarrer le serveur API :

```bash
python -m src.api.app
```

L'API sera accessible sur `http://localhost:8000`

**Documentation interactive :**
- **Swagger UI** : `http://localhost:8000/docs`
- **ReDoc** : `http://localhost:8000/redoc`

Endpoints disponibles :
- `POST /api/v1/analyze` : Analyser un document (pipeline complet)
- `POST /api/v1/pipeline/colometry` : Normalisation colométrie
- `POST /api/v1/pipeline/geometry` : Normalisation géométrie
- `POST /api/v1/pipeline/features` : Extraction de features
- `GET /api/v1/pipeline/status` : Statut du pipeline
- `GET /api/v1/results/{task_id}` : Statut et résultats d'une tâche asynchrone

## Structure du projet

```
doc_analysis/
├── src/
│   ├── pipeline/          # Étapes du pipeline
│   │   ├── preprocessing.py  # Prétraitement (contraste + classification)
│   │   ├── geometry.py       # Normalisation géométrique
│   │   ├── colometry.py      # Normalisation colométrie
│   │   └── features.py       # Extraction de features
│   ├── api/               # API REST
│   ├── cli/               # Interface ligne de commande
│   ├── utils/             # Utilitaires
│   └── models/            # Modèles ML
├── data/
│   ├── input/             # Documents d'entrée
│   ├── processed/         # Documents traités (intermédiaires)
│   │   └── preprocessing/ # Sortie de l'étape preprocessing
│   └── output/            # Résultats finaux
│       └── geometry/       # Sortie de l'étape geometry
├── tests/                 # Tests unitaires
└── docs/                  # Documentation
```

## Rapport QA

Générer un rapport QA HTML avec galerie avant/après et statistiques :

```bash
python qa_report.py --output-dir data/output --output qa_report.html
```

Le rapport inclut :
- **Flags de qualité** : low_confidence_orientation, overcrop_risk, no_quad_detected, dewarp_applied, low_contrast_after_enhance, too_small_final
- **KPIs** : Orientation accuracy, taux d'overcrop, taux no_quad_detected, temps moyen par page
- **Galerie** : Vignettes avant/après avec masques/contours pour chaque page
- **Tableau des flags** : Vue d'ensemble de tous les flags par page

Les fichiers `.qa.json` sont générés automatiquement pour chaque image traitée.

## Développement

### Exécuter les tests

```bash
pytest tests/
```

### Formatage du code

```bash
black src/
flake8 src/
```

## Licence

MIT

