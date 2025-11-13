# Variables d'Environnement - Documentation Complète

Ce document liste toutes les variables d'environnement supportées et comment elles sont chargées.

## Chargement des Variables d'Environnement

Les variables d'environnement sont chargées depuis le fichier `.env` à plusieurs endroits :

1. **`src/utils/config_loader.py`** : Chargé au démarrage du module (priorité la plus haute)
2. **`src/workers.py`** : Chargé au démarrage des workers Dramatiq
3. **`src/api/app.py`** : Chargé au démarrage de l'API FastAPI
4. **`src/utils/storage.py`** : Chargé au démarrage du module de stockage
5. **`src/utils/logger.py`** : Chargé dans `setup_logging()`

**Important** : En Docker, les variables d'environnement sont passées directement via `docker-compose.yml` ou `docker-compose.prod.yml` et n'ont pas besoin d'un fichier `.env`.

## Variables d'Environnement Supportées

### API Configuration
- `API_HOST` : Adresse IP d'écoute de l'API (défaut: `0.0.0.0`)
- `API_PORT` : Port d'écoute de l'API (défaut: `8000`)
- `API_DEBUG` : Mode debug (défaut: `False`)

**Utilisation** : Utilisées directement via `os.getenv()` dans `src/api/app.py`

### Redis Configuration
- `REDIS_HOST` : Host Redis pour Dramatiq (défaut: `redis` en Docker, `localhost` en local)

**Mapping** : `config.get('redis.host')` → `REDIS_HOST`

### Chemins de Données
- `INPUT_DIR` : Répertoire d'entrée (défaut: `data/input`)
- `OUTPUT_DIR` : Répertoire de sortie (défaut: `data/output`)
- `PROCESSED_DIR` : Répertoire des fichiers traités (défaut: `data/processed`)
- `MODEL_PATH` : Répertoire des modèles ML (défaut: `models/`)
- `TEMP_STORAGE_DIR` : Répertoire de stockage temporaire (défaut: `data/temp_storage`)

**Mapping** :
- `config.get('paths.input_dir')` → `INPUT_DIR`
- `config.get('paths.output_dir')` → `OUTPUT_DIR`
- `config.get('paths.processed_dir')` → `PROCESSED_DIR`
- `config.get('paths.model_path')` → `MODEL_PATH`
- `config.get('paths.temp_storage_dir')` → `TEMP_STORAGE_DIR`

### Modèles de Classification
- `CLASSIFICATION_MODEL_PATH` : Chemin vers le modèle de classification (défaut: `training_data/artifacts/document_classifier.joblib`)
- `CLASSIFICATION_EMBEDDING_MODEL` : Modèle sentence-transformers pour les embeddings (défaut: `antoinelouis/french-me5-base`)

**Mapping** :
- `config.get('classification.model_path')` → `CLASSIFICATION_MODEL_PATH`
- `config.get('classification.embedding_model')` → `CLASSIFICATION_EMBEDDING_MODEL`

### Chemins d'Entraînement
- `TRAINING_RAW_DIR` : Répertoire des données brutes d'entraînement (défaut: `training_data/raw`)
- `TRAINING_PROCESSED_DIR` : Répertoire des données préparées (défaut: `training_data/processed`)
- `TRAINING_ARTIFACTS_DIR` : Répertoire des modèles et rapports (défaut: `training_data/artifacts`)

**Mapping** :
- `config.get('paths.training_raw_dir')` → `TRAINING_RAW_DIR`
- `config.get('paths.training_processed_dir')` → `TRAINING_PROCESSED_DIR`
- `config.get('paths.training_artifacts_dir')` → `TRAINING_ARTIFACTS_DIR`

### Performance
- `PERFORMANCE_BATCH_SIZE` : Taille des lots pour le traitement (défaut: `10`)
- `PERFORMANCE_MAX_WORKERS` : Nombre maximum de workers parallèles (défaut: `4`)

**Mapping** :
- `config.get('performance.batch_size')` → `PERFORMANCE_BATCH_SIZE`
- `config.get('performance.max_workers')` → `PERFORMANCE_MAX_WORKERS`

**Utilisation** : Utilisées directement via `os.getenv()` dans `src/utils/config_loader.py._load_performance_config()`

### Storage
- `STORAGE_BACKEND` : Backend de stockage (`local`, `s3`, `minio`) (défaut: `local`)

**Mapping** : `config.get('storage.backend')` → `STORAGE_BACKEND`

**Utilisation** : Utilisée directement via `os.getenv()` dans `src/utils/storage.py.get_storage()`

### Logging
- `LOG_LEVEL` : Niveau de log (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) (défaut: `INFO`)
- `LOG_FILE` : Chemin vers le fichier de log (optionnel)

**Utilisation** : Utilisées directement via `os.getenv()` dans `src/utils/logger.py.setup_logging()`

### Workers Dramatiq
- `DRAMATIQ_PROCESSES` : Nombre de processus workers (défaut: `4`)
- `DRAMATIQ_THREADS` : Nombre de threads par processus (défaut: `1`)

**Utilisation** : Utilisées directement dans les commandes Docker via `${DRAMATIQ_PROCESSES}` et `${DRAMATIQ_THREADS}`

### OCR Worker (Docker uniquement)
- `OCR_WORKER_PROCESSES` : Nombre de processus workers OCR (défaut: `2`)
- `OCR_WORKER_THREADS` : Nombre de threads par processus OCR (défaut: `1`)
- `OCR_STORAGE_DIR` : Répertoire de stockage pour le worker OCR (défaut: `/app/data/temp_storage`)

**Utilisation** : Utilisées directement dans les commandes Docker

## Priorité de Chargement

Pour les variables mappées dans `config_loader.py`, la priorité est :

1. **Variable d'environnement** (`.env` ou Docker `environment`)
2. **config.yaml** (si la variable n'est pas définie)
3. **Valeur par défaut** (dans le code)

Pour les variables utilisées directement via `os.getenv()` :

1. **Variable d'environnement** (`.env` ou Docker `environment`)
2. **Valeur par défaut** (dans le code)

## Vérification

Pour vérifier qu'une variable d'environnement est bien chargée :

1. **Vérifier les logs** : Les modules loggent souvent quelle valeur est utilisée
2. **Tester dans le code** : Ajouter un log temporaire pour voir la valeur
3. **Vérifier Docker** : `docker-compose exec <service> env | grep <VAR_NAME>`

## Exemple de .env

```bash
# API
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=False

# Redis
REDIS_HOST=redis

# Chemins
INPUT_DIR=data/input
OUTPUT_DIR=data/output
PROCESSED_DIR=data/processed
MODEL_PATH=models/
TEMP_STORAGE_DIR=data/temp_storage

# Classification
CLASSIFICATION_MODEL_PATH=training_data/artifacts/document_classifier.joblib
CLASSIFICATION_EMBEDDING_MODEL=antoinelouis/french-me5-base

# Training
TRAINING_RAW_DIR=training_data/raw
TRAINING_PROCESSED_DIR=training_data/processed
TRAINING_ARTIFACTS_DIR=training_data/artifacts

# Performance
PERFORMANCE_BATCH_SIZE=10
PERFORMANCE_MAX_WORKERS=4

# Storage
STORAGE_BACKEND=local

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Workers
DRAMATIQ_PROCESSES=4
DRAMATIQ_THREADS=1
```

