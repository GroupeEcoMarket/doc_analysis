# Vérification Complète des Variables d'Environnement

Ce document liste toutes les vérifications effectuées pour s'assurer que les variables d'environnement sont correctement chargées depuis le fichier `.env`.

## ✅ Points de Chargement du .env

Les fichiers suivants chargent maintenant le `.env` au démarrage :

1. **`src/utils/config_loader.py`** ✅
   - Chargé au niveau du module (ligne 15-19)
   - Priorité la plus haute - chargé avant toute utilisation de `os.getenv()`

2. **`src/workers.py`** ✅
   - Chargé au début du fichier (ligne 27-33)
   - Avant l'import des modules qui utilisent `get_config()`

3. **`src/api/app.py`** ✅
   - Chargé au début du fichier (ligne 19-23)
   - Avant `setup_logging()`

4. **`src/utils/storage.py`** ✅
   - Chargé au début du fichier (ligne 27-31)
   - Avant toute utilisation de `os.getenv()`

5. **`src/utils/logger.py`** ✅
   - Chargé dans `setup_logging()` (ligne 29-33)
   - Déjà présent

## ✅ Mapping des Variables d'Environnement

Toutes les variables d'environnement définies dans `env.example` sont maintenant mappées dans `config_loader.py` :

### Classification
- ✅ `CLASSIFICATION_MODEL_PATH` → `config.get('classification.model_path')`
- ✅ `CLASSIFICATION_EMBEDDING_MODEL` → `config.get('classification.embedding_model')`

### Chemins de Données
- ✅ `INPUT_DIR` → `config.get('paths.input_dir')`
- ✅ `OUTPUT_DIR` → `config.get('paths.output_dir')`
- ✅ `PROCESSED_DIR` → `config.get('paths.processed_dir')`
- ✅ `MODEL_PATH` → `config.get('paths.model_path')`
- ✅ `TEMP_STORAGE_DIR` → `config.get('paths.temp_storage_dir')`

### Chemins d'Entraînement
- ✅ `TRAINING_RAW_DIR` → `config.get('paths.training_raw_dir')`
- ✅ `TRAINING_PROCESSED_DIR` → `config.get('paths.training_processed_dir')`
- ✅ `TRAINING_ARTIFACTS_DIR` → `config.get('paths.training_artifacts_dir')`

### Redis
- ✅ `REDIS_HOST` → `config.get('redis.host')`
- ✅ Utilisé explicitement dans `src/workers.py` et `src/utils/ocr_client.py` pour configurer `RedisBroker`

### Storage
- ✅ `STORAGE_BACKEND` → `config.get('storage.backend')`
- ✅ Utilisé directement via `os.getenv()` dans `src/utils/storage.py` (avec `.env` chargé)

### Performance
- ✅ `PERFORMANCE_BATCH_SIZE` → `config.get('performance.batch_size')`
- ✅ `PERFORMANCE_MAX_WORKERS` → `config.get('performance.max_workers')`
- ✅ Utilisé directement via `os.getenv()` dans `config_loader.py._load_performance_config()` (avec `.env` chargé)

### Metrics (Prometheus)
- ✅ `METRICS_WORKERS_PORT` → `config.get('metrics.workers_port')`
- ✅ `METRICS_WORKERS_HOST` → `config.get('metrics.workers_host')`
- ✅ `METRICS_QUEUE_MONITOR_INTERVAL` → `config.get('metrics.queue_monitor_interval')`

### API (utilisées directement)
- ✅ `API_HOST` → `os.getenv()` dans `src/api/app.py` (avec `.env` chargé)
- ✅ `API_PORT` → `os.getenv()` dans `src/api/app.py` (avec `.env` chargé)
- ✅ `API_DEBUG` → `os.getenv()` dans `src/api/app.py` (avec `.env` chargé)

### Logging (utilisées directement)
- ✅ `LOG_LEVEL` → `os.getenv()` dans `src/utils/logger.py` (avec `.env` chargé)
- ✅ `LOG_FILE` → `os.getenv()` dans `src/utils/logger.py` (avec `.env` chargé)

## ✅ Corrections Apportées

### 1. Chargement du .env
- ✅ Ajouté dans `src/utils/config_loader.py` au niveau du module
- ✅ Ajouté dans `src/workers.py` au début du fichier
- ✅ Ajouté dans `src/api/app.py` au début du fichier
- ✅ Ajouté dans `src/utils/storage.py` au début du fichier

### 2. Mapping des Variables
- ✅ Ajouté tous les mappings manquants dans `config_loader.py` :
  - `paths.input_dir`, `paths.output_dir`, `paths.processed_dir`, `paths.model_path`
  - `redis.host`
  - `storage.backend`
  - `metrics.workers_port`, `metrics.workers_host`, `metrics.queue_monitor_interval`

### 3. Configuration Redis
- ✅ `REDIS_HOST` est maintenant utilisé explicitement dans `src/workers.py`
- ✅ `REDIS_HOST` est maintenant utilisé explicitement dans `src/utils/ocr_client.py`
- ✅ Construction de l'URL Redis : `redis://{REDIS_HOST}:{REDIS_PORT}`

### 4. Logs de Débogage
- ✅ Ajout de logs dans `classifier_service.py` pour indiquer quelle source est utilisée pour le modèle d'embedding
- ✅ Avertissement si `CLASSIFICATION_EMBEDDING_MODEL` n'est pas définie

## ✅ Tests de Vérification

Pour vérifier que toutes les variables sont bien chargées, vous pouvez :

1. **Vérifier les logs au démarrage** :
   ```bash
   docker-compose logs workers | grep -i "embedding\|redis\|config"
   ```

2. **Tester dans Python** :
   ```python
   from src.utils.config_loader import get_config
   config = get_config()
   
   # Vérifier les valeurs
   print(f"Embedding model: {config.get('classification.embedding_model')}")
   print(f"Redis host: {config.get('redis.host')}")
   print(f"Input dir: {config.get('paths.input_dir')}")
   ```

3. **Vérifier dans Docker** :
   ```bash
   docker-compose exec workers python -c "from src.utils.config_loader import get_config; c = get_config(); print(c.get('classification.embedding_model'))"
   ```

## ⚠️ Notes Importantes

1. **Priorité** : Variable d'environnement > config.yaml > valeur par défaut
2. **Docker** : Les variables sont passées via `docker-compose.yml` et n'ont pas besoin de `.env` (mais le chargement est fait pour compatibilité)
3. **Redis** : `REDIS_HOST` est maintenant utilisé explicitement pour construire l'URL Redis
4. **Performance** : `PERFORMANCE_BATCH_SIZE` et `PERFORMANCE_MAX_WORKERS` utilisent `os.getenv()` directement mais le `.env` est chargé dans `config_loader.py`

## 📋 Checklist de Vérification

- [x] `.env` chargé dans `config_loader.py`
- [x] `.env` chargé dans `workers.py`
- [x] `.env` chargé dans `api/app.py`
- [x] `.env` chargé dans `storage.py`
- [x] Tous les mappings ajoutés dans `config_loader.py`
- [x] `REDIS_HOST` utilisé explicitement pour RedisBroker
- [x] `CLASSIFICATION_EMBEDDING_MODEL` avec logs de débogage
- [x] Toutes les variables utilisées via `os.getenv()` ont le `.env` chargé

