# Document Analysis Pipeline

Pipeline d'analyse de documents avec Machine Learning pour la normalisation et l'extraction de features.

## Architecture

Le projet est organisé en plusieurs modules :

- **Pipeline** : Étapes de traitement (prétraitement, colométrie, géométrie, features)
- **API** : Interface REST pour l'analyse de documents
- **CLI** : Interface en ligne de commande pour exécuter les étapes séparément
- **Utils** : Utilitaires et configuration
- **Services** : Microservices isolés (OCR)

### Architecture Microservices

Le projet utilise une architecture microservices pour isoler certains composants :

- **Service OCR isolé** (`services/ocr_service/`) : Microservice dédié à l'extraction de texte via PaddleOCR
  - Déployable indépendamment
  - Configuration autonome
  - Queue Dramatiq dédiée (`ocr-queue`)
  - Communication via messages asynchrones

### Flux du Pipeline

```
Document brut (PDF/Image)
    ↓
[Prétraitement] → Amélioration contraste + Classification SCAN/PHOTO
    ↓
[Géométrie] → Crop, Deskew, Rotation (orientation 0/90/180/270)
    ↓
[Features] → Extraction de features (OCR via microservice, checkboxes, etc.)
```

**Note** : L'extraction OCR est maintenant effectuée par le microservice OCR isolé via Dramatiq, permettant une meilleure scalabilité et isolation.

Voir [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) pour plus de détails.

## Installation

### 1. Installer Poetry

Si Poetry n'est pas déjà installé :

```bash
# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# Linux/Mac
curl -sSL https://install.python-poetry.org | python3 -
```

**Note** : Après l'installation, ajouter Poetry au PATH ou redémarrer le terminal. Voir la [documentation officielle](https://python-poetry.org/docs/#installation) pour plus de détails.

### 2. Initialiser Poetry (si nécessaire)

Si le projet n'a pas encore de fichier `pyproject.toml` configuré pour Poetry :

```bash
poetry init
```

⚠️ **Attention** : Si un fichier `pyproject.toml` existe déjà, `poetry init` peut le modifier ou vous demander de le faire manuellement. Dans ce cas, il est préférable de configurer Poetry directement dans le `pyproject.toml` existant en ajoutant une section `[tool.poetry]` plutôt que d'utiliser `poetry init`.

### 3. Installer les dépendances

```bash
poetry install
```

Poetry créera automatiquement un environnement virtuel et installera toutes les dépendances définies dans `pyproject.toml`.

**Note** : Plus besoin de créer manuellement un environnement virtuel (`venv`) - Poetry le gère automatiquement. Vous pouvez :
- Utiliser `poetry run` avant chaque commande (recommandé)
- Ou activer l'environnement avec `poetry shell` pour une session interactive

**Note sur les PDFs** : Le pipeline utilise PyMuPDF par défaut pour convertir les PDFs (aucune dépendance externe requise). Si vous préférez utiliser `pdf2image`, vous devrez installer poppler :
- **Windows** : Télécharger [poppler](https://github.com/oschwartz10612/poppler-windows/releases/) et l'ajouter au PATH
- **Linux** : `sudo apt-get install poppler-utils`
- **Mac** : `brew install poppler`

**Note sur le microservice OCR** : Le microservice OCR (`services/ocr_service/`) a ses propres dépendances isolées. Ces dépendances seront gérées automatiquement par Docker lors de la construction de l'image (voir la section [Lancement en Développement](#lancement-en-développement-approche-hybride)).

### 4. Installer Docker et Docker Compose

Docker et Docker Compose sont requis pour lancer les services de fond (Redis et le worker OCR) dans un environnement isolé.

#### Windows

1. Télécharger et installer [Docker Desktop pour Windows](https://www.docker.com/products/docker-desktop/)
2. Vérifier l'installation :
   ```powershell
   docker --version
   docker-compose --version
   ```

#### Linux

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker.io docker-compose

# Démarrer Docker
sudo systemctl start docker
sudo systemctl enable docker

# Vérifier l'installation
docker --version
docker-compose --version
```

#### Mac

```bash
# Installer via Homebrew
brew install docker docker-compose

# Ou télécharger Docker Desktop pour Mac
# https://www.docker.com/products/docker-desktop/
```

**Note** : Redis sera lancé automatiquement via Docker Compose (voir la section [Lancement en Développement](#lancement-en-développement-approche-hybride)). Plus besoin d'installer Redis manuellement !

### 5. Configuration

#### Variables d'environnement

Copier `env.example` vers `.env` et configurer les variables d'environnement si nécessaire :

```bash
# Windows
copy env.example .env

# Linux/Mac
cp env.example .env
```

**Variables importantes** :

- **Dramatiq Workers** :
  - `DRAMATIQ_PROCESSES` : Nombre de processus workers (défaut: 4)
  - `DRAMATIQ_THREADS` : Nombre de threads par processus (défaut: 1)

- **Chemins et Modèles** (configurables via variables d'environnement) :
  - `CLASSIFICATION_MODEL_PATH` : Chemin vers le modèle de classification (défaut: `training_data/artifacts/document_classifier.joblib`)
  - `CLASSIFICATION_EMBEDDING_MODEL` : Modèle sentence-transformers pour les embeddings (défaut: `antoinelouis/french-me5-base`)
  - `TEMP_STORAGE_DIR` : Répertoire de stockage temporaire (défaut: `data/temp_storage`)
  - `TRAINING_RAW_DIR` : Répertoire des données brutes d'entraînement (défaut: `training_data/raw`)
  - `TRAINING_PROCESSED_DIR` : Répertoire des données préparées (défaut: `training_data/processed`)
  - `TRAINING_ARTIFACTS_DIR` : Répertoire des modèles et rapports (défaut: `training_data/artifacts`)

- **Performance** :
  - `PERFORMANCE_BATCH_SIZE` : Taille des lots pour le traitement (défaut: 10)
  - `PERFORMANCE_MAX_WORKERS` : Nombre maximum de workers parallèles (défaut: 4)

**Note** : En production, ces variables d'environnement sont prioritaires sur les valeurs du `config.yaml`. Cela permet de configurer l'application sans reconstruire l'image Docker. Les chemins et identifiants de modèles peuvent être modifiés via les variables d'environnement sans reconstruire l'image.

Voir la section [Démarrer les workers Dramatiq](#démarrer-les-workers-dramatiq-requis-pour-la-classification) pour plus de détails sur la configuration recommandée par environnement.

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

# Configuration du microservice OCR
ocr_service:
  queue_name: 'ocr-queue'  # Queue Dramatiq pour le microservice
  timeout_ms: 30000         # Timeout en millisecondes (30 secondes)
  max_retries: 3            # Nombre de tentatives en cas d'échec
```

**Configuration du microservice OCR** : Le microservice OCR a sa propre configuration dans `services/ocr_service/config.yaml` (langue, GPU, MKLDNN, etc.). Cette configuration est totalement autonome et ne dépend pas du `config.yaml` principal.

**Dead Letter Queue (DLQ)** : Les tâches qui échouent après avoir atteint leur nombre maximum de tentatives sont automatiquement envoyées vers la Dead Letter Queue. La durée de vie des messages dans la DLQ est configurée dans `config.yaml` (`dramatiq.dead_message_ttl`, défaut: 7 jours). Utilisez les endpoints `/api/v1/dlq/*` pour inspecter et gérer les messages échoués.

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
poetry run python -m src.cli.main pipeline --input data/input --output data/output
```

3. **Ou exécuter les étapes séparément** :
```bash
# Étape 1: Prétraitement (amélioration contraste + classification)
poetry run python -m src.cli.main preprocessing --input data/input --output data/processed/preprocessing

# Étape 2: Normalisation géométrique (crop, deskew, rotation)
poetry run python -m src.cli.main geometry --input data/processed/preprocessing --output data/output/geometry
```

4. **Ou utiliser le script de test** :
```bash
poetry run python test_pdf_example.py
```

**Note** : Pour les PDFs multi-pages, chaque page sera traitée et sauvegardée avec le suffixe `_page1.png`, `_page2.png`, etc.

Voir `docs/TESTING.md` pour plus de détails.

### Ligne de commande

Exécuter une étape spécifique du pipeline :

```bash
# Étape 1: Prétraitement (amélioration contraste + classification SCAN/PHOTO)
poetry run python -m src.cli.main preprocessing --input data/input/ --output data/processed/preprocessing/

# Étape 2: Normalisation géométrie (crop, deskew, rotation)
poetry run python -m src.cli.main geometry --input data/processed/preprocessing/ --output data/output/geometry/

# Normalisation colométrie (optionnel)
poetry run python -m src.cli.main colometry --input data/input/ --output data/processed/colometry/

# Extraction de features
poetry run python -m src.cli.main features --input data/output/geometry/ --output data/output/

# Exécuter tout le pipeline (prétraitement → géométrie → features)
poetry run python -m src.cli.main pipeline --input data/input/ --output data/output/

# Exécuter des étapes spécifiques du pipeline
poetry run python -m src.cli.main pipeline --input data/input/ --output data/output/ --stages preprocessing --stages geometry
```

**Important** : L'étape `geometry` attend maintenant la sortie de `preprocessing` en entrée. Les images doivent être prétraitées avec leurs métadonnées JSON correspondantes.

### Lancement en Développement (Approche Hybride)

Cette méthode offre le meilleur des deux mondes : les services lourds (Redis, OCR) tournent dans Docker de manière isolée, tandis que l'application principale s'exécute localement pour un développement rapide.

**Prérequis** :
- Docker et Docker Compose installés (voir section [Installation](#4-installer-docker-et-docker-compose))
- Poetry et Python 3.11 installés sur ta machine locale

#### Étape 1 : Lancer les services de fond (une seule fois)

Ouvre un terminal et lance les conteneurs pour Redis et le worker OCR. Ils tourneront en arrière-plan.

```bash
# Cette commande va construire l'image du worker OCR (la première fois) et lancer les services.
docker-compose up --build -d
```

*Pour voir les logs des services : `docker-compose logs -f`*

*Pour arrêter les services : `docker-compose down`*

#### Étape 2 : Installer les dépendances locales (une seule fois)

Dans un autre terminal, à la racine du projet, installe les dépendances de l'application principale.

```bash
poetry install
```

#### Étape 3 : Lancer l'application principale localement

Tu as maintenant besoin de deux terminaux pour ton développement quotidien :

**Terminal 1 : Lancer l'API FastAPI**

```bash
# Le --reload permet de redémarrer automatiquement à chaque modification du code
poetry run uvicorn src.api.app:app --reload
```

**Terminal 2 : Lancer les workers de classification**

```bash
poetry run dramatiq src.workers
```

C'est tout ! Ton API locale sur `http://localhost:8000` peut maintenant communiquer avec le worker OCR et Redis qui tournent dans Docker.

**Documentation interactive de l'API** :
- **Swagger UI** : `http://localhost:8000/docs`
- **ReDoc** : `http://localhost:8000/redoc`

#### Résumé des Avantages de cette Approche

- **Développement Rapide** : Tu modifies le code de `src/` et Uvicorn recharge instantanément. Pas de `docker build`.
- **Stabilité** : Les dépendances complexes du service OCR sont encapsulées dans une image Docker. Fini les problèmes d'installation de PaddleOCR sur ta machine. Redis est également géré.
- **Simplicité** : Une seule commande (`docker-compose up`) pour lancer tous les services. Puis tu travailles localement comme d'habitude.
- **Performance** : Ton code local s'exécute nativement, sans la surcouche de Docker, ce qui peut être légèrement plus rapide pour le débogage et les tests.

**Note importante** : Le code source du service OCR (`services/ocr_service/`) est monté comme volume dans le conteneur Docker. Si tu modifies `services/ocr_service/actors.py`, le worker redémarrera automatiquement grâce à l'option `--watch` de Dramatiq, sans avoir à recompiler l'image.

**Healthchecks** : Les services Docker incluent des healthchecks pour garantir leur disponibilité :
- **Redis** : Vérifie la connexion avec `redis-cli ping`
- **OCR Worker** : Vérifie que le moteur PaddleOCR est initialisé via le script `healthcheck.py`

### API

#### Endpoints disponibles

- `POST /api/v1/analyze` : Analyser un document (pipeline complet)
- `POST /api/v1/pipeline/colometry` : Normalisation colométrie
- `POST /api/v1/pipeline/geometry` : Normalisation géométrie
- `POST /api/v1/pipeline/features` : Extraction de features (nécessite le microservice OCR)
- `POST /api/v1/classify` : Classification de document (asynchrone, nécessite les workers Dramatiq)
- `GET /api/v1/classify/results/{task_id}` : Récupérer les résultats d'une classification
- `GET /api/v1/pipeline/status` : Statut du pipeline
- `GET /api/v1/results/{task_id}` : Statut et résultats d'une tâche asynchrone
- `GET /api/v1/ocr/health` : Health check du microservice OCR (vérifie que le service est prêt)
- `POST /api/v1/warmup` : Warm-up des workers (pré-initialise les modèles)
- `GET /metrics` : Métriques Prometheus de l'API (format Prometheus, disponible en production)
- `GET /api/v1/dlq/messages` : Liste les messages dans la Dead Letter Queue (tâches échouées)
- `GET /api/v1/dlq/statistics` : Statistiques sur la Dead Letter Queue
- `POST /api/v1/dlq/replay/{message_id}` : Rejoue un message depuis la DLQ
- `DELETE /api/v1/dlq/messages/{message_id}` : Supprime un message de la DLQ
- `DELETE /api/v1/dlq/clear` : Vide complètement la Dead Letter Queue

#### Exemple d'utilisation de l'endpoint de classification

1. **Démarrer Redis** (si pas déjà fait)
2. **Démarrer les workers Dramatiq** dans un terminal séparé
3. **Démarrer l'API** dans un autre terminal
4. **Soumettre une tâche de classification** :

```bash
curl -X POST "http://localhost:8000/api/v1/classify" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

La réponse contiendra un `task_id` :

```json
{
  "task_id": "abc123...",
  "status": "pending",
  "message": "La tâche de classification a été créée..."
}
```

5. **Récupérer les résultats** :

```bash
curl "http://localhost:8000/api/v1/classify/results/{task_id}"
```

Si la tâche est terminée, vous recevrez les résultats de classification. Sinon, le statut sera `"pending"`.

#### Exemple d'utilisation avec le microservice OCR

Pour utiliser l'extraction OCR (endpoints `/api/v1/pipeline/features` et `/api/v1/analyze`), assure-toi que les services Docker sont lancés (voir [Étape 1](#étape-1--lancer-les-services-de-fond-une-seule-fois)).

1. **Vérifier que les services Docker sont lancés** :
   ```bash
   docker-compose ps
   ```
   Tu devrais voir `redis` et `ocr-worker` en cours d'exécution.

2. **Démarrer l'API** (si pas déjà fait) :
   ```bash
   poetry run uvicorn src.api.app:app --reload
   ```

3. **Soumettre une tâche d'extraction de features** :

```bash
curl -X POST "http://localhost:8000/api/v1/pipeline/features" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.png"
```

L'API utilisera automatiquement le microservice OCR (qui tourne dans Docker) pour extraire le texte de l'image.

### Déploiement en Production

Le projet inclut une configuration Docker optimisée pour la production avec `docker-compose.prod.yml`.

#### Prérequis

- Docker et Docker Compose installés
- Les modèles ML dans le répertoire `models/`
- Les données dans les répertoires `data/input/`, `data/output/`, etc.

#### Build des images

```bash
# Builder l'image de l'application principale (API + Workers)
docker build -t doc-analysis-api:latest .

# Builder l'image du service OCR (déjà fait via docker-compose)
docker-compose build ocr-worker
```

**Optimisations Docker** :
- **Build multi-stage** : Les dépendances sont compilées dans un stage séparé, l'image finale ne contient que le runtime
- **Versions pinnées** : Les images de base utilisent des versions spécifiques (`python:3.11.9-slim`, `redis:7.2-alpine`) pour garantir la reproductibilité
- **Paquets -dev exclus** : Seules les librairies d'exécution sont incluses dans l'image finale (pas les en-têtes de compilation)
- **Utilisateur non-root** : Les services s'exécutent avec un utilisateur non-privilégié pour la sécurité

#### Lancement en production

```bash
# Lancer tous les services en production
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

Cette commande lance :
- **Redis** : Broker de messages avec optimisation mémoire (512MB max)
- **OCR Worker** : Service OCR optimisé (4 processus, 2 threads, sans `--watch`)
- **API** : FastAPI avec 4 workers Uvicorn et healthcheck
- **Workers** : Workers Dramatiq pour la classification (4 processus, 2 threads)
- **Prometheus** : Collecte et stockage des métriques (port 9091)
- **Grafana** : Visualisation des métriques (port 3000, admin/admin)

#### Différences entre Développement et Production

| Aspect | Développement | Production |
|--------|---------------|------------|
| **Code source** | Monté comme volume (modifications en temps réel) | Intégré dans l'image Docker |
| **OCR Worker** | Utilise `--watch` pour rechargement automatique | Sans `--watch` (plus stable) |
| **Dépendances OCR** | `poetry install` (avec dev) | `poetry install --no-dev` (optimisé) |
| **API** | 1 worker Uvicorn avec `--reload` | 4 workers Uvicorn sans reload |
| **Workers** | Lancés localement avec Poetry | Conteneurisés avec Docker |
| **Healthchecks** | Présents pour monitoring | Présents + optimisés |
| **Logging** | stdout/stderr (pas de rotation) | Rotation automatique (10 Mo, 5 fichiers) |
| **Monitoring** | Non disponible | Prometheus + Grafana inclus |
| **Images Docker** | Versions génériques | Versions pinnées (reproductibilité) |

#### Vérifier le statut des services

```bash
# Voir le statut de tous les services
docker-compose -f docker-compose.yml -f docker-compose.prod.yml ps

# Voir les logs
docker-compose -f docker-compose.yml -f docker-compose.prod.yml logs -f

# Voir les logs d'un service spécifique
docker-compose -f docker-compose.yml -f docker-compose.prod.yml logs -f api
docker-compose -f docker-compose.yml -f docker-compose.prod.yml logs -f ocr-worker
```

#### Healthchecks

Les services incluent des healthchecks pour garantir leur disponibilité :

- **Redis** : `redis-cli ping` (intervalle: 10s)
- **OCR Worker** : Script `healthcheck.py` qui vérifie l'initialisation du moteur PaddleOCR (intervalle: 30s, timeout: 20s)
- **API** : Endpoint `/health` (intervalle: 30s, timeout: 10s)
- **Workers** : Endpoint `/health` sur le port 9090 (intervalle: 30s, timeout: 10s)

Les healthchecks permettent à Docker de redémarrer automatiquement les services en cas de problème.

#### Logging

Les services utilisent une configuration de logging avec rotation automatique pour éviter de saturer le disque :

- **Driver** : `json-file` (par défaut Docker)
- **Taille max par fichier** : 10 Mo
- **Nombre de fichiers conservés** : 5 (rotation automatique)
- **Espace disque maximum** : ~50 Mo par service

Les logs sont accessibles via `docker-compose logs` ou directement dans `/var/lib/docker/containers/` sur le host.

**Note** : Pour une vraie production, envisagez un système de logging centralisé (fluentd, splunk, gelf) pour collecter et analyser les logs de tous les services en un seul endroit.

#### Monitoring

Le déploiement en production inclut Prometheus et Grafana pour le monitoring :

- **Prometheus** : Collecte les métriques depuis l'API (`/metrics`) et les workers (port 9090)
  - Interface web : `http://localhost:9091`
  - Configuration : `monitoring/prometheus/prometheus.yml`
  
- **Grafana** : Visualisation des métriques avec dashboards pré-configurés
  - Interface web : `http://localhost:3000`
  - Identifiants par défaut : `admin` / `admin`
  - Dashboards : `monitoring/grafana/dashboards/`

Voir [`monitoring/README.md`](monitoring/README.md) pour plus de détails sur la configuration du monitoring.

#### Arrêter les services

```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml down
```

Pour supprimer aussi les volumes (⚠️ supprime les données) :
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml down -v
```

## Structure du projet

```
doc_analysis/
├── src/
│   ├── pipeline/          # Étapes du pipeline
│   │   ├── preprocessing.py  # Prétraitement (contraste + classification)
│   │   ├── geometry.py       # Normalisation géométrique
│   │   ├── colometry.py      # Normalisation colométrie
│   │   └── features.py       # Extraction de features (utilise le microservice OCR)
│   ├── api/               # API REST
│   ├── cli/               # Interface ligne de commande
│   ├── utils/             # Utilitaires
│   │   ├── ocr_client.py  # Client pour communiquer avec le microservice OCR
│   │   ├── dlq_manager.py # Gestion de la Dead Letter Queue
│   │   └── storage.py     # Gestion du stockage temporaire pour les workers
│   └── models/            # Modèles ML
├── services/              # Microservices isolés
│   └── ocr_service/       # Microservice OCR isolé
│       ├── actors.py      # Workers Dramatiq pour l'OCR
│       ├── config.yaml    # Configuration autonome du service
│       ├── pyproject.toml # Dépendances isolées
│       ├── Dockerfile     # Image Docker du service OCR
│       ├── run_worker.sh  # Script pour lancer le service (dev)
│       └── healthcheck.py # Script de healthcheck pour Docker
├── Dockerfile             # Image Docker multi-stage pour l'application principale
├── docker-compose.yml     # Configuration Docker pour le développement
├── docker-compose.prod.yml # Configuration Docker pour la production
├── .dockerignore          # Fichiers exclus de l'image Docker
├── monitoring/            # Configuration du monitoring
│   ├── prometheus/        # Configuration Prometheus
│   └── grafana/          # Configuration Grafana (dashboards, datasources)
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
poetry run python qa_report.py --output-dir data/output --output qa_report.html
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
poetry run pytest tests/
```

### Formatage du code

```bash
poetry run black src/
poetry run flake8 src/
```

## Licence

MIT

