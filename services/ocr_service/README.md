# Service OCR Isolé

Ce service OCR est isolé du reste du monorepo `doc_analysis`. Il se concentre uniquement sur l'extraction de texte via PaddleOCR.

## Structure

```
services/ocr_service/
├── actors.py          # Workers Dramatiq pour l'OCR (isolé)
├── config.yaml        # Configuration minimaliste pour PaddleOCR
├── pyproject.toml     # Dépendances isolées (clé de l'isolation)
├── run_worker.sh      # Script pour lancer le service
└── README.md          # Ce fichier
```

## Installation

### Avec Poetry (recommandé)

```bash
cd services/ocr_service
poetry install
```

### Sans Poetry

Assurez-vous que les dépendances listées dans `pyproject.toml` sont installées dans votre environnement Python.

## Configuration

Éditez `config.yaml` pour configurer :
- La langue OCR (`default_language`)
- L'utilisation du GPU (`use_gpu`)
- Les optimisations MKLDNN (`enable_mkldnn`)
- Les paramètres Dramatiq (retries, timeouts, etc.)

## Lancement du Worker

### Sur Linux/macOS

```bash
# Configuration par défaut (1 processus, 1 thread)
./run_worker.sh

# Avec configuration personnalisée via variables d'environnement
export OCR_WORKER_PROCESSES=4
export OCR_WORKER_THREADS=2
./run_worker.sh
```

### Sur Windows

**PowerShell** :

```powershell
# Configuration par défaut
poetry run python -m dramatiq actors --processes 1 --threads 1 --queues ocr-queue

# Avec configuration personnalisée
$env:OCR_WORKER_PROCESSES = 4
$env:OCR_WORKER_THREADS = 2
poetry run python -m dramatiq actors --processes $env:OCR_WORKER_PROCESSES --threads $env:OCR_WORKER_THREADS --queues ocr-queue
```

**CMD** :

```cmd
set OCR_WORKER_PROCESSES=4
set OCR_WORKER_THREADS=2
poetry run python -m dramatiq actors --processes %OCR_WORKER_PROCESSES% --threads %OCR_WORKER_THREADS% --queues ocr-queue
```

### Configuration via Variables d'Environnement

Le script `run_worker.sh` et les commandes manuelles supportent les variables d'environnement suivantes :

- `OCR_WORKER_PROCESSES` : Nombre de processus workers (défaut: 1)
- `OCR_WORKER_THREADS` : Nombre de threads par processus (défaut: 1)

**Recommandations par environnement** :

- **Développement local** : `OCR_WORKER_PROCESSES=1`, `OCR_WORKER_THREADS=1` (pour économiser les ressources)
- **Staging** : `OCR_WORKER_PROCESSES=2`, `OCR_WORKER_THREADS=2` (équilibre performance/ressources)
- **Production** : `OCR_WORKER_PROCESSES=4`, `OCR_WORKER_THREADS=2` (maximum de débit, ajuster selon les ressources CPU disponibles)

**Note** : Le nombre total de workers OCR = `OCR_WORKER_PROCESSES × OCR_WORKER_THREADS`. Par exemple, avec `OCR_WORKER_PROCESSES=4` et `OCR_WORKER_THREADS=2`, vous aurez 8 workers OCR au total.

## Utilisation

Le service expose un acteur Dramatiq `perform_ocr_task` qui peut être appelé depuis d'autres services :

```python
from services.ocr_service.actors import perform_ocr_task

# Envoyer une tâche OCR
# L'image doit être sauvegardée temporairement par l'application principale
result = perform_ocr_task.send("file:///data/temp_storage/image-xyz.png", page_index=0)
```

## Isolation

L'isolation est garantie par :
1. **pyproject.toml** : Dépendances séparées du projet principal
2. **config.yaml** : Configuration dédiée au service OCR
3. **actors.py** : Code isolé sans dépendances vers le reste du monorepo

## Notes

- Le service nécessite Redis pour fonctionner (broker Dramatiq)
- Assurez-vous que Redis est démarré avant de lancer le worker
- Le worker écoute la queue `ocr-queue` (dédiée et isolée)
- L'acteur `perform_ocr_task` accepte uniquement des URIs de fichiers (pas de données brutes)
- Le modèle PaddleOCR est initialisé une seule fois par processus worker (variable globale) pour optimiser les performances

