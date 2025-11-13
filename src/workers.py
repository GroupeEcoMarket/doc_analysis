# src/workers.py

import os
import time
import uuid
import json
import threading
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor
from filelock import FileLock

import dramatiq
import numpy as np
import cv2
from dramatiq.brokers.redis import RedisBroker
from dramatiq.results import Results
from dramatiq.results.backends import RedisBackend

# --- Bootstrap des workers ---
# Il n'est plus nécessaire de l'appeler au niveau du module car chaque processus worker
# l'appellera lors de son initialisation, garantissant que l'environnement est configuré
# avant que les modèles ne soient importés.

# --- Imports des modules du projet ---
from src.config.manager import get_config_manager
from src.utils.pdf_handler import pdf_buffer_to_images, is_pdf
from src.utils.logger import get_logger
from src.pipeline.features import FeatureExtractor
from src.classification.classifier_service import DocumentClassifier
from src.pipeline.models import PageClassificationResult
from src.utils.exceptions import FeatureExtractionError
from src.utils.metrics import (
    pages_processed_total,
    document_processing_duration_seconds,
    page_processing_duration_seconds,
    documents_processed_total,
    processing_errors_total,
    init_queue_monitor,
    start_metrics_server
)

# --- Logger ---
logger = get_logger(__name__)

# --- Configuration de Dramatiq ---
# Connexion à Redis (qui tourne sur localhost par défaut)
# Configuration de la Dead Letter Queue (DLQ) pour capturer les tâches échouées
# Dramatiq gère automatiquement la DLQ avec le nom 'dramatiq:dead_letter'
# On peut configurer la durée de vie des messages dans la DLQ via dead_message_ttl
try:
    config = get_config_manager()
    # Durée de vie des messages dans la DLQ en millisecondes (défaut: 7 jours)
    dead_message_ttl = config.get('dramatiq', 'dead_message_ttl', default=604800000)  # 7 jours
except:
    dead_message_ttl = 604800000  # 7 jours par défaut

# Configuration Redis depuis les variables d'environnement
try:
    config = get_config_manager()
    redis_host = config.get('env', 'redis', 'host', default='localhost')
    redis_port = config.get('env', 'redis', 'port', default=6379)
    redis_url = f"redis://{redis_host}:{redis_port}"
    redis_broker = RedisBroker(url=redis_url, dead_message_ttl=dead_message_ttl)
    result_backend = RedisBackend(url=redis_url)
except Exception as e:
    logger.warning(f"Erreur lors de la configuration Redis, utilisation des valeurs par défaut: {e}")
    # Fallback vers les valeurs par défaut
    redis_broker = RedisBroker(dead_message_ttl=dead_message_ttl)
    result_backend = RedisBackend()

# Ajouter un backend de résultats Redis pour stocker les résultats des tâches
redis_broker.add_middleware(Results(backend=result_backend))
dramatiq.set_broker(redis_broker)

logger.info(f"Broker Dramatiq configuré avec Dead Letter Queue (TTL: {dead_message_ttl}ms)")

# Initialiser le moniteur de taille de file d'attente
try:
    config = get_config_manager()
    queue_monitor_interval = config.get('metrics', 'queue_monitor_interval', default=10.0)
    init_queue_monitor(redis_broker, update_interval=queue_monitor_interval)
except Exception as e:
    logger.warning(f"Impossible d'initialiser le moniteur de file d'attente: {e}")

# Démarrer le serveur de métriques pour les workers
try:
    config = get_config_manager()
    metrics_port = config.get('metrics', 'workers_port', default=9090)
    metrics_host = config.get('metrics', 'workers_host', default='0.0.0.0')
    start_metrics_server(port=metrics_port, host=metrics_host)
except Exception as e:
    logger.warning(f"Impossible de démarrer le serveur de métriques: {e}")

# --- Variables globales pour les modèles ---
# Ces variables seront initialisées UNE SEULE FOIS par processus (worker du pool ou processus principal),
# ce qui est extrêmement performant.
_feature_extractor: FeatureExtractor = None
_document_classifier: DocumentClassifier = None

# --- Verrous pour l'initialisation thread-safe et inter-processus ---
# _init_lock: Verrou thread-safe pour garantir qu'un seul thread à la fois peut initialiser
#             les modèles dans un même processus, évitant les race conditions lors de
#             l'initialisation de PaddleOCR et PyTorch qui ne sont pas thread-safe.
# _process_lock: Verrou inter-processus (FileLock) pour garantir qu'un seul processus
#                à la fois sur toute la machine peut initialiser les modèles.
#                Cela évite les conflits lors de l'initialisation simultanée par plusieurs
#                workers Dramatiq dans différents processus.
_init_lock = threading.Lock()

# Créer un fichier de verrou temporaire pour la synchronisation inter-processus
config = get_config_manager()
temp_storage_path_str = config.get('env', 'paths', 'temp', default='data/temp_storage')
lock_dir = Path(temp_storage_path_str).resolve()
lock_dir.mkdir(parents=True, exist_ok=True)
_lock_file_path = lock_dir / ".worker_init.lock"

# Récupérer le timeout depuis la configuration (défaut: 60 secondes)
lock_timeout = config.get('dramatiq', 'worker_init_lock_timeout', default=60)
_process_lock = FileLock(_lock_file_path, timeout=lock_timeout)

def init_worker():
    """
    Initialise les modèles lourds une seule fois par processus.
    
    Cette fonction orchestre l'initialisation en utilisant les fonctions centralisées
    de src.utils.worker_init. Elle est utilisée :
    - Par ProcessPoolExecutor (appelée automatiquement lors de la création de chaque worker)
    - Par le processus principal (appelée manuellement pour le traitement séquentiel)
    - Par les workers Dramatiq (appelée lors de l'initialisation de chaque worker)
    
    IMPORTANT: Cette fonction est thread-safe ET inter-processus-safe grâce à deux verrous :
    1. _process_lock (FileLock) : Garantit qu'un seul processus à la fois sur toute la machine
       peut initialiser les modèles. Les autres processus attendent.
    2. _init_lock (threading.Lock) : Garantit qu'un seul thread à la fois dans un même processus
       peut initialiser les modèles. Cela évite les race conditions lors de l'initialisation
       de PaddleOCR et PyTorch qui ne sont pas thread-safe.
    """
    global _feature_extractor, _document_classifier
    
    # Vérification rapide : si les modèles sont déjà chargés dans ce processus, on sort immédiatement
    # C'est très performant et évite les acquisitions de verrous inutiles
    if _feature_extractor is not None:
        return _feature_extractor, _document_classifier
    
    # Acquérir le verrou inter-processus : seul UN processus à la fois sur toute la machine
    # peut entrer dans ce bloc. Les autres processus attendront que celui-ci ait fini.
    with _process_lock:
        # Double-check : une fois le verrou de processus obtenu, on revérifie.
        # Il est possible qu'un autre processus ait déjà tout initialisé pendant que celui-ci attendait.
        # Dans ce cas, cette condition ne serait plus vraie pour les processus suivants.
        # C'est une optimisation importante.
        if _feature_extractor is None:
            # On garde le threading.Lock par sécurité pour les threads à l'intérieur d'un même processus
            # (même si avec --threads 1 ce n'est pas strictement nécessaire, c'est une bonne pratique)
            with _init_lock:
                # Vérifier une dernière fois après l'acquisition du verrou de thread
                # (un autre thread dans le même processus a peut-être initialisé pendant l'attente)
                if _feature_extractor is None:
                    print(f"[Worker PID: {os.getpid()}] Initialisation des dépendances...")
                    
                    # Utiliser la fonction centralisée d'initialisation
                    from src.utils.worker_init import init_classification_worker
                    
                    config = get_config_manager()
                    _feature_extractor, _document_classifier = init_classification_worker(config)
                    
                    print(f"[Worker PID: {os.getpid()}] Dépendances prêtes.")
    
    return _feature_extractor, _document_classifier

def process_single_page(page_tuple: Tuple[int, np.ndarray]) -> Tuple[int, Dict[str, Any]]:
    """
    Traite une seule page dans un processus.
    
    Cette fonction peut être exécutée :
    - Par chaque worker du ProcessPoolExecutor (traitement parallèle)
    - Par le processus principal (traitement séquentiel d'une seule page ou fallback)
    
    Utilise les modèles initialisés dans init_worker.
    Note: init_worker doit être appelé avant cette fonction (via initializer du pool ou manuellement).
    
    Args:
        page_tuple: Tuple (page_index, image_np) où:
                   - page_index: Index de la page (0-based)
                   - image_np: Image numpy de la page
    
    Returns:
        Tuple[int, Dict[str, Any]]: (page_index, classification_result)
    """
    global _feature_extractor, _document_classifier
    
    page_index, image_np = page_tuple
    
    feature_extractor = _feature_extractor
    document_classifier = _document_classifier
    
    if document_classifier is None:
        return page_index, {
            'document_type': None,
            'confidence': 0.0,
            'error': "Le service de classification n'est pas activé ou configuré."
        }
    
    # Traiter la page
    try:
        ocr_lines = feature_extractor.extract_ocr(image_np)
        ocr_data = {'ocr_lines': ocr_lines}
        classification_result = document_classifier.predict(ocr_data)
        return page_index, classification_result
    
    except FeatureExtractionError as e:
        print(f"[Worker PID: {os.getpid()}] Erreur OCR sur la page {page_index + 1}: {e}")
        # Retourner un résultat d'erreur clair pour cette page
        return page_index, {
            'document_type': None,
            'confidence': 0.0,
            'error': f"OCR processing failed: {e}"
        }
    except Exception as e:
        print(f"[Worker PID: {os.getpid()}] Erreur inattendue sur la page {page_index + 1}: {e}")
        return page_index, {
            'document_type': None,
            'confidence': 0.0,
            'error': f"An unexpected error occurred: {e}"
        }

# --- Configuration des retries depuis la config (défaut: 3) ---
# Doit être défini avant les acteurs Dramatiq
try:
    config = get_config_manager()
    max_retries = config.get('dramatiq', 'default_max_retries', default=3)
    time_limit = config.get('dramatiq', 'default_time_limit', default=300_000)
except:
    max_retries = 3
    time_limit = 300_000

# --- Nouvel Actor Dramatiq : Page Worker ---

@dramatiq.actor(time_limit=time_limit, max_retries=0, store_results=False)
def warmup_worker_task(**kwargs) -> Dict[str, Any]:
    """
    Tâche de "warm-up" pour pré-initialiser les workers.
    
    Cette tâche ne fait rien d'utile mais force l'initialisation des modèles
    (PaddleOCR, PyTorch, etc.) dans le worker qui l'exécute. C'est particulièrement
    utile en production avec auto-scaling pour "chauffer" les nouveaux workers avant
    qu'ils ne reçoivent de vraies tâches.
    
    Args:
        **kwargs: Arguments additionnels passés par Dramatiq (options, etc.)
            Acceptés pour compatibilité avec les futures évolutions du framework.
    
    Returns:
        Dict[str, Any]: Statut de l'initialisation
            Format: {
                'status': 'ready',
                'worker_pid': int,
                'models_initialized': bool
            }
    """
    global _feature_extractor, _document_classifier
    
    # Initialiser explicitement les modèles en premier (thread-safe et inter-processus-safe)
    # Cela garantit que les modèles sont prêts avant toute autre opération
    init_worker()
    
    # Récupérer le request_id depuis les métadonnées du message pour le traçage
    from src.utils.logger import set_request_id
    try:
        message = dramatiq.get_current_message()
        if message and message.message_metadata:
            request_id = message.message_metadata.get('request_id')
            if request_id:
                set_request_id(request_id)
    except Exception:
        # Si on ne peut pas récupérer le message, continuer sans request_id
        pass
    
    worker_pid = os.getpid()
    
    # Vérifier que les modèles sont bien initialisés
    models_ready = _feature_extractor is not None and _document_classifier is not None
    
    logger.info(
        f"Worker warm-up terminé (PID: {worker_pid})",
        extra={
            "metrics": {
                "worker_pid": worker_pid,
                "models_initialized": models_ready,
                "status": "ready"
            }
        }
    )
    
    return {
        'status': 'ready',
        'worker_pid': worker_pid,
        'models_initialized': models_ready
    }

@dramatiq.actor(time_limit=time_limit, max_retries=max_retries, store_results=True)
def process_page_task(image_identifier: str, page_index: int = 0, document_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """
    L'Ouvrier Spécialisé : Tâche asynchrone Dramatiq pour traiter une seule page.
    
    Sa seule responsabilité est de traiter une seule page :
    1. Reçoit un identifiant d'image
    2. Appelle le service OCR (via sa propre tâche asynchrone perform_ocr_task)
    3. Une fois l'OCR obtenu, appelle le service de classification
    4. Stocke le résultat pour cette page dans Redis avec l'ID du document global et son numéro de page
    
    Le résultat est stocké dans Redis avec la clé : document:{document_id}:page:{page_index}
    Un set Redis document:{document_id}:pages est aussi maintenu pour suivre toutes les pages du document.
    
    Args:
        image_identifier: Identifiant de l'image (nom du fichier) dans le stockage
        page_index: Index de la page (0-based), optionnel, par défaut 0
        document_id: ID du document global (requis pour le stockage dans Redis)
        **kwargs: Arguments additionnels passés par Dramatiq (options, etc.)
    
    Returns:
        Dict[str, Any]: Résultat de la classification pour cette page, ou dictionnaire d'erreur
            Format de succès: {
                'page_index': int,
                'document_type': str,
                'confidence': float,
                'document_id': str,
                'timestamp': float
            }
            Format d'erreur: {
                'page_index': int,
                'document_type': None,
                'confidence': 0.0,
                'error': str,
                'document_id': str,
                'timestamp': float
            }
    """
    global _feature_extractor, _document_classifier
    
    # Initialiser explicitement les modèles en premier (thread-safe et inter-processus-safe)
    # Cela garantit que les modèles sont prêts avant toute autre opération
    init_worker()
    
    # Récupérer le request_id depuis les métadonnées du message pour le traçage
    from src.utils.logger import set_request_id
    try:
        message = dramatiq.get_current_message()
        if message and message.message_metadata:
            request_id = message.message_metadata.get('request_id')
            if request_id:
                set_request_id(request_id)
    except Exception:
        # Si on ne peut pas récupérer le message, continuer sans request_id
        pass
    
    page_start_time = time.time()
    worker_pid = os.getpid()
    
    # Log au début du traitement de la page
    logger.info(f"Début traitement page {page_index + 1} pour document_id: {document_id}.")
    
    # Vérifier que les modèles sont disponibles
    if _document_classifier is None:
        page_processing_time = time.time() - page_start_time
        logger.error(
            f"Service de classification non disponible pour la page {page_index + 1}",
            extra={
                "metrics": {
                    "page_index": page_index,
                    "page_number": page_index + 1,
                    "processing_time_seconds": round(page_processing_time, 3),
                    "status": "error",
                    "error_type": "classifier_not_available",
                    "worker_pid": worker_pid
                }
            }
        )
        error_result = {
            'page_index': page_index,
            'document_type': None,
            'confidence': 0.0,
            'error': "Le service de classification n'est pas activé ou configuré.",
            'document_id': document_id,
            'timestamp': time.time()
        }
        
        # Stocker aussi les erreurs dans Redis
        if document_id:
            try:
                redis_client = redis_broker.client
                redis_key = f"document:{document_id}:page:{page_index}"
                result_json = json.dumps(error_result)
                redis_client.setex(redis_key, 86400, result_json)
                
                document_pages_key = f"document:{document_id}:pages"
                redis_client.sadd(document_pages_key, page_index)
                redis_client.expire(document_pages_key, 86400)
            except Exception:
                pass
        
        # Métriques Prometheus pour les erreurs
        pages_processed_total.labels(status='failed', doc_type='Unknown').inc()
        page_processing_duration_seconds.observe(page_processing_time)
        processing_errors_total.labels(error_type='classifier_not_available').inc()
        
        return error_result
    
    try:
        # === ÉTAPE 1 : Effectuer l'OCR via le microservice OCR ===
        # Le service OCR charge lui-même l'image depuis le stockage via l'identifiant
        from src.utils.ocr_client import perform_ocr_task
        from dramatiq.results.errors import ResultTimeout
        
        # Récupérer le timeout depuis la configuration
        try:
            config = get_config_manager()
            ocr_timeout_ms = config.get('ocr_service', 'timeout_ms', default=30000)  # 30 secondes par défaut
        except:
            ocr_timeout_ms = 30000  # 30 secondes par défaut
        
        ocr_start_time = time.time()
        # Appeler le service OCR via sa propre tâche asynchrone
        ocr_message = perform_ocr_task.send(image_identifier, page_index)
        try:
            ocr_result = ocr_message.get_result(block=True, timeout=ocr_timeout_ms)
        except ResultTimeout:
            ocr_time = time.time() - ocr_start_time
            error_msg = f"Timeout OCR après {ocr_timeout_ms/1000:.1f}s - Le service OCR n'a pas répondu à temps"
            logger.error(
                f"Timeout OCR pour la page {page_index + 1}: {error_msg}",
                extra={
                    "metrics": {
                        "page_index": page_index,
                        "ocr_time_seconds": round(ocr_time, 3),
                        "status": "error",
                        "error_type": "ocr_timeout",
                        "error_message": error_msg,
                        "worker_pid": worker_pid
                    }
                }
            )
            error_result = {
                'page_index': page_index,
                'document_type': None,
                'confidence': 0.0,
                'error': error_msg,
                'document_id': document_id,
                'timestamp': time.time()
            }
            
            # Métriques Prometheus
            processing_errors_total.labels(error_type='ocr_timeout').inc()
            
            return error_result
        
        ocr_time = time.time() - ocr_start_time
        
        # Log DEBUG pour la durée OCR
        logger.debug(f"Appel OCR pour page {page_index + 1} terminé en {ocr_time:.3f}s.")
        
        if ocr_result.get('status') != 'success':
            error_msg = ocr_result.get('error', 'Erreur OCR inconnue')
            page_processing_time = time.time() - page_start_time
            logger.error(
                f"Échec de l'OCR pour la page {page_index + 1}: {error_msg}",
                extra={
                    "metrics": {
                        "page_index": page_index,
                        "processing_time_seconds": round(page_processing_time, 3),
                        "ocr_time_seconds": round(ocr_time, 3),
                        "status": "error",
                        "error_type": "ocr_failed",
                        "error_message": error_msg,
                        "worker_pid": worker_pid
                    }
                }
            )
            error_result = {
                'page_index': page_index,
                'document_type': None,
                'confidence': 0.0,
                'error': f"OCR failed: {error_msg}",
                'document_id': document_id,
                'timestamp': time.time()
            }
            
            # Stocker aussi les erreurs dans Redis
            if document_id:
                try:
                    redis_client = redis_broker.client
                    redis_key = f"document:{document_id}:page:{page_index}"
                    result_json = json.dumps(error_result)
                    redis_client.setex(redis_key, 86400, result_json)
                    
                    # Ajouter la page au set même en cas d'erreur
                    document_pages_key = f"document:{document_id}:pages"
                    redis_client.sadd(document_pages_key, page_index)
                    redis_client.expire(document_pages_key, 86400)
                except Exception:
                    pass  # Ignorer les erreurs Redis pour les erreurs de traitement
            
            # Métriques Prometheus pour les erreurs OCR
            pages_processed_total.labels(status='failed', doc_type='Unknown').inc()
            page_processing_duration_seconds.observe(page_processing_time)
            processing_errors_total.labels(error_type='ocr_failed').inc()
            
            return error_result
        
        ocr_lines = ocr_result.get('ocr_lines', [])
        
        # === ÉTAPE 2 : Classifier le document avec les données OCR ===
        classification_start_time = time.time()
        ocr_data = {'ocr_lines': ocr_lines}
        classification_result = _document_classifier.predict(ocr_data)
        classification_time = time.time() - classification_start_time
        
        # Log DEBUG pour la durée de classification
        logger.debug(f"Classification pour page {page_index + 1} terminée en {classification_time:.3f}s.")
        
        page_processing_time = time.time() - page_start_time
        
        # Ajouter le page_index au résultat
        classification_result['page_index'] = page_index
        classification_result['document_id'] = document_id
        classification_result['timestamp'] = time.time()
        
        # === ÉTAPE 3 : Stocker le résultat dans Redis ===
        if document_id:
            try:
                # Utiliser le client Redis du broker
                redis_client = redis_broker.client
                
                # Clé Redis : document:{document_id}:page:{page_index}
                redis_key = f"document:{document_id}:page:{page_index}"
                
                # Stocker le résultat en JSON avec un TTL de 24 heures
                result_json = json.dumps(classification_result)
                redis_client.setex(redis_key, 86400, result_json)  # 24 heures = 86400 secondes
                
                # Ajouter aussi la page à un set pour suivre toutes les pages du document
                document_pages_key = f"document:{document_id}:pages"
                redis_client.sadd(document_pages_key, page_index)
                redis_client.expire(document_pages_key, 86400)  # TTL de 24 heures aussi
                
                logger.debug(
                    f"Résultat stocké dans Redis pour document {document_id}, page {page_index + 1}",
                    extra={
                        "metrics": {
                            "document_id": document_id,
                            "page_index": page_index,
                            "redis_key": redis_key
                        }
                    }
                )
            except Exception as redis_error:
                # Ne pas faire échouer la tâche si le stockage Redis échoue
                logger.warning(
                    f"Impossible de stocker le résultat dans Redis: {redis_error}",
                    exc_info=True,
                    extra={
                        "metrics": {
                            "document_id": document_id,
                            "page_index": page_index,
                            "error": str(redis_error)
                        }
                    }
                )
        
        # Log INFO final avec toutes les durées
        doc_type = classification_result.get('document_type', 'Unknown')
        confidence = classification_result.get('confidence', 0.0)
        logger.info(f"Fin traitement page {page_index + 1}. Résultat: '{doc_type}' ({confidence:.2f}). Durée totale: {page_processing_time:.3f}s (OCR: {ocr_time:.3f}s, Classification: {classification_time:.3f}s).")
        
        # Log structuré avec métriques
        logger.info(
            f"Page {page_index + 1} traitée avec succès (OCR + Classification)",
            extra={
                "metrics": {
                    "page_index": page_index,
                    "page_number": page_index + 1,
                    "document_id": document_id,
                    "processing_time_seconds": round(page_processing_time, 3),
                    "ocr_time_seconds": round(ocr_time, 3),
                    "classification_time_seconds": round(classification_time, 3),
                    "status": "success",
                    "document_type": classification_result.get('document_type'),
                    "confidence": classification_result.get('confidence', 0.0),
                    "num_ocr_lines": len(ocr_lines),
                    "worker_pid": worker_pid
                }
            }
        )
        
        # Métriques Prometheus
        doc_type = classification_result.get('document_type', 'Unknown')
        pages_processed_total.labels(status='success', doc_type=doc_type).inc()
        page_processing_duration_seconds.observe(page_processing_time)
        
        return classification_result
        
    except FileNotFoundError as e:
        page_processing_time = time.time() - page_start_time
        logger.error(
            f"Image non trouvée pour la page {page_index + 1}: {image_identifier}",
            exc_info=True,
            extra={
                "metrics": {
                    "page_index": page_index,
                    "image_identifier": image_identifier,
                    "processing_time_seconds": round(page_processing_time, 3),
                    "status": "error",
                    "error_type": "image_not_found",
                    "error_message": str(e),
                    "worker_pid": worker_pid
                }
            }
        )
        error_result = {
            'page_index': page_index,
            'document_type': None,
            'confidence': 0.0,
            'error': f"Image not found: {str(e)}",
            'document_id': document_id,
            'timestamp': time.time()
        }
        
        # Stocker aussi les erreurs dans Redis
        if document_id:
            try:
                redis_client = redis_broker.client
                redis_key = f"document:{document_id}:page:{page_index}"
                result_json = json.dumps(error_result)
                redis_client.setex(redis_key, 86400, result_json)
                
                document_pages_key = f"document:{document_id}:pages"
                redis_client.sadd(document_pages_key, page_index)
                redis_client.expire(document_pages_key, 86400)
            except Exception:
                pass
        
        # Métriques Prometheus pour les erreurs
        pages_processed_total.labels(status='failed', doc_type='Unknown').inc()
        page_processing_duration_seconds.observe(page_processing_time)
        processing_errors_total.labels(error_type='image_not_found').inc()
        
        return error_result
    except Exception as e:
        page_processing_time = time.time() - page_start_time
        logger.error(
            f"Erreur inattendue sur la page {page_index + 1}",
            exc_info=True,
            extra={
                "metrics": {
                    "page_index": page_index,
                    "processing_time_seconds": round(page_processing_time, 3),
                    "status": "error",
                    "error_type": "unexpected_error",
                    "error_message": str(e),
                    "worker_pid": worker_pid
                }
            }
        )
        error_result = {
            'page_index': page_index,
            'document_type': None,
            'confidence': 0.0,
            'error': f"An unexpected error occurred: {e}",
            'document_id': document_id,
            'timestamp': time.time()
        }
        
        # Stocker aussi les erreurs dans Redis
        if document_id:
            try:
                redis_client = redis_broker.client
                redis_key = f"document:{document_id}:page:{page_index}"
                result_json = json.dumps(error_result)
                redis_client.setex(redis_key, 86400, result_json)
                
                document_pages_key = f"document:{document_id}:pages"
                redis_client.sadd(document_pages_key, page_index)
                redis_client.expire(document_pages_key, 86400)
            except Exception:
                pass
        
        # Métriques Prometheus pour les erreurs inattendues
        pages_processed_total.labels(status='failed', doc_type='Unknown').inc()
        page_processing_duration_seconds.observe(page_processing_time)
        processing_errors_total.labels(error_type='unexpected_error').inc()
        
        return error_result

# --- Le Contremaître : Tâche de finalisation du document ---
@dramatiq.actor(time_limit=time_limit, max_retries=max_retries, store_results=True)
def finalize_document_task(
    document_id: str,
    original_filename: str,
    num_pages: int,
    **kwargs
) -> Dict[str, Any]:
    """
    Le Contremaître : Tâche qui vérifie périodiquement si tous les résultats de page sont disponibles
    dans Redis, puis agrège les résultats, génère la réponse finale, la stocke, et gère le nettoyage.
    
    Cette tâche est résiliente et ne doit JAMAIS échouer complètement. Même si certaines pages
    manquent ou échouent, elle retourne toujours un résultat partiel avec des informations détaillées.
    
    Cette tâche :
    1. Reçoit l'ID du document global et le nombre de pages attendues
    2. S'exécute avec un délai initial (eta) pour laisser le temps aux pages de se traiter
    3. Vérifie périodiquement si tous les résultats de page sont disponibles dans Redis
    4. A un timeout global (5 minutes par défaut, configurable) pour éviter d'attendre indéfiniment
    5. Vérifie la DLQ pour identifier les pages qui ont échoué et obtenir des informations détaillées
    6. Agrège TOUS les résultats disponibles (même partiels) et génère une réponse finale
    7. Marque le statut comme "success", "partial_success" ou "error" selon les résultats
    8. Liste explicitement les pages manquantes ou en erreur dans le résultat final
    9. Stocke le résultat final dans Redis
    10. Gère le nettoyage des fichiers temporaires
    
    **Résilience :**
    - Ne lève JAMAIS d'exception, retourne toujours un résultat (même partiel)
    - En cas de timeout, agrège les résultats disponibles et marque le statut comme "partial_success"
    - Fournit des messages d'erreur explicites pour chaque page manquante ou en erreur
    - Vérifie la DLQ pour identifier les pages qui ont échoué et inclure cette information
    
    Args:
        document_id: ID du document global
        original_filename: Nom original du fichier
        num_pages: Nombre de pages attendues
        **kwargs: Arguments additionnels passés par Dramatiq (options, etc.)
    
    Returns:
        Dict[str, Any]: Résultats agrégés de classification avec :
            - status: "success", "partial_success" ou "error"
            - results_by_page: Liste détaillée des résultats par page
            - page_summary: Résumé lisible par page (ex: "Page 1: OK - TypeA (confiance: 95%)", "Page 2: ERREUR - OCR Timeout")
            - missing_pages: Liste des numéros de pages manquantes
            - timeout_reached: Indicateur si le timeout a été atteint
    """
    from src.utils.logger import set_request_id
    from src.pipeline.models import PageClassificationResult
    from src.utils.storage import get_storage
    
    # Récupérer le request_id depuis les métadonnées du message pour le traçage
    request_id = None
    try:
        message = dramatiq.get_current_message()
        if message and message.message_metadata:
            request_id = message.message_metadata.get('request_id')
            if request_id:
                set_request_id(request_id)
    except Exception:
        pass
    
    start_time = time.time()
    master_pid = os.getpid()
    
    logger.info(
        f"Début de la finalisation pour le document '{document_id}' (fichier: '{original_filename}', {num_pages} page(s))",
        extra={
            "metrics": {
                "document_id": document_id,
                "filename": original_filename,
                "total_pages": num_pages,
                "worker_pid": master_pid,
                "stage": "finalization_start"
            }
        }
    )
    
    # Envelopper toute la logique dans un try/except pour garantir qu'on retourne toujours un résultat
    try:
        # Récupérer le client Redis
        redis_client = redis_broker.client
        
        # Timeout global : 5 minutes par document (configurable)
        # Ce timeout est indépendant du time_limit de Dramatiq pour garantir une finalisation même en cas de problèmes
        try:
            config = get_config_manager()
            global_timeout_seconds = config.get('dramatiq', 'finalization_timeout_seconds', default=300)  # 5 minutes par défaut
        except:
            global_timeout_seconds = 300  # 5 minutes par défaut
        
        # Vérifier périodiquement si tous les résultats sont disponibles
        check_interval = 1.0  # Vérifier toutes les secondes
        wait_start = time.time()
        page_results = {}
        timeout_reached = False
        iteration_count = 0
        
        logger.info(
            f"Attente des résultats pour {num_pages} page(s) (timeout global: {global_timeout_seconds}s)",
            extra={
                "metrics": {
                    "document_id": document_id,
                    "total_pages": num_pages,
                    "timeout_seconds": global_timeout_seconds
                }
            }
        )
        
        while len(page_results) < num_pages:
            iteration_count += 1
            # Vérifier le timeout global
            elapsed = time.time() - wait_start
            if elapsed >= global_timeout_seconds:
                timeout_reached = True
                logger.warning(
                    f"Timeout global ({global_timeout_seconds}s) atteint pour le document '{document_id}'. "
                    f"Agrégation partielle avec {len(page_results)}/{num_pages} page(s) disponible(s)",
                    extra={
                        "metrics": {
                            "document_id": document_id,
                            "total_pages": num_pages,
                            "pages_received": len(page_results),
                            "pages_missing": num_pages - len(page_results),
                            "elapsed_seconds": round(elapsed, 2),
                            "timeout_reached": True
                        }
                    }
                )
                break
            
            # Récupérer toutes les pages disponibles depuis Redis
            document_pages_key = f"document:{document_id}:pages"
            available_pages = redis_client.smembers(document_pages_key)
            
            # Convertir les bytes en int si nécessaire
            available_page_indices = set()
            for page_idx in available_pages:
                if isinstance(page_idx, bytes):
                    available_page_indices.add(int(page_idx.decode()))
                else:
                    available_page_indices.add(int(page_idx))
            
            # Récupérer les résultats pour chaque page disponible
            for page_index in available_page_indices:
                if page_index not in page_results:
                    redis_key = f"document:{document_id}:page:{page_index}"
                    result_json = redis_client.get(redis_key)
                    
                    if result_json:
                        try:
                            result = json.loads(result_json.decode() if isinstance(result_json, bytes) else result_json)
                            page_results[page_index] = result
                            logger.debug(f"Résultat récupéré pour la page {page_index + 1}")
                        except Exception as e:
                            logger.warning(f"Erreur lors du parsing du résultat pour la page {page_index + 1}: {e}")
            
            # Si on a tous les résultats, sortir de la boucle
            if len(page_results) >= num_pages:
                break
            
            # Logger l'état d'avancement toutes les 5 itérations
            if iteration_count % 5 == 0:
                logger.info(f"Attente des résultats... {len(page_results)}/{num_pages} page(s) reçue(s).")
            
            # Attendre avant de réessayer
            time.sleep(check_interval)
        
        results_wait_time = time.time() - wait_start
        
        logger.info(
            f"Résultats récupérés pour le document '{document_id}': {len(page_results)}/{num_pages} page(s)",
            extra={
                "metrics": {
                    "document_id": document_id,
                    "filename": original_filename,
                    "total_pages": num_pages,
                    "results_received": len(page_results),
                    "results_wait_time_seconds": round(results_wait_time, 3),
                    "worker_pid": master_pid,
                    "stage": "results_received"
                }
            }
        )
        
        # Vérifier la DLQ pour les pages manquantes et obtenir plus d'informations
        missing_pages_info = {}
        if timeout_reached or len(page_results) < num_pages:
            try:
                # Vérifier dans la DLQ si des tâches process_page_task ont échoué pour ce document
                dlq_name = 'dramatiq:dead_letter'
                dlq_messages = redis_client.lrange(dlq_name, 0, 1000)  # Vérifier jusqu'à 1000 messages
                
                for msg_data in dlq_messages:
                    try:
                        msg = json.loads(msg_data.decode('utf-8') if isinstance(msg_data, bytes) else msg_data)
                        # Vérifier si c'est une tâche process_page_task pour ce document_id
                        if (msg.get('actor_name') == 'process_page_task' and 
                            len(msg.get('args', [])) > 2 and 
                            msg.get('args', [])[2] == document_id):  # document_id est le 3ème argument
                            page_index = msg.get('args', [])[1] if len(msg.get('args', [])) > 1 else None
                            if page_index is not None and page_index not in page_results:
                                # Cette page a échoué et est dans la DLQ
                                missing_pages_info[page_index] = {
                                    'status': 'failed_in_dlq',
                                    'message': f"La tâche de traitement de la page a échoué et est dans la Dead Letter Queue",
                                    'message_id': msg.get('message_id'),
                                    'retries': msg.get('retries', 0),
                                    'max_retries': msg.get('max_retries', 0)
                                }
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"Erreur lors de la vérification de la DLQ pour les pages manquantes: {e}")
        
        # Construire la liste des résultats formatés et analyser les erreurs
        formatted_results = []
        successful_pages = 0
        failed_pages = 0
        missing_pages = []
        
        for page_index in range(num_pages):
            result = page_results.get(page_index)
            page_number = page_index + 1
            
            if result is None:
                # Page manquante - déterminer la raison
                failed_pages += 1
                missing_pages.append(page_number)
                
                # Vérifier si on a des informations depuis la DLQ
                page_info = missing_pages_info.get(page_index)
                if page_info:
                    error_message = (
                        f"Page non traitée - La tâche a échoué après {page_info['retries']}/{page_info['max_retries']} tentatives "
                        f"et a été envoyée dans la Dead Letter Queue. Message ID: {page_info['message_id']}"
                    )
                elif timeout_reached:
                    error_message = (
                        f"Page non traitée - Timeout global ({global_timeout_seconds}s) atteint avant que la page ne soit traitée. "
                        f"La tâche peut encore être en cours ou avoir échoué silencieusement."
                    )
                else:
                    error_message = "Résultat non disponible dans Redis - La page n'a pas été traitée ou le résultat a expiré"
                
                formatted_results.append(PageClassificationResult(
                    page_number=page_number,
                    document_type=None,
                    confidence=0.0,
                    error=error_message
                ).model_dump())
            else:
                # Vérifier si cette page a réussi ou échoué
                has_error = result.get('error') is not None
                if has_error:
                    failed_pages += 1
                    # Améliorer le message d'erreur pour être plus explicite
                    error_msg = result.get('error', 'Erreur inconnue')
                    if 'OCR failed' in error_msg:
                        error_msg = f"OCR Timeout ou Erreur: {error_msg}"
                    elif 'Image not found' in error_msg:
                        error_msg = f"Image non trouvée: {error_msg}"
                else:
                    successful_pages += 1
                    error_msg = None
                
                formatted_results.append(PageClassificationResult(
                    page_number=page_number,
                    document_type=result.get('document_type'),
                    confidence=result.get('confidence', 0.0),
                    error=error_msg
                ).model_dump())
        
        processing_time = time.time() - start_time
        avg_time_per_page = processing_time / num_pages if num_pages > 0 else 0
        
        # Déterminer le statut final avec des informations détaillées
        if failed_pages == 0:
            final_status = "success"
            status_message = f"Toutes les pages ont été traitées avec succès ({successful_pages}/{num_pages})"
        elif successful_pages == 0:
            final_status = "error"
            if missing_pages:
                status_message = (
                    f"Toutes les pages ont échoué ({failed_pages}/{num_pages}). "
                    f"Pages manquantes: {', '.join(map(str, missing_pages))}"
                )
            else:
                status_message = f"Toutes les pages ont échoué ({failed_pages}/{num_pages})"
        else:
            final_status = "partial_success"
            if missing_pages:
                status_message = (
                    f"Traitement partiel: {successful_pages} page(s) réussie(s), {failed_pages} page(s) échouée(s) "
                    f"sur {num_pages} total. Pages manquantes: {', '.join(map(str, missing_pages))}"
                )
            else:
                status_message = (
                    f"Traitement partiel: {successful_pages} page(s) réussie(s), {failed_pages} page(s) échouée(s) "
                    f"sur {num_pages} total"
                )
        
        # Ajouter un avertissement si le timeout a été atteint
        if timeout_reached:
            status_message += f" [TIMEOUT: Le timeout global de {global_timeout_seconds}s a été atteint]"
        
        # Construire le résultat final avec des informations détaillées
        final_result = {
            "status": final_status,
            "filename": original_filename,
            "total_pages": num_pages,
            "successful_pages": successful_pages,
            "failed_pages": failed_pages,
            "missing_pages": missing_pages,  # Liste explicite des numéros de pages manquantes
            "results_by_page": formatted_results,
            "processing_time": processing_time,
            "timeout_reached": timeout_reached,  # Indicateur si le timeout a été atteint
            "message": status_message
        }
        
        # Ajouter un résumé détaillé par page pour faciliter le debugging
        page_summary = []
        for result in formatted_results:
            page_num = result.get('page_number')
            if result.get('error'):
                page_summary.append(f"Page {page_num}: ERREUR - {result.get('error')}")
            else:
                doc_type = result.get('document_type', 'Unknown')
                confidence = result.get('confidence', 0.0)
                page_summary.append(f"Page {page_num}: OK - {doc_type} (confiance: {confidence:.2%})")
        
        final_result["page_summary"] = page_summary  # Résumé lisible pour le debugging
        
        # Stocker le résultat final dans Redis
        final_result_key = f"document:{document_id}:final"
        try:
            final_result_json = json.dumps(final_result)
            redis_client.setex(final_result_key, 86400, final_result_json)  # 24 heures
            logger.debug(f"Résultat final stocké dans Redis: {final_result_key}")
        except Exception as e:
            logger.warning(f"Impossible de stocker le résultat final dans Redis: {e}")
        
        # === Nettoyage des fichiers temporaires ===
        try:
            storage = get_storage()
            # Récupérer les identifiants d'images depuis Redis
            image_identifiers_key = f"document:{document_id}:image_identifiers"
            image_identifiers_json = redis_client.get(image_identifiers_key)
            
            if image_identifiers_json:
                try:
                    image_identifiers = json.loads(image_identifiers_json.decode() if isinstance(image_identifiers_json, bytes) else image_identifiers_json)
                    
                    # Supprimer chaque image
                    deleted_count = 0
                    for image_identifier in image_identifiers:
                        try:
                            if storage.delete_file(image_identifier):
                                deleted_count += 1
                        except Exception as e:
                            logger.debug(f"Impossible de supprimer l'image {image_identifier}: {e}")
                    
                    # Supprimer la clé Redis des identifiants
                    redis_client.delete(image_identifiers_key)
                    
                    logger.info(
                        f"Nettoyage des fichiers temporaires pour le document '{document_id}': {deleted_count} image(s) supprimée(s)",
                        extra={
                            "metrics": {
                                "document_id": document_id,
                                "images_deleted": deleted_count,
                                "total_images": len(image_identifiers)
                            }
                        }
                    )
                except Exception as e:
                    logger.warning(f"Erreur lors du parsing des identifiants d'images: {e}")
            else:
                logger.debug(f"Aucun identifiant d'image trouvé pour le document '{document_id}'")
        except Exception as e:
            logger.warning(f"Erreur lors du nettoyage des fichiers temporaires: {e}", exc_info=True)
        
        # Log final avec le statut et le temps total
        logger.info(f"Finalisation terminée pour {document_id}. Statut: {final_status}. Durée totale du document: {processing_time:.1f}s.")
        
        logger.info(
            f"Finalisation terminée pour le document '{document_id}': {status_message}",
            extra={
                "metrics": {
                    "document_id": document_id,
                    "filename": original_filename,
                    "status": final_status,
                    "total_pages": num_pages,
                    "successful_pages": successful_pages,
                    "failed_pages": failed_pages,
                    "success_rate": round(successful_pages / num_pages * 100, 2) if num_pages > 0 else 0,
                    "total_processing_time_seconds": round(processing_time, 3),
                    "avg_time_per_page_seconds": round(avg_time_per_page, 3),
                    "worker_pid": master_pid,
                    "stage": "finalization_complete"
                }
            }
        )
        
        # Métriques Prometheus
        document_processing_duration_seconds.observe(processing_time)
        documents_processed_total.labels(status=final_status).inc()
        
        return final_result
    
    except Exception as e:
        # En cas d'erreur inattendue, retourner un résultat d'erreur plutôt que de faire échouer la tâche
        processing_time = time.time() - start_time
        logger.error(
            f"Erreur inattendue lors de la finalisation du document '{document_id}': {e}",
            exc_info=True,
            extra={
                "metrics": {
                    "document_id": document_id,
                    "filename": original_filename,
                    "total_pages": num_pages,
                    "processing_time_seconds": round(processing_time, 3),
                    "worker_pid": master_pid,
                    "stage": "finalization_error",
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            }
        )
        
        # Retourner un résultat d'erreur avec toutes les pages marquées comme échouées
        error_results = []
        for page_index in range(num_pages):
            error_results.append(PageClassificationResult(
                page_number=page_index + 1,
                document_type=None,
                confidence=0.0,
                error=f"Erreur lors de la finalisation: {str(e)}"
            ).model_dump())
        
        error_result = {
            "status": "error",
            "filename": original_filename,
            "total_pages": num_pages,
            "successful_pages": 0,
            "failed_pages": num_pages,
            "missing_pages": list(range(1, num_pages + 1)),
            "results_by_page": error_results,
            "processing_time": processing_time,
            "timeout_reached": False,
            "message": f"Erreur lors de la finalisation: {str(e)}",
            "page_summary": [f"Page {i+1}: ERREUR - Erreur lors de la finalisation: {str(e)}" for i in range(num_pages)]
        }
        
        # Essayer de stocker le résultat d'erreur dans Redis
        try:
            redis_client = redis_broker.client
            final_result_key = f"document:{document_id}:final"
            error_result_json = json.dumps(error_result)
            redis_client.setex(final_result_key, 86400, error_result_json)
        except Exception:
            pass  # Ignorer les erreurs de stockage Redis
        
        # Métriques Prometheus pour les erreurs de finalisation
        document_processing_duration_seconds.observe(processing_time)
        documents_processed_total.labels(status='error').inc()
        processing_errors_total.labels(error_type='finalization_error').inc()
        
        return error_result

# --- Tâche d'agrégation des résultats (ancienne, conservée pour compatibilité) ---
@dramatiq.actor(time_limit=time_limit, max_retries=max_retries, store_results=True)
def aggregate_classification_results(
    document_id: str,
    original_filename: str,
    message_ids: List[str],
    **kwargs
) -> Dict[str, Any]:
    """
    Tâche d'agrégation qui attend les résultats de toutes les tâches process_page_task
    et les agrège en un résultat final.
    
    Args:
        document_id: ID du document global
        original_filename: Nom original du fichier
        message_ids: Liste des IDs de messages des tâches process_page_task
        **kwargs: Arguments additionnels passés par Dramatiq (options, etc.)
    
    Returns:
        Dict[str, Any]: Résultats agrégés de classification pour toutes les pages
    """
    from src.utils.logger import set_request_id
    from src.pipeline.models import PageClassificationResult
    from dramatiq.results.errors import ResultMissing
    
    # Récupérer le request_id depuis les métadonnées du message pour le traçage
    request_id = None
    try:
        message = dramatiq.get_current_message()
        if message and message.message_metadata:
            request_id = message.message_metadata.get('request_id')
            if request_id:
                set_request_id(request_id)
    except Exception:
        pass
    
    start_time = time.time()
    master_pid = os.getpid()
    num_pages = len(message_ids)
    
    logger.info(
        f"Début de l'agrégation des résultats pour le document '{document_id}' (fichier: '{original_filename}')",
        extra={
            "metrics": {
                "document_id": document_id,
                "filename": original_filename,
                "total_pages": num_pages,
                "worker_pid": master_pid,
                "stage": "aggregation_start"
            }
        }
    )
    
    # Récupérer le backend de résultats pour récupérer les résultats des messages
    broker = process_page_task.broker
    results_middleware = None
    for middleware in broker.middleware:
        if isinstance(middleware, Results):
            results_middleware = middleware
            break
    
    if results_middleware is None:
        raise RuntimeError("Backend de résultats non configuré")
    
    backend = results_middleware.backend
    
    # Attendre tous les résultats en récupérant chaque résultat depuis le backend
    results_wait_start_time = time.time()
    page_results = []
    
    try:
        results_timeout_ms = time_limit - 10000  # Garde 10 secondes de marge
        logger.info(f"Attente des résultats avec un timeout de {results_timeout_ms / 1000} secondes...")
        
        # Récupérer les résultats pour chaque message_id
        for message_id in message_ids:
            # Créer un message factice avec le message_id pour récupérer le résultat
            message = process_page_task.message().copy(message_id=message_id)
            
            # Attendre le résultat avec retry
            max_wait_time = results_timeout_ms / num_pages  # Répartir le timeout
            wait_start = time.time()
            
            while True:
                try:
                    result = backend.get_result(message, block=False)
                    page_results.append(result)
                    break
                except ResultMissing:
                    # Attendre un peu avant de réessayer
                    import time as time_module
                    elapsed = (time_module.time() - wait_start) * 1000
                    if elapsed >= max_wait_time:
                        # Timeout pour ce message
                        logger.warning(f"Timeout pour le message {message_id}")
                        page_results.append({
                            'page_index': len(page_results),
                            'document_type': None,
                            'confidence': 0.0,
                            'error': f"Timeout lors de la récupération du résultat"
                        })
                        break
                    time_module.sleep(0.5)  # Attendre 500ms avant de réessayer
                except Exception as e:
                    logger.error(f"Erreur lors de la récupération du résultat pour {message_id}: {e}")
                    page_results.append({
                        'page_index': len(page_results),
                        'document_type': None,
                        'confidence': 0.0,
                        'error': f"Erreur lors de la récupération: {str(e)}"
                    })
                    break
        
        results_wait_time = time.time() - results_wait_start_time
        
        logger.info(
            f"Tous les résultats reçus pour le document '{document_id}'",
            extra={
                "metrics": {
                    "document_id": document_id,
                    "filename": original_filename,
                    "total_pages": num_pages,
                    "results_received": len(page_results),
                    "results_wait_time_seconds": round(results_wait_time, 3),
                    "worker_pid": master_pid,
                    "stage": "results_received"
                }
            }
        )
    except Exception as e:
        results_wait_time = time.time() - results_wait_start_time
        error_msg = f"Échec de la récupération des résultats pour le document '{document_id}': {str(e)}"
        logger.error(
            error_msg,
            exc_info=True,
            extra={
                "metrics": {
                    "document_id": document_id,
                    "filename": original_filename,
                    "total_pages": num_pages,
                    "results_wait_time_seconds": round(results_wait_time, 3),
                    "status": "error",
                    "error_type": "results_retrieval_failed",
                    "worker_pid": master_pid
                }
            }
        )
        raise RuntimeError(error_msg) from e
    
    # Construire la liste des résultats formatés et analyser les erreurs
    formatted_results = []
    successful_pages = 0
    failed_pages = 0
    
    for result in page_results:
        page_index = result.get('page_index', 0)
        page_number = page_index + 1
        
        # Vérifier si cette page a réussi ou échoué
        has_error = result.get('error') is not None
        if has_error:
            failed_pages += 1
        else:
            successful_pages += 1
        
        formatted_results.append(PageClassificationResult(
            page_number=page_number,
            document_type=result.get('document_type'),
            confidence=result.get('confidence', 0.0),
            error=result.get('error')
        ).model_dump())
    
    processing_time = time.time() - start_time
    avg_time_per_page = processing_time / num_pages if num_pages > 0 else 0
    
    # Déterminer le statut final
    if failed_pages == 0:
        final_status = "success"
        status_message = f"Toutes les pages ont été traitées avec succès ({successful_pages}/{num_pages})"
    elif successful_pages == 0:
        final_status = "error"
        status_message = f"Toutes les pages ont échoué ({failed_pages}/{num_pages})"
    else:
        final_status = "partial_success"
        status_message = f"Traitement partiel: {successful_pages} page(s) réussie(s), {failed_pages} page(s) échouée(s) sur {num_pages} total"
    
    logger.info(
        f"Agrégation terminée pour le document '{document_id}': {status_message}",
        extra={
            "metrics": {
                "document_id": document_id,
                "filename": original_filename,
                "status": final_status,
                "total_pages": num_pages,
                "successful_pages": successful_pages,
                "failed_pages": failed_pages,
                "success_rate": round(successful_pages / num_pages * 100, 2) if num_pages > 0 else 0,
                "total_processing_time_seconds": round(processing_time, 3),
                "avg_time_per_page_seconds": round(avg_time_per_page, 3),
                "worker_pid": master_pid,
                "stage": "aggregation_complete"
            }
        }
    )
    
    return {
        "status": final_status,
        "filename": original_filename,
        "total_pages": num_pages,
        "successful_pages": successful_pages,
        "failed_pages": failed_pages,
        "results_by_page": formatted_results,
        "processing_time": processing_time,
        "message": status_message
        }

# --- Définition de la Tâche (Actor Dramatiq) ---
# Utiliser les mêmes valeurs de retries et time_limit que pour process_page_task
@dramatiq.actor(time_limit=time_limit, max_retries=max_retries, store_results=True)
def classify_document_task(
    original_filename: str,
    ocr_results_by_page: Dict[int, List[Dict[str, Any]]],
    ocr_errors_by_page: Optional[Dict[int, str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Tâche asynchrone pour classifier un document à partir de données OCR déjà extraites.
    
    Cette fonction reçoit les données OCR extraites par l'orchestrateur (API) et
    se contente de classifier chaque page. L'extraction OCR est complètement découplée.
    
    Args:
        original_filename: Nom original du fichier
        ocr_results_by_page: Dictionnaire {page_index: [lignes OCR]} avec les données OCR déjà extraites
        ocr_errors_by_page: Dictionnaire {page_index: error_message} avec les erreurs OCR (optionnel)
        **kwargs: Arguments additionnels passés par Dramatiq (options, etc.)
    
    Returns:
        Dict[str, Any]: Résultats de classification pour toutes les pages
    """
    from src.utils.logger import set_request_id
    
    # Récupérer le request_id depuis les métadonnées du message pour le traçage
    request_id = None
    try:
        message = dramatiq.get_current_message()
        if message and message.message_metadata:
            request_id = message.message_metadata.get('request_id')
            if request_id:
                set_request_id(request_id)
    except Exception:
        pass
    
    start_time = time.time()
    
    # Vérifier et désérialiser si ocr_results_by_page est une chaîne JSON
    if isinstance(ocr_results_by_page, str):
        import json
        try:
            ocr_results_by_page = json.loads(ocr_results_by_page)
        except json.JSONDecodeError:
            # Si ce n'est pas du JSON valide, c'est peut-être un URI de fichier (ancien format)
            # Dans ce cas, cette fonction ne devrait pas être appelée avec ces arguments
            raise ValueError(
                f"ocr_results_by_page doit être un dictionnaire ou une chaîne JSON valide. "
                f"Reçu une chaîne qui n'est pas du JSON: {ocr_results_by_page[:100]}... "
                f"Cette fonction attend des données OCR déjà extraites, pas un URI de fichier. "
                f"Utilisez process_page_task ou finalize_document_task à la place."
            )
    
    # Vérifier que ocr_results_by_page est bien un dictionnaire
    if not isinstance(ocr_results_by_page, dict):
        raise TypeError(
            f"ocr_results_by_page doit être un dictionnaire, reçu: {type(ocr_results_by_page)}. "
            f"Valeur: {str(ocr_results_by_page)[:200]}... "
            f"Cette fonction attend des données OCR déjà extraites. "
            f"Si vous essayez de classifier un document, utilisez l'endpoint /api/v1/classify qui utilise process_page_task."
        )
    
    if ocr_errors_by_page is None:
        ocr_errors_by_page = {}
    elif isinstance(ocr_errors_by_page, str):
        import json
        try:
            ocr_errors_by_page = json.loads(ocr_errors_by_page)
        except json.JSONDecodeError as e:
            raise ValueError(f"ocr_errors_by_page doit être un dictionnaire ou une chaîne JSON valide: {e}")
    
    if not isinstance(ocr_errors_by_page, dict):
        raise TypeError(f"ocr_errors_by_page doit être un dictionnaire, reçu: {type(ocr_errors_by_page)}")
    
    num_pages = len(ocr_results_by_page) + len(ocr_errors_by_page)
    if num_pages == 0:
        raise ValueError("Aucune donnée OCR fournie pour la classification")

    # === NOUVELLE LOGIQUE : Classification uniquement (OCR déjà extrait) ===
    # Le worker de classification ne fait plus d'extraction OCR.
    # Il reçoit les données OCR déjà extraites et se contente de classifier.
    
    master_pid = os.getpid()
    
    logger.info(
        f"Début de la classification pour le fichier '{original_filename}'",
        extra={
            "metrics": {
                "filename": original_filename,
                "total_pages": num_pages,
                "worker_pid": master_pid,
                "stage": "classification_start"
            }
        }
    )
    
    # Créer une tâche de classification pour chaque page avec ses données OCR
    page_messages = []
    classification_start_time = time.time()
    
    # Obtenir tous les page_index (ceux avec OCR et ceux avec erreurs)
    all_page_indices = set(ocr_results_by_page.keys()) | set(ocr_errors_by_page.keys())
    
    for page_index in sorted(all_page_indices):
        # Récupérer les données OCR pour cette page
        page_ocr_lines = ocr_results_by_page.get(page_index)
        page_ocr_error = ocr_errors_by_page.get(page_index)
        
        # Si l'OCR a échoué pour cette page, passer None pour que la tâche retourne une erreur
        if page_ocr_error:
            page_ocr_lines = None
        
        # Créer un message Dramatiq pour classifier cette page avec les données OCR
        # NOTE: On ne passe plus d'image_uri car on n'en a plus besoin
        if request_id:
            message = process_page_task.send(
                "",  # image_uri vide (plus nécessaire)
                page_index,
                page_ocr_lines,  # Passer les données OCR déjà extraites
                options={"message_metadata": {"request_id": request_id}}
            )
        else:
            message = process_page_task.send("", page_index, page_ocr_lines)
        page_messages.append(message)
        
        logger.debug(
            f"Tâche de classification créée pour la page {page_index + 1}",
            extra={
                "metrics": {
                    "filename": original_filename,
                    "page_index": page_index,
                    "page_number": page_index + 1,
                    "has_ocr_data": page_ocr_lines is not None,
                    "message_id": message.message_id,
                    "worker_pid": master_pid
                }
            }
        )
    
    # Utiliser les groupes Dramatiq pour exécuter toutes les tâches en parallèle
    # et attendre leurs résultats
    logger.info(
        f"Création du groupe Dramatiq avec {len(page_messages)} tâche(s)",
        extra={
            "metrics": {
                "filename": original_filename,
                "total_pages": num_pages,
                "orchestration_time_seconds": round(orchestration_time, 3),
                "worker_pid": master_pid,
                "stage": "group_creation"
            }
        }
    )
    
    group = dramatiq.group(page_messages)
    
    # Exécuter le groupe (lance toutes les tâches en parallèle)
    group_result = group.run()
    
    # Attendre tous les résultats (block=True met en pause jusqu'à ce que toutes les tâches soient terminées)
    # get_results() retourne un générateur, on doit le convertir en liste pour pouvoir utiliser len()
    results_wait_start_time = time.time()
    try:
        # Le time_limit de l'acteur est en millisecondes. 
        # On donne presque tout ce temps pour récupérer les résultats, moins une marge de sécurité.
        results_timeout_ms = time_limit - 10000  # Garde 10 secondes de marge.

        logger.info(f"Attente des résultats avec un timeout de {results_timeout_ms / 1000} secondes...")

        page_results_generator = group_result.get_results(
            block=True,
            timeout=results_timeout_ms
        )
        
        # Convertir le générateur en liste pour pouvoir utiliser len() et itérer plusieurs fois
        page_results = list(page_results_generator)
        results_wait_time = time.time() - results_wait_start_time
        
        logger.info(
            f"Tous les résultats reçus pour le fichier '{original_filename}'",
            extra={
                "metrics": {
                    "filename": original_filename,
                    "total_pages": num_pages,
                    "results_received": len(page_results),
                    "results_wait_time_seconds": round(results_wait_time, 3),
                    "worker_pid": master_pid,
                    "stage": "results_received"
                }
            }
        )
    except Exception as e:
        results_wait_time = time.time() - results_wait_start_time
        error_msg = f"Échec de la récupération des résultats des tâches de traitement pour le fichier '{original_filename}': {str(e)}"
        logger.error(
            error_msg,
            exc_info=True,
            extra={
                "metrics": {
                    "filename": original_filename,
                    "total_pages": num_pages,
                    "results_wait_time_seconds": round(results_wait_time, 3),
                    "status": "error",
                    "error_type": "results_retrieval_failed",
                    "worker_pid": master_pid
                }
            }
        )
        raise RuntimeError(error_msg) from e
    
    # Vérifier que nous avons reçu le bon nombre de résultats
    if len(page_results) != num_pages:
        logger.warning(
            f"Nombre de résultats ne correspond pas au nombre de pages pour '{original_filename}'",
            extra={
                "metrics": {
                    "filename": original_filename,
                    "expected_pages": num_pages,
                    "received_results": len(page_results),
                    "worker_pid": master_pid
                }
            }
        )
    
    # Construire la liste des résultats formatés et analyser les erreurs
    formatted_results = []
    successful_pages = 0
    failed_pages = 0
    page_processing_times = []
    
    for i, result in enumerate(page_results):
        # Le résultat est déjà un dictionnaire avec page_index, document_type, confidence, etc.
        # Utiliser page_index du résultat si disponible, sinon utiliser l'index de la boucle
        page_index_from_result = result.get('page_index', i)
        page_number = page_index_from_result + 1
        
        # Vérifier si cette page a réussi ou échoué
        has_error = result.get('error') is not None
        if has_error:
            failed_pages += 1
        else:
            successful_pages += 1
        
        formatted_results.append(PageClassificationResult(
            page_number=page_number,
            document_type=result.get('document_type'),
            confidence=result.get('confidence', 0.0),
            error=result.get('error')
        ).model_dump())
    
    processing_time = time.time() - start_time
    
    # Calculer le temps moyen par page
    avg_time_per_page = processing_time / num_pages if num_pages > 0 else 0
    
    # Déterminer le statut final en fonction des résultats
    if failed_pages == 0:
        final_status = "success"
        status_message = f"Toutes les pages ont été traitées avec succès ({successful_pages}/{num_pages})"
    elif successful_pages == 0:
        final_status = "error"
        status_message = f"Toutes les pages ont échoué ({failed_pages}/{num_pages})"
    else:
        final_status = "partial_success"
        status_message = f"Traitement partiel: {successful_pages} page(s) réussie(s), {failed_pages} page(s) échouée(s) sur {num_pages} total"
    
    # Log structuré final avec toutes les métriques
    logger.info(
        f"Traitement terminé pour le fichier '{original_filename}': {status_message}",
        extra={
            "metrics": {
                "filename": original_filename,
                "status": final_status,
                "total_pages": num_pages,
                "successful_pages": successful_pages,
                "failed_pages": failed_pages,
                "success_rate": round(successful_pages / num_pages * 100, 2) if num_pages > 0 else 0,
                "total_processing_time_seconds": round(processing_time, 3),
                "avg_time_per_page_seconds": round(avg_time_per_page, 3),
                "orchestration_time_seconds": round(orchestration_time, 3),
                "worker_pid": master_pid,
                "stage": "completion"
            }
        }
    )
    
    # Retourner les résultats complets au format JSON final
    # NOTE: Le nettoyage des fichiers temporaires (images et fichier source) est géré par
    # une tâche périodique (cleanup_temporary_files) qui s'exécute toutes les heures.
    # Cela évite les problèmes de verrouillage de fichiers sur Windows et garantit que
    # les fichiers ne sont jamais supprimés pendant qu'ils sont en cours d'utilisation.
    return {
        "status": final_status,
        "filename": original_filename,
        "total_pages": num_pages,
        "successful_pages": successful_pages,
        "failed_pages": failed_pages,
        "results_by_page": formatted_results,
        "processing_time": processing_time,
        "message": status_message
    }
# ==========================================
# Tâche Périodique de Nettoyage
# ==========================================

try:
    from dramatiq.cron import cron
    CRON_AVAILABLE = True
except ImportError:
    # Si dramatiq.cron n'est pas disponible, on désactive les tâches périodiques
    CRON_AVAILABLE = False
    logger.warning("dramatiq.cron n'est pas disponible. Les tâches périodiques de nettoyage seront désactivées.")


if CRON_AVAILABLE:
    @dramatiq.actor(priority='low')
    @cron('*/30 * * * *')  # S'exécute toutes les 30 minutes
    def cleanup_or_retry_stale_tasks(**kwargs) -> Dict[str, Any]:
        """
        Le Janitor (Nettoyeur/Récupérateur) : Tâche périodique pour détecter et récupérer les documents orphelins.
        
        Cette tâche s'exécute toutes les 30 minutes et :
        1. Scanne Redis à la recherche de documents "orphelins"
        2. Un document est orphelin si :
           - Il a des résultats de pages stockés (document:{doc_id}:page:*)
           - Mais la tâche de finalisation correspondante n'existe plus dans les queues ou a échoué
           - Et le dernier résultat de page date de plus d'une heure
        3. Pour chaque document orphelin trouvé, relance une nouvelle tâche finalize_document_task
        
        C'est un mécanisme d'auto-guérison (self-healing) simple mais extrêmement puissant.
        
        Args:
            **kwargs: Arguments additionnels passés par Dramatiq (options, etc.)
        
        Returns:
            Dict[str, Any]: Statut du scan avec le nombre de documents orphelins trouvés et récupérés
        """
        logger.info("Exécution du Janitor : scan des documents orphelins...")
        
        redis_client = redis_broker.client
        orphaned_docs = []
        recovered_docs = []
        current_time = time.time()
        stale_threshold_seconds = 3600  # 1 heure
        
        try:
            # Scanner toutes les clés document:*:pages pour trouver les documents
            # Note: Redis SCAN est préférable à KEYS pour la performance en production,
            # mais KEYS est plus simple pour cette implémentation. Pour de grandes bases,
            # considérer l'utilisation de SCAN avec un curseur.
            pattern = "document:*:pages"
            try:
                # Essayer d'utiliser SCAN si possible (plus efficace)
                all_keys = []
                cursor = 0
                while True:
                    cursor, keys = redis_client.scan(cursor, match=pattern, count=100)
                    all_keys.extend(keys)
                    if cursor == 0:
                        break
            except Exception:
                # Fallback sur KEYS si SCAN n'est pas disponible
                all_keys = redis_client.keys(pattern)
            
            logger.debug(f"Scan de {len(all_keys)} document(s) potentiel(s)")
            
            for pages_key in all_keys:
                try:
                    # Extraire le document_id depuis la clé (format: document:{doc_id}:pages)
                    if isinstance(pages_key, bytes):
                        pages_key_str = pages_key.decode()
                    else:
                        pages_key_str = pages_key
                    
                    # Format: "document:doc_abc123:pages"
                    parts = pages_key_str.split(':')
                    if len(parts) != 3 or parts[0] != 'document' or parts[2] != 'pages':
                        continue
                    
                    document_id = parts[1]
                    
                    # Vérifier si le document a un résultat final
                    final_result_key = f"document:{document_id}:final"
                    has_final_result = redis_client.exists(final_result_key)
                    
                    if has_final_result:
                        # Le document a déjà été finalisé, on peut l'ignorer
                        continue
                    
                    # Récupérer les pages disponibles
                    available_pages = redis_client.smembers(pages_key)
                    if not available_pages:
                        # Aucune page, on peut ignorer
                        continue
                    
                    # Vérifier le timestamp du dernier résultat de page et compter les pages avec résultats
                    latest_timestamp = 0
                    pages_with_results = 0
                    total_pages_in_set = len(available_pages)
                    
                    for page_idx in available_pages:
                        if isinstance(page_idx, bytes):
                            page_index = int(page_idx.decode())
                        else:
                            page_index = int(page_idx)
                        
                        page_result_key = f"document:{document_id}:page:{page_index}"
                        page_result_json = redis_client.get(page_result_key)
                        
                        if page_result_json:
                            try:
                                page_result = json.loads(page_result_json.decode() if isinstance(page_result_json, bytes) else page_result_json)
                                page_timestamp = page_result.get('timestamp', 0)
                                if page_timestamp > latest_timestamp:
                                    latest_timestamp = page_timestamp
                                pages_with_results += 1
                            except Exception:
                                pass
                    
                    if pages_with_results == 0:
                        # Aucune page avec résultat, on peut ignorer
                        continue
                    
                    # Utiliser le nombre de pages dans le set comme référence (peut être différent si certaines pages ont échoué)
                    # On considère qu'on a besoin d'au moins une page avec résultat pour être orphelin
                    num_pages = total_pages_in_set
                    
                    # Vérifier si le dernier résultat date de plus d'une heure
                    age_seconds = current_time - latest_timestamp
                    if age_seconds < stale_threshold_seconds:
                        # Trop récent, on attend encore
                        continue
                    
                    # Vérifier si la tâche de finalisation existe encore dans les queues
                    # On vérifie dans la queue par défaut et dans la DLQ
                    task_exists = False
                    try:
                        # Vérifier dans la queue par défaut
                        queue_name = finalize_document_task.queue_name or "default"
                        full_queue_name = f"dramatiq:queue:{queue_name}"
                        queue_messages = redis_client.lrange(full_queue_name, 0, 100)  # Vérifier les 100 premiers
                        
                        for msg_data in queue_messages:
                            try:
                                msg = json.loads(msg_data.decode('utf-8') if isinstance(msg_data, bytes) else msg_data)
                                # Vérifier si c'est une tâche finalize_document_task pour ce document_id
                                if (msg.get('actor_name') == 'finalize_document_task' and 
                                    len(msg.get('args', [])) > 0 and 
                                    msg.get('args', [])[0] == document_id):
                                    task_exists = True
                                    break
                            except Exception:
                                pass
                        
                        # Vérifier aussi dans la DLQ
                        if not task_exists:
                            dlq_name = 'dramatiq:dead_letter'
                            dlq_messages = redis_client.lrange(dlq_name, 0, 100)
                            for msg_data in dlq_messages:
                                try:
                                    msg = json.loads(msg_data.decode('utf-8') if isinstance(msg_data, bytes) else msg_data)
                                    if (msg.get('actor_name') == 'finalize_document_task' and 
                                        len(msg.get('args', [])) > 0 and 
                                        msg.get('args', [])[0] == document_id):
                                        # La tâche est dans la DLQ, donc elle a échoué
                                        # On peut quand même la considérer comme orpheline
                                        break
                                except Exception:
                                    pass
                    except Exception as e:
                        logger.debug(f"Erreur lors de la vérification des queues pour {document_id}: {e}")
                    
                    # Si la tâche n'existe pas, le document est orphelin
                    if not task_exists:
                        orphaned_docs.append({
                            'document_id': document_id,
                            'num_pages': num_pages,
                            'latest_timestamp': latest_timestamp,
                            'age_seconds': age_seconds
                        })
                
                except Exception as e:
                    logger.warning(f"Erreur lors du traitement de la clé {pages_key}: {e}")
                    continue
            
            # Pour chaque document orphelin, relancer finalize_document_task
            for orphan in orphaned_docs:
                document_id = orphan['document_id']
                num_pages = orphan['num_pages']
                
                try:
                    # Récupérer le filename depuis Redis
                    filename_key = f"document:{document_id}:filename"
                    filename_json = redis_client.get(filename_key)
                    
                    if filename_json:
                        if isinstance(filename_json, bytes):
                            original_filename = filename_json.decode()
                        else:
                            original_filename = filename_json
                    else:
                        # Filename non trouvé, utiliser un nom par défaut
                        original_filename = f"document_{document_id}"
                    
                    # Relancer la tâche de finalisation
                    logger.info(
                        f"Récupération du document orphelin '{document_id}' ({num_pages} page(s), âge: {orphan['age_seconds']:.0f}s)",
                        extra={
                            "metrics": {
                                "document_id": document_id,
                                "num_pages": num_pages,
                                "age_seconds": round(orphan['age_seconds']),
                                "action": "retry_finalization"
                            }
                        }
                    )
                    
                    # Relancer sans délai (eta) car le document est déjà ancien
                    finalize_document_task.send(document_id, original_filename, num_pages)
                    recovered_docs.append(document_id)
                    
                except Exception as e:
                    logger.error(
                        f"Erreur lors de la récupération du document orphelin '{document_id}': {e}",
                        exc_info=True,
                        extra={
                            "metrics": {
                                "document_id": document_id,
                                "error": str(e),
                                "action": "retry_failed"
                            }
                        }
                    )
            
            logger.info(
                f"Janitor terminé: {len(orphaned_docs)} document(s) orphelin(s) trouvé(s), {len(recovered_docs)} récupéré(s)",
                extra={
                    "metrics": {
                        "orphaned_docs_found": len(orphaned_docs),
                        "recovered_docs": len(recovered_docs),
                        "status": "success"
                    }
                }
            )
            
            return {
                "status": "success",
                "orphaned_docs_found": len(orphaned_docs),
                "recovered_docs": len(recovered_docs),
                "message": f"Scan terminé: {len(orphaned_docs)} document(s) orphelin(s) trouvé(s), {len(recovered_docs)} récupéré(s)"
            }
        
        except Exception as e:
            logger.error(
                f"Erreur lors du scan des documents orphelins: {e}",
                exc_info=True,
                extra={
                    "metrics": {
                        "status": "error",
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    }
                }
            )
            return {
                "status": "error",
                "message": f"Erreur lors du scan: {str(e)}"
            }
    
    @dramatiq.actor(priority='low')
    @cron('0 * * * *')  # S'exécute toutes les heures (à la minute 0 de chaque heure)
    def cleanup_temporary_files(**kwargs) -> Dict[str, Any]:
        """
        Tâche périodique pour nettoyer les anciens fichiers temporaires.
        
        Cette tâche s'exécute automatiquement toutes les heures et supprime les fichiers
        temporaires qui sont plus anciens qu'un certain seuil (par défaut 1 heure).
        Cela évite les problèmes de verrouillage de fichiers sur Windows et garantit
        que les fichiers ne sont jamais supprimés pendant qu'ils sont en cours d'utilisation.
        
        Args:
            **kwargs: Arguments additionnels passés par Dramatiq (options, etc.)
        
        Returns:
            Dict[str, Any]: Statut du nettoyage avec le nombre de fichiers supprimés
        """
        from src.utils.storage import get_storage
        
        logger.info("Exécution de la tâche de nettoyage des fichiers temporaires...")
        
        try:
            storage = get_storage()
            
            if hasattr(storage, 'cleanup_old_files'):
                # Nettoyer les fichiers de plus d'1 heure
                deleted_count = storage.cleanup_old_files(max_age_hours=1)
                
                logger.info(
                    f"Nettoyage terminé. {deleted_count} fichier(s) supprimé(s).",
                    extra={
                        "metrics": {
                            "deleted_files_count": deleted_count,
                            "status": "success",
                            "max_age_hours": 1
                        }
                    }
                )
                
                return {
                    "status": "success",
                    "deleted_files_count": deleted_count,
                    "message": f"Nettoyage terminé. {deleted_count} fichier(s) supprimé(s)."
                }
            else:
                logger.warning("Le backend de stockage actuel ne supporte pas le nettoyage automatique.")
                return {
                    "status": "skipped",
                    "message": "Le backend de stockage ne supporte pas le nettoyage automatique."
                }
        except Exception as e:
            logger.error(
                f"Erreur lors du nettoyage des fichiers temporaires: {e}",
                exc_info=True,
                extra={
                    "metrics": {
                        "status": "error",
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    }
                }
            )
            return {
                "status": "error",
                "message": f"Erreur lors du nettoyage: {str(e)}"
            }
else:
    # Si cron n'est pas disponible, créer des fonctions vides pour éviter les erreurs
    def cleanup_or_retry_stale_tasks(**kwargs) -> Dict[str, Any]:
        """Fonction Janitor désactivée (dramatiq.cron non disponible)"""
        logger.warning("cleanup_or_retry_stale_tasks appelée mais dramatiq.cron n'est pas disponible")
        return {
            "status": "disabled",
            "message": "Janitor désactivé (dramatiq.cron non disponible)"
        }
    
    def cleanup_temporary_files(**kwargs) -> Dict[str, Any]:
        """Fonction de nettoyage désactivée (dramatiq.cron non disponible)"""
        logger.warning("cleanup_temporary_files appelée mais dramatiq.cron n'est pas disponible")
        return {
            "status": "disabled",
            "message": "Nettoyage périodique désactivé (dramatiq.cron non disponible)"
        }