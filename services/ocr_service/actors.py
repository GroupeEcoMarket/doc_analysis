# services/ocr_service/actors.py
"""
Acteurs Dramatiq isolés pour le service OCR.

Ce module contient uniquement les workers OCR, isolés du reste de l'application.
Il se concentre uniquement sur l'extraction de texte via PaddleOCR.
"""

import os
import time
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
from filelock import FileLock

import dramatiq
import numpy as np
from dramatiq.brokers.redis import RedisBroker
from dramatiq.results import Results
from dramatiq.results.backends import RedisBackend

# --- Bootstrap PaddleOCR (essentiel) ---
# Doit être appelé AVANT toute importation de PaddleOCR
def configure_paddle_environment():
    """
    Configure l'environnement pour PaddleOCR.
    
    Lit la configuration locale et désactive MKL-DNN si nécessaire.
    """
    try:
        # Trouver le fichier config.yaml dans ce service
        service_root = Path(__file__).parent
        config_path = service_root / "config.yaml"

        if not config_path.exists():
            print("AVERTISSEMENT [ocr_service]: config.yaml non trouvé. MKL-DNN pourrait être activé par défaut.")
            return

        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Vérifier la configuration et désactiver MKL-DNN si nécessaire
        enable_mkldnn = config.get('ocr', {}).get('enable_mkldnn', False)

        if not enable_mkldnn:
            os.environ["FLAGS_use_mkldnn"] = "0"
            print("INFO [ocr_service]: MKL-DNN a été désactivé via FLAGS_use_mkldnn.")
        else:
            print("INFO [ocr_service]: MKL-DNN est activé selon la configuration.")

    except Exception as e:
        print(f"ERREUR [ocr_service]: Impossible de configurer l'environnement Paddle. Erreur : {e}")
        # Par sécurité, on désactive MKL-DNN en cas d'erreur
        os.environ["FLAGS_use_mkldnn"] = "0"
        print("INFO [ocr_service]: MKL-DNN a été désactivé par mesure de sécurité.")

# Configurer l'environnement Paddle AVANT toute importation
configure_paddle_environment()

# --- Imports après configuration ---
from paddleocr import PaddleOCR

# --- Configuration de Dramatiq ---
try:
    # Essayer de charger la config depuis le service
    import yaml
    service_root = Path(__file__).parent
    config_path = service_root / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        dead_message_ttl = config.get('dramatiq', {}).get('dead_message_ttl', 604800000)  # 7 jours
    else:
        dead_message_ttl = 604800000
except:
    dead_message_ttl = 604800000  # 7 jours par défaut

redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", 6379))
redis_url = f"redis://{redis_host}:{redis_port}"
redis_broker = RedisBroker(url=redis_url, dead_message_ttl=dead_message_ttl)
result_backend = RedisBackend(url=redis_url)
redis_broker.add_middleware(Results(backend=result_backend))
dramatiq.set_broker(redis_broker)

print(f"INFO [ocr_service]: Broker Dramatiq configuré avec Dead Letter Queue (TTL: {dead_message_ttl}ms)")

# --- Variables globales pour le moteur OCR ---
_ocr_engine: Optional[PaddleOCR] = None
_init_lock = threading.Lock()

# Créer un fichier de verrou pour la synchronisation inter-processus
service_root = Path(__file__).parent
lock_dir = service_root / ".locks"
lock_dir.mkdir(exist_ok=True)
_lock_file_path = lock_dir / ".ocr_init.lock"
lock_timeout = 60  # 60 secondes
_process_lock = FileLock(_lock_file_path, timeout=lock_timeout)


def init_ocr_engine():
    """
    Initialise le moteur PaddleOCR une seule fois par processus.
    
    Cette fonction est thread-safe ET inter-processus-safe grâce à deux verrous :
    1. _process_lock (FileLock) : Garantit qu'un seul processus à la fois peut initialiser
    2. _init_lock (threading.Lock) : Garantit qu'un seul thread à la fois peut initialiser
    """
    global _ocr_engine
    
    # Vérification rapide : si déjà initialisé, on sort
    if _ocr_engine is not None:
        return _ocr_engine
    
    # Acquérir le verrou inter-processus
    with _process_lock:
        if _ocr_engine is None:
            with _init_lock:
                if _ocr_engine is None:
                    print(f"[OCR Worker PID: {os.getpid()}] Initialisation du moteur PaddleOCR...")
                    
                    # Charger la configuration
                    try:
                        import yaml
                        config_path = Path(__file__).parent / "config.yaml"
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config = yaml.safe_load(f)
                        ocr_config = config.get('ocr', {})
                    except Exception as e:
                        print(f"AVERTISSEMENT [ocr_service]: Impossible de charger config.yaml: {e}. Utilisation des valeurs par défaut.")
                        ocr_config = {}
                    
                    # Paramètres PaddleOCR
                    language = ocr_config.get('default_language', 'fr')
                    use_gpu = ocr_config.get('use_gpu', False)
                    runtime_options = ocr_config.get('runtime_options', {})
                    
                    device_str = 'gpu' if use_gpu else 'cpu'
                    
                    init_kwargs = {
                        "lang": language,
                        "use_textline_orientation": False,
                        "device": device_str
                    }
                    
                    if runtime_options:
                        init_kwargs.update(runtime_options)
                    
                    _ocr_engine = PaddleOCR(**init_kwargs)
                    
                    print(f"[OCR Worker PID: {os.getpid()}] Moteur PaddleOCR initialisé.")
    
    return _ocr_engine


def extract_ocr_lines(image_np: np.ndarray, max_dimension: int = 3500) -> List[Dict[str, Any]]:
    """
    Extrait les lignes de texte d'une image via PaddleOCR.
    
    Args:
        image_np: Image numpy array
        max_dimension: Dimension maximale (redimensionne si nécessaire)
        
    Returns:
        List[Dict[str, Any]]: Liste de lignes OCR avec text, confidence, bounding_box
    """
    global _ocr_engine
    
    # Initialiser le moteur si nécessaire
    init_ocr_engine()
    
    # Pré-traitement : Redimensionner si l'image est trop grande
    if image_np is not None and len(image_np.shape) >= 2:
        height, width = image_np.shape[:2]
        max_side = max(height, width)
        
        if max_side > max_dimension:
            scale = max_dimension / max_side
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            print(f"INFO [ocr_service]: Image trop grande ({width}x{height}), redimensionnement à ({new_width}x{new_height})")
            
            import cv2
            image_np = cv2.resize(image_np, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Utiliser le verrou pour protéger l'appel à PaddleOCR (pas thread-safe)
    try:
        with _init_lock:
            result = _ocr_engine.ocr(image_np)
    except Exception as e:
        print(f"ERREUR [ocr_service]: PaddleOCR a échoué: {type(e).__name__}: {e}")
        raise RuntimeError(f"PaddleOCR failed: {e}") from e
    
    if not result or not result[0]:
        return []
    
    # Parser le résultat de PaddleOCR
    lines = []
    ocr_data = result[0]
    
    # Format standard de PaddleOCR: [[[box], (text, confidence)], ...]
    if isinstance(ocr_data, list):
        for item in ocr_data:
            try:
                if item is None or len(item) != 2:
                    continue
                box_points, text_info = item
                
                if box_points is None:
                    continue
                
                if isinstance(text_info, (list, tuple)) and len(text_info) == 2:
                    text_content, confidence = text_info
                else:
                    continue
                
                # Convertir les coordonnées en liste de tuples
                bounding_box = []
                for point in box_points:
                    if point is not None and hasattr(point, '__iter__'):
                        bounding_box.append(tuple(map(int, point)))
                
                if len(bounding_box) == 0:
                    continue
                
                lines.append({
                    'text': text_content,
                    'confidence': float(confidence),
                    'bounding_box': bounding_box
                })
            except Exception as e:
                print(f"AVERTISSEMENT [ocr_service]: Erreur lors du parsing d'une ligne OCR: {e}")
                continue
    
    return lines


# --- Configuration des acteurs Dramatiq ---
try:
    import yaml
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        max_retries = config.get('dramatiq', {}).get('default_max_retries', 3)
        time_limit = config.get('dramatiq', {}).get('default_time_limit', 300_000)
    else:
        max_retries = 3
        time_limit = 300_000
except:
    max_retries = 3
    time_limit = 300_000  # 5 minutes


@dramatiq.actor(
    queue_name="ocr-queue",
    time_limit=time_limit,
    max_retries=max_retries,
    store_results=True
)
def perform_ocr_task(image_identifier: str, page_index: int = 0, **kwargs) -> Dict[str, Any]:
    """
    Acteur Dramatiq pour effectuer l'OCR sur une seule page d'image.
    
    C'est le cœur du microservice OCR isolé. Cet acteur :
    - Reçoit un identifiant de fichier (nom du fichier) au lieu d'une URI
    - Reconstruit le chemin complet à partir de son propre répertoire de stockage
    - Charge l'image depuis le chemin reconstruit
    - Exécute la reconnaissance OCR avec PaddleOCR
    - Gère les erreurs proprement
    - Retourne un résultat structuré avec les lignes de texte extraites
    
    Le modèle PaddleOCR est initialisé une seule fois par processus worker
    (variable globale _ocr_engine) pour optimiser les performances.
    
    Args:
        image_identifier: Identifiant de l'image (nom du fichier uniquement)
                         Le worker reconstruit le chemin complet à partir de son storage_dir.
        page_index: Index de la page (0-based), optionnel, par défaut 0
        **kwargs: Arguments additionnels passés par Dramatiq
        
    Returns:
        Dict[str, Any]: Résultat OCR pour cette page
            Format de succès: {
                'page_index': int,
                'ocr_lines': List[Dict[str, Any]],  # Liste de lignes avec text, confidence, bounding_box
                'status': 'success',
                'processing_time_seconds': float,
                'ocr_time_seconds': float,
                'num_lines': int,
                'worker_pid': int
            }
            Format d'erreur: {
                'page_index': int,
                'ocr_lines': [],
                'status': 'error',
                'error': str,
                'processing_time_seconds': float
            }
    """
    # Log de réception de la tâche
    worker_pid = os.getpid()
    print(f"INFO [ocr_service]: Tâche OCR reçue pour l'image '{image_identifier}' par le worker PID: {worker_pid}.")
    
    # Initialiser le moteur OCR
    init_ocr_engine()
    
    page_start_time = time.time()
    
    # Charger l'image
    # NOTE : On suppose que le conteneur a une variable d'env OCR_STORAGE_DIR
    # ou un chemin codé en dur qui correspond au volume Docker.
    # Par défaut, ce sera /app/data/temp_storage
    try:
        storage_dir = Path(os.getenv("OCR_STORAGE_DIR", "/app/data/temp_storage"))
        image_path = storage_dir / image_identifier
        
        # Log de reconstruction du chemin fichier
        print(f"DEBUG [ocr_service]: Tentative de chargement. Storage Dir: '{storage_dir}', Chemin complet: '{image_path}'.")
        
        # Log de vérification d'existence du fichier
        if not os.path.exists(image_path):
            print(f"ERROR [ocr_service]: Fichier image non trouvé au chemin: '{image_path}'. Vérifiez le montage du volume Docker.")
            raise FileNotFoundError(f"Impossible de charger l'image depuis le chemin reconstruit: {image_path}")
        
        # Charger l'image
        import cv2
        image_np = cv2.imread(str(image_path))
        
        # Log de chargement de l'image
        if image_np is None:
            print(f"ERROR [ocr_service]: cv2.imread a échoué pour le fichier '{image_path}'. Le fichier est peut-être corrompu ou dans un format non supporté.")
            raise FileNotFoundError(f"Impossible de charger l'image depuis le chemin reconstruit: {image_path}")
        else:
            print(f"DEBUG [ocr_service]: Image chargée avec succès. Dimensions: {image_np.shape}.")
            
    except FileNotFoundError as e:
        page_processing_time = time.time() - page_start_time
        print(f"ERREUR [ocr_service]: Image non trouvée pour la page {page_index + 1}: {e}")
        return {
            'page_index': page_index,
            'ocr_lines': [],
            'status': 'error',
            'error': f"Image non trouvée: {e}",
            'processing_time_seconds': round(page_processing_time, 3)
        }
    except Exception as e:
        page_processing_time = time.time() - page_start_time
        print(f"ERREUR [ocr_service]: Erreur lors du chargement de l'image pour la page {page_index + 1}: {e}")
        return {
            'page_index': page_index,
            'ocr_lines': [],
            'status': 'error',
            'error': f"Failed to load image: {e}",
            'processing_time_seconds': round(page_processing_time, 3)
        }
    
    # Effectuer l'OCR
    try:
        ocr_start_time = time.time()
        
        # Récupérer max_dimension depuis la config si disponible
        try:
            import yaml
            config_path = Path(__file__).parent / "config.yaml"
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            max_dimension = config.get('ocr', {}).get('max_image_dimension', 3500)
        except:
            max_dimension = 3500
        
        # Log d'exécution de l'OCR - début
        print(f"DEBUG [ocr_service]: Début de l'extraction OCR avec PaddleOCR.")
        
        ocr_lines = extract_ocr_lines(image_np, max_dimension=max_dimension)
        
        # Log d'exécution de l'OCR - fin
        print(f"DEBUG [ocr_service]: Fin de l'extraction OCR. {len(ocr_lines)} lignes trouvées.")
        
        ocr_time = time.time() - ocr_start_time
        page_processing_time = time.time() - page_start_time
        
        print(f"INFO [ocr_service]: Page {page_index + 1} traitée avec succès (OCR: {ocr_time:.3f}s, Total: {page_processing_time:.3f}s)")
        
        return {
            'page_index': page_index,
            'ocr_lines': ocr_lines,
            'status': 'success',
            'processing_time_seconds': round(page_processing_time, 3),
            'ocr_time_seconds': round(ocr_time, 3),
            'num_lines': len(ocr_lines),
            'worker_pid': worker_pid
        }
        
    except Exception as e:
        page_processing_time = time.time() - page_start_time
        print(f"ERROR [ocr_service]: Erreur durant l'appel à PaddleOCR: {type(e).__name__}: {e}")
        return {
            'page_index': page_index,
            'ocr_lines': [],
            'status': 'error',
            'error': f"OCR processing failed: {e}",
            'processing_time_seconds': round(page_processing_time, 3),
            'worker_pid': worker_pid
        }


@dramatiq.actor(
    queue_name="ocr-queue",
    time_limit=time_limit,
    max_retries=0,
    store_results=True  # Permettre de récupérer le résultat pour le health check
)
def warmup_ocr_worker(**kwargs) -> Dict[str, Any]:
    """
    Tâche de "warm-up" pour pré-initialiser le worker OCR.
    
    Cette tâche force l'initialisation du moteur PaddleOCR dans le worker
    qui l'exécute, ce qui est utile pour "chauffer" les nouveaux workers.
    
    Args:
        **kwargs: Arguments additionnels passés par Dramatiq
        
    Returns:
        Dict[str, Any]: Statut de l'initialisation
    """
    init_ocr_engine()
    
    worker_pid = os.getpid()
    engine_ready = _ocr_engine is not None
    
    print(f"INFO [ocr_service]: Worker warm-up terminé (PID: {worker_pid}, Engine ready: {engine_ready})")
    
    return {
        'status': 'ready',
        'worker_pid': worker_pid,
        'engine_initialized': engine_ready
    }

