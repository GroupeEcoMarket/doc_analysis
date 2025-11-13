"""
Client OCR pour communiquer avec le microservice OCR isolé.

Ce module définit un acteur Dramatiq "proxy" qui permet à l'application principale
d'envoyer des tâches OCR au microservice isolé via la queue dédiée ocr-queue.

L'acteur proxy n'a pas d'implémentation - il sert uniquement de point d'entrée
pour envoyer des messages à la queue du microservice OCR.
"""

import dramatiq
from dramatiq.brokers.redis import RedisBroker
from dramatiq.results import Results
from dramatiq.results.backends import RedisBackend

from src.config.manager import get_config_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)

# --- Configuration du broker Dramatiq (même que l'application principale) ---
try:
    config = get_config_manager()
    dead_message_ttl = config.get('dramatiq', 'dead_message_ttl', default=604800000)  # 7 jours
    # Configuration Redis depuis les variables d'environnement
    redis_host = config.get('env', 'redis', 'host', default='localhost')
    redis_port = config.get('env', 'redis', 'port', default=6379)
    redis_url = f"redis://{redis_host}:{redis_port}"
    redis_broker = RedisBroker(url=redis_url, dead_message_ttl=dead_message_ttl)
    result_backend = RedisBackend(url=redis_url)
except Exception as e:
    logger.warning(f"Erreur lors de la configuration Redis, utilisation des valeurs par défaut: {e}")
    # Fallback vers les valeurs par défaut
    dead_message_ttl = 604800000  # 7 jours par défaut
    redis_broker = RedisBroker(dead_message_ttl=dead_message_ttl)
    result_backend = RedisBackend()

redis_broker.add_middleware(Results(backend=result_backend))
dramatiq.set_broker(redis_broker)

logger.debug("Client OCR: Broker Dramatiq configuré")


# --- Acteur Proxy pour le microservice OCR ---
# Cet acteur pointe vers la même queue que le microservice OCR (ocr-queue)
# mais n'a pas d'implémentation. Il sert uniquement à envoyer des messages.
# Le microservice OCR (services/ocr_service/actors.py) écoute cette queue
# et exécute réellement les tâches.

try:
    config = get_config_manager()
    queue_name = config.get('ocr_service', 'queue_name', default='ocr-queue')
    time_limit = config.get('ocr_service', 'timeout_ms', default=30000)  # 30 secondes par défaut
    max_retries = config.get('ocr_service', 'max_retries', default=3)
except:
    queue_name = 'ocr-queue'
    time_limit = 30000  # 30 secondes
    max_retries = 3


@dramatiq.actor(
    queue_name=queue_name,
    time_limit=time_limit,
    max_retries=max_retries,
    store_results=True
)
def perform_ocr_task(image_identifier: str, page_index: int = 0, **kwargs):
    """
    Acteur proxy pour envoyer des tâches OCR au microservice isolé.
    
    Cet acteur ne contient pas d'implémentation - il sert uniquement de point
    d'entrée pour envoyer des messages à la queue ocr-queue. Le microservice
    OCR isolé (services/ocr_service/actors.py) écoute cette queue et exécute
    réellement les tâches.
    
    Args:
        image_identifier: Identifiant de l'image (nom du fichier uniquement)
                         Le worker reconstruit le chemin complet à partir de son storage_dir.
        page_index: Index de la page (0-based), optionnel
        **kwargs: Arguments additionnels
        
    Returns:
        Dict[str, Any]: Résultat OCR du microservice
    """
    # Cette fonction ne devrait jamais être appelée directement
    # Elle sert uniquement de proxy pour envoyer des messages via .send()
    raise NotImplementedError(
        "perform_ocr_task est un acteur proxy. "
        "Utilisez perform_ocr_task.send() pour envoyer des tâches au microservice OCR."
    )


@dramatiq.actor(
    queue_name=queue_name,
    time_limit=30000,  # 30 secondes pour le warm-up
    max_retries=0,
    store_results=True  # Permettre de récupérer le résultat pour le health check
)
def warmup_ocr_worker(**kwargs):
    """
    Acteur proxy pour envoyer une tâche de warm-up au microservice OCR isolé.
    
    Cet acteur ne contient pas d'implémentation - il sert uniquement de point
    d'entrée pour envoyer des messages à la queue ocr-queue. Le microservice
    OCR isolé (services/ocr_service/actors.py) écoute cette queue et exécute
    réellement les tâches.
    
    Args:
        **kwargs: Arguments additionnels
        
    Returns:
        Dict[str, Any]: Statut de l'initialisation du moteur OCR
    """
    # Cette fonction ne devrait jamais être appelée directement
    # Elle sert uniquement de proxy pour envoyer des messages via .send()
    raise NotImplementedError(
        "warmup_ocr_worker est un acteur proxy. "
        "Utilisez warmup_ocr_worker.send() pour envoyer une tâche de warm-up au microservice OCR."
    )

