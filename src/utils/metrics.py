"""
Module de métriques Prometheus pour l'observabilité du système.

Ce module expose des métriques custom pour :
- Le nombre de pages traitées (succès/échec) par type de document
- La durée de traitement des documents
- La taille des files d'attente Dramatiq
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
from typing import Optional
import threading
import time
from src.utils.logger import get_logger

logger = get_logger(__name__)

# --- Métriques Custom ---

# Counter pour le nombre de pages traitées
pages_processed_total = Counter(
    'pages_processed_total',
    'Nombre total de pages traitées',
    ['status', 'doc_type']  # Labels: status (success/failed), doc_type (Facture/Attestation_CEE/etc.)
)

# Histogram pour la durée de traitement des documents
document_processing_duration_seconds = Histogram(
    'document_processing_duration_seconds',
    'Durée de traitement d\'un document en secondes',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]  # Buckets en secondes
)

# Gauge pour la taille des files d'attente Dramatiq
dramatiq_queue_size = Gauge(
    'dramatiq_queue_size',
    'Taille actuelle de la file d\'attente Dramatiq',
    ['queue_name']  # Label: queue_name (default/ocr-queue/etc.)
)

# Counter pour le nombre de documents traités
documents_processed_total = Counter(
    'documents_processed_total',
    'Nombre total de documents traités',
    ['status']  # Labels: status (success/partial_success/error)
)

# Histogram pour la durée de traitement d'une page
page_processing_duration_seconds = Histogram(
    'page_processing_duration_seconds',
    'Durée de traitement d\'une page en secondes',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

# Counter pour les erreurs par type
processing_errors_total = Counter(
    'processing_errors_total',
    'Nombre total d\'erreurs de traitement',
    ['error_type']  # Labels: error_type (ocr_failed/classification_error/etc.)
)


class QueueSizeMonitor:
    """
    Moniteur pour mettre à jour périodiquement la taille des files d'attente Dramatiq.
    """
    
    def __init__(self, broker, update_interval: float = 10.0):
        """
        Args:
            broker: Broker Dramatiq (RedisBroker)
            update_interval: Intervalle de mise à jour en secondes (défaut: 10s)
        """
        self.broker = broker
        self.update_interval = update_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._consecutive_errors = 0
        self._max_consecutive_errors = 5  # Arrêter après 5 erreurs consécutives
        self._disabled = False
    
    def start(self):
        """Démarre le moniteur en arrière-plan."""
        with self._lock:
            if self._running:
                logger.warning("QueueSizeMonitor est déjà en cours d'exécution")
                return
            
            self._running = True
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()
            logger.info(f"QueueSizeMonitor démarré (intervalle: {self.update_interval}s)")
    
    def stop(self):
        """Arrête le moniteur."""
        with self._lock:
            if not self._running:
                return
            
            self._running = False
            if self._thread:
                self._thread.join(timeout=5.0)
            logger.info("QueueSizeMonitor arrêté")
    
    def _monitor_loop(self):
        """Boucle principale du moniteur."""
        while self._running:
            try:
                if self._disabled:
                    # Si désactivé, réessayer périodiquement (toutes les 30 secondes)
                    # pour voir si Redis est redevenu disponible
                    time.sleep(30.0)
                    # Réessayer une fois pour voir si Redis est disponible
                    try:
                        self._update_queue_sizes()
                        # Si succès, réactiver le monitoring
                        self._disabled = False
                        self._consecutive_errors = 0
                        logger.info("QueueSizeMonitor réactivé - Redis est maintenant disponible")
                    except Exception:
                        # Redis toujours indisponible, continuer à attendre
                        continue
                else:
                    self._update_queue_sizes()
                    # Réinitialiser le compteur d'erreurs en cas de succès
                    self._consecutive_errors = 0
            except Exception as e:
                self._consecutive_errors += 1
                
                # Si trop d'erreurs consécutives, désactiver le monitoring
                if self._consecutive_errors >= self._max_consecutive_errors:
                    if not self._disabled:
                        self._disabled = True
                        logger.warning(
                            f"QueueSizeMonitor désactivé après {self._max_consecutive_errors} erreurs consécutives. "
                            f"Redis n'est probablement pas disponible. "
                            f"Vérifiez votre configuration REDIS_HOST (actuellement: {self._get_redis_host()}). "
                            f"Le monitoring reprendra automatiquement si Redis redevient disponible."
                        )
                elif self._consecutive_errors == 1:
                    # Logger seulement la première erreur pour éviter le spam
                    logger.debug(
                        f"Erreur de connexion Redis lors de la mise à jour des tailles de file: {e}. "
                        f"Tentatives: {self._consecutive_errors}/{self._max_consecutive_errors}"
                    )
            
            # Attendre avant la prochaine mise à jour
            if not self._disabled:
                time.sleep(self.update_interval)
    
    def _get_redis_host(self) -> str:
        """Récupère le host Redis depuis la configuration pour les messages d'erreur."""
        try:
            from src.config.manager import get_config_manager
            config = get_config_manager()
            return config.get('env', 'redis', 'host', default='non configuré')
        except:
            return 'non configuré'
    
    def _update_queue_sizes(self):
        """Met à jour les métriques de taille de file d'attente."""
        from redis.exceptions import ConnectionError as RedisConnectionError
        
        try:
            redis_client = self.broker.client
            
            # Obtenir toutes les files d'attente Dramatiq
            # Les files sont stockées dans Redis avec le préfixe "dramatiq:queue:"
            queue_pattern = "dramatiq:queue:*"
            queue_keys = redis_client.keys(queue_pattern)
            
            # Extraire les noms de files
            queues = {}
            for key in queue_keys:
                if isinstance(key, bytes):
                    key = key.decode('utf-8')
                
                # Format: dramatiq:queue:queue_name
                parts = key.split(':')
                if len(parts) >= 3:
                    queue_name = ':'.join(parts[2:])  # Gérer les noms avec ':'
                    if queue_name not in queues:
                        queues[queue_name] = []
                    queues[queue_name].append(key)
            
            # Mettre à jour les métriques pour chaque file
            for queue_name, keys in queues.items():
                total_size = 0
                for key in keys:
                    try:
                        size = redis_client.llen(key)  # Longueur de la liste Redis
                        total_size += size
                    except Exception as e:
                        logger.debug(f"Erreur lors de la lecture de la taille de {key}: {e}")
                
                # Mettre à jour la gauge
                dramatiq_queue_size.labels(queue_name=queue_name).set(total_size)
            
            # Si aucune file n'a été trouvée, mettre à jour avec 0 pour la file par défaut
            if not queues:
                dramatiq_queue_size.labels(queue_name='default').set(0)
                
        except RedisConnectionError as e:
            # Relancer l'exception pour qu'elle soit gérée par _monitor_loop
            raise
        except Exception as e:
            # Autres erreurs (non-connexion) : logger en debug seulement
            logger.debug(f"Erreur lors de la mise à jour des tailles de file: {e}")
            # Ne pas relancer pour éviter de spammer les logs


# Instance globale du moniteur (sera initialisée dans les workers)
_queue_monitor: Optional[QueueSizeMonitor] = None


def init_queue_monitor(broker, update_interval: float = 10.0):
    """
    Initialise et démarre le moniteur de taille de file d'attente.
    
    Args:
        broker: Broker Dramatiq (RedisBroker)
        update_interval: Intervalle de mise à jour en secondes (défaut: 10s)
    """
    global _queue_monitor
    
    if _queue_monitor is not None:
        logger.warning("QueueSizeMonitor déjà initialisé")
        return
    
    _queue_monitor = QueueSizeMonitor(broker, update_interval)
    _queue_monitor.start()
    logger.info("QueueSizeMonitor initialisé et démarré")


def stop_queue_monitor():
    """Arrête le moniteur de taille de file d'attente."""
    global _queue_monitor
    
    if _queue_monitor is not None:
        _queue_monitor.stop()
        _queue_monitor = None


def get_metrics():
    """
    Retourne les métriques Prometheus au format texte.
    
    Returns:
        bytes: Métriques au format Prometheus
    """
    return generate_latest(REGISTRY)


def start_metrics_server(port: int = 9090, host: str = '0.0.0.0'):
    """
    Démarre un serveur HTTP simple pour exposer les métriques Prometheus.
    
    Cette fonction démarre un serveur HTTP dans un thread séparé qui expose
    les métriques sur l'endpoint /metrics. Utile pour les workers Dramatiq
    qui n'ont pas de serveur HTTP intégré.
    
    IMPORTANT: Si plusieurs processus workers démarrent, seul le premier pourra
    utiliser le port. Les autres échoueront silencieusement. C'est normal car
    tous les workers partagent les mêmes métriques Prometheus via le registre global.
    
    Args:
        port: Port sur lequel exposer les métriques (défaut: 9090)
        host: Adresse IP sur laquelle écouter (défaut: 0.0.0.0)
    
    Returns:
        threading.Thread: Thread du serveur HTTP, ou None si le démarrage a échoué
    """
    import socket
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from prometheus_client import CONTENT_TYPE_LATEST
    
    class MetricsHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/metrics' or self.path == '/metrics/':
                self.send_response(200)
                self.send_header('Content-Type', CONTENT_TYPE_LATEST)
                self.end_headers()
                try:
                    metrics_data = generate_latest(REGISTRY)
                    self.wfile.write(metrics_data)
                except Exception as e:
                    logger.error(f"Erreur lors de la génération des métriques: {e}", exc_info=True)
                    self.send_response(500)
                    self.end_headers()
            elif self.path == '/health' or self.path == '/health/':
                self.send_response(200)
                self.send_header('Content-Type', 'text/plain')
                self.end_headers()
                self.wfile.write(b'OK')
            else:
                self.send_response(404)
                self.send_header('Content-Type', 'text/plain')
                self.end_headers()
                self.wfile.write(b'Not Found')
        
        def log_message(self, format, *args):
            # Désactiver les logs du serveur HTTP pour éviter le bruit
            pass
    
    def run_server():
        try:
            # Vérifier si le port est déjà utilisé
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                test_socket.bind((host, port))
                test_socket.close()
            except OSError as e:
                if e.errno == 98 or e.errno == 10048:  # Port déjà utilisé (Linux/Windows)
                    logger.warning(
                        f"Le port {port} est déjà utilisé. Le serveur de métriques ne démarrera pas dans ce processus. "
                        f"C'est normal si plusieurs workers sont en cours d'exécution - un seul serveur suffit."
                    )
                    return
                else:
                    raise
            
            server = HTTPServer((host, port), MetricsHandler)
            logger.info(f"Serveur de métriques démarré sur http://{host}:{port}/metrics")
            server.serve_forever()
        except OSError as e:
            if e.errno == 98 or e.errno == 10048:  # Port déjà utilisé
                logger.warning(
                    f"Le port {port} est déjà utilisé. Le serveur de métriques ne démarrera pas dans ce processus. "
                    f"C'est normal si plusieurs workers sont en cours d'exécution."
                )
            else:
                logger.error(f"Erreur lors du démarrage du serveur de métriques: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Erreur dans le serveur de métriques: {e}", exc_info=True)
    
    server_thread = threading.Thread(target=run_server, daemon=True, name="MetricsServer")
    server_thread.start()
    
    # Attendre un peu pour voir si le serveur démarre correctement
    import time
    time.sleep(0.1)
    
    return server_thread

