"""
Configuration du logging structuré pour l'application
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from contextvars import ContextVar
from typing import Optional
from src.config.manager import get_config_manager

# Context variable pour stocker le request_id dans le contexte de chaque requête
request_id_context: ContextVar[Optional[str]] = ContextVar('request_id', default=None)


def setup_logging():
    """
    Configure le système de logging de l'application.
    
    Le niveau de log et le fichier de log sont configurés via la configuration:
    - env.logging.level: Niveau de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - env.logging.file: Chemin vers le fichier de log (optionnel)
    
    Si env.logging.file n'est pas défini, les logs ne sont écrits que dans la console.
    """
    # Récupérer la configuration depuis ConfigManager
    config = get_config_manager()
    log_level_str = config.get("env", "logging", "level", default="INFO").upper()
    log_file = config.get("env", "logging", "file", default=None)
    
    # Convertir le niveau de log en constante logging
    log_level = getattr(logging, log_level_str, logging.INFO)
    
    # Format standard pour les logs avec request_id, thread ID et process ID
    # Le PID est particulièrement utile pour déboguer les workers Dramatiq multiprocessing
    # Format amélioré avec millisecondes pour plus de précision
    log_format = "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-30s | [request_id=%(request_id)s] [thread=%(thread)d] [pid=%(process)d] | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Configuration du formatter personnalisé
    formatter = RequestIdFormatter(log_format, datefmt=date_format)
    
    # Configuration du handler console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Configuration du handler fichier (si spécifié)
    handlers = [console_handler]
    
    if log_file:
        # Créer le répertoire de logs si nécessaire
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Utiliser RotatingFileHandler pour limiter la taille des logs
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configuration du logger racine
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        format=log_format,
        datefmt=date_format
    )
    
    # Réduire le niveau de log pour certaines bibliothèques tierces
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


class RequestIdFormatter(logging.Formatter):
    """
    Formatter personnalisé qui inclut automatiquement le request_id dans les logs.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Formate le log en incluant le request_id depuis le contexte.
        
        Args:
            record: LogRecord à formater
            
        Returns:
            str: Message de log formaté
        """
        # Récupérer le request_id depuis le contexte
        request_id = request_id_context.get()
        
        # Ajouter le request_id au record
        record.request_id = request_id if request_id else "N/A"
        
        return super().format(record)


def set_request_id(request_id: str) -> None:
    """
    Définit le request_id dans le contexte actuel.
    
    Args:
        request_id: Identifiant de corrélation pour la requête
    """
    request_id_context.set(request_id)


def get_request_id() -> Optional[str]:
    """
    Récupère le request_id depuis le contexte actuel.
    
    Returns:
        str ou None: Identifiant de corrélation actuel
    """
    return request_id_context.get()


def get_logger(name: str) -> logging.Logger:
    """
    Récupère un logger pour un module donné.
    Le logger inclut automatiquement le request_id dans tous les messages de log.
    
    Args:
        name: Nom du module (généralement __name__)
        
    Returns:
        Instance de logging.Logger configurée avec support du request_id
    """
    # S'assurer que le logging est configuré
    if not logging.root.handlers:
        setup_logging()
    
    return logging.getLogger(name)

