"""
FastAPI application for document analysis
"""

import logging
import sys
import re
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from .routes import router
from .middleware import RequestIdMiddleware
from src.utils.logger import setup_logging
from src.config.manager import get_config_manager

# Initialize configuration manager (loads .env and config.yaml)
config = get_config_manager()

# Initialiser le logging au démarrage de l'application
setup_logging()

# Codes de couleur ANSI
class Colors:
    """Codes de couleur ANSI pour les logs"""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    # Couleurs pour les niveaux
    DEBUG = "\033[36m"  # Cyan
    INFO = "\033[32m"   # Vert
    WARNING = "\033[33m"  # Jaune
    ERROR = "\033[31m"   # Rouge
    CRITICAL = "\033[35m"  # Magenta
    
    # Couleurs pour les éléments de log
    TIMESTAMP = "\033[90m"  # Gris foncé
    NAME = "\033[94m"  # Bleu
    MESSAGE = "\033[97m"  # Blanc
    STATUS_SUCCESS = "\033[32m"  # Vert pour 2xx, 3xx
    STATUS_ERROR = "\033[31m"  # Rouge pour 4xx, 5xx


class ColoredFormatter(logging.Formatter):
    """
    Formatter personnalisé avec couleurs ANSI pour les logs Uvicorn.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        # Couleur selon le niveau
        level_colors = {
            logging.DEBUG: Colors.DEBUG,
            logging.INFO: Colors.INFO,
            logging.WARNING: Colors.WARNING,
            logging.ERROR: Colors.ERROR,
            logging.CRITICAL: Colors.CRITICAL,
        }
        level_color = level_colors.get(record.levelno, Colors.RESET)
        
        # Formater le timestamp
        timestamp = self.formatTime(record, self.datefmt)
        if hasattr(record, 'msecs'):
            timestamp = f"{timestamp}.{int(record.msecs):03d}"
        
        # Formater les éléments avec couleurs
        colored_timestamp = f"{Colors.TIMESTAMP}{timestamp}{Colors.RESET}"
        colored_level = f"{level_color}{record.levelname:<8}{Colors.RESET}"
        colored_name = f"{Colors.NAME}{record.name:<30}{Colors.RESET}"
        
        # Formater le message avec couleurs pour les codes HTTP
        message = record.getMessage()
        
        # Colorer les codes de statut HTTP dans les logs d'accès
        if "uvicorn.access" in record.name:
            # Pattern pour détecter les codes HTTP (ex: "202", "404", "500")
            def color_status(match):
                status = int(match.group(1))
                if 200 <= status < 400:
                    return f"{Colors.STATUS_SUCCESS}{status}{Colors.RESET}"
                else:
                    return f"{Colors.STATUS_ERROR}{status}{Colors.RESET}"
            
            message = re.sub(r'\b(\d{3})\b', color_status, message)
            # Colorer "INFO:" en vert
            message = re.sub(r'\bINFO:\s*', f"{Colors.INFO}INFO:{Colors.RESET} ", message)
        
        colored_message = f"{Colors.MESSAGE}{message}{Colors.RESET}"
        
        return f"{colored_timestamp} | {colored_level} | {colored_name} | {colored_message}"


# Configuration du format de log pour Uvicorn avec date/heure et couleurs
def get_uvicorn_log_config():
    """
    Retourne la configuration de logging pour Uvicorn avec format amélioré et couleurs.
    Cette fonction peut être utilisée même si l'application est lancée via 'uvicorn src.api.app:app'.
    
    Note: Uvicorn formate les logs d'accès avec des arguments positionnels, donc on utilise
    simplement le format standard qui affiche le message déjà formaté par Uvicorn.
    """
    # Vérifier si la sortie supporte les couleurs (terminal)
    use_colors = sys.stdout.isatty()
    
    # Créer la configuration de base
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-30s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "access": {
                "format": "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-30s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["access"],
                "level": "INFO",
                "propagate": False,
            },
        },
    }
    
    return log_config

app = FastAPI(
    title="Document Analysis API",
    description="API pour l'analyse de documents avec pipeline ML",
    version="0.1.0"
)


@app.on_event("startup")
async def setup_colored_logging():
    """Applique les formatters colorés aux loggers Uvicorn au démarrage."""
    if sys.stdout.isatty():
        colored_formatter = ColoredFormatter(
            fmt="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-30s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Appliquer aux handlers Uvicorn
        uvicorn_logger = logging.getLogger("uvicorn")
        uvicorn_error_logger = logging.getLogger("uvicorn.error")
        uvicorn_access_logger = logging.getLogger("uvicorn.access")
        
        for logger in [uvicorn_logger, uvicorn_error_logger]:
            for handler in logger.handlers:
                handler.setFormatter(colored_formatter)
        
        for handler in uvicorn_access_logger.handlers:
            handler.setFormatter(colored_formatter)

# Request ID middleware (doit être ajouté en premier pour capturer toutes les requêtes)
app.add_middleware(RequestIdMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus instrumentation
# Expose automatiquement un endpoint /metrics avec les métriques standards
Instrumentator().instrument(app).expose(app)

# Include routes
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "Document Analysis API"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    
    log_config = get_uvicorn_log_config()
    # Retirer la clé _apply_colors si elle existe (elle n'est plus nécessaire)
    log_config.pop("_apply_colors", None)
    
    # Get configuration from ConfigManager
    config = get_config_manager()
    
    uvicorn.run(
        app,
        host=config.get("env", "api", "host", default="0.0.0.0"),
        port=config.get("env", "api", "port", default=8000),
        reload=config.get("env", "api", "debug", default=False),
        log_config=log_config
    )

