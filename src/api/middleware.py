"""
Middleware for FastAPI application
"""

import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.utils.logger import set_request_id, get_logger

logger = get_logger(__name__)


class RequestIdMiddleware(BaseHTTPMiddleware):
    """
    Middleware qui génère et propage un request_id pour chaque requête.
    
    Le request_id est :
    - Généré automatiquement pour chaque requête (UUID)
    - Ajouté dans le header de réponse 'X-Request-ID'
    - Stocké dans le contexte pour être accessible dans tous les logs
    - Peut être fourni par le client via le header 'X-Request-ID'
    """
    
    def __init__(self, app: ASGIApp):
        """
        Initialise le middleware.
        
        Args:
            app: Application ASGI
        """
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next):
        """
        Traite chaque requête en générant/propagant le request_id.
        
        Args:
            request: Requête HTTP
            call_next: Fonction pour appeler le prochain middleware/handler
            
        Returns:
            Response: Réponse HTTP avec header X-Request-ID
        """
        # Vérifier si le client a fourni un request_id
        client_request_id = request.headers.get("X-Request-ID")
        
        # Générer un nouveau request_id ou utiliser celui du client
        if client_request_id:
            request_id = client_request_id
        else:
            request_id = str(uuid.uuid4())
        
        # Définir le request_id dans le contexte pour les logs
        set_request_id(request_id)
        
        # Logger le début de la requête
        logger.info(
            f"Requête reçue: {request.method} {request.url.path}",
            extra={"request_id": request_id}
        )
        
        try:
            # Appeler le prochain middleware/handler
            response = await call_next(request)
            
            # Ajouter le request_id dans le header de réponse
            response.headers["X-Request-ID"] = request_id
            
            # Logger la fin de la requête
            logger.info(
                f"Réponse envoyée: {request.method} {request.url.path} - Status: {response.status_code}",
                extra={"request_id": request_id}
            )
            
            return response
            
        except Exception as e:
            # Logger les erreurs avec le request_id
            logger.error(
                f"Erreur lors du traitement de la requête: {str(e)}",
                exc_info=True,
                extra={"request_id": request_id}
            )
            raise
        finally:
            # Nettoyer le contexte (optionnel, mais bon pour la clarté)
            # Le contexte sera automatiquement nettoyé à la fin de la requête
            pass

