"""
Model Registry for centralized ML model management.

This module provides a ModelRegistry class that centralizes the loading
and management of ML models, enabling lazy loading and dependency injection.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from onnxtr.models import from_hub
from onnxtr.models.classification.zoo import page_orientation_predictor
from doctr.models import detection

from src.pipeline.geometry import OrientationModelAdapter
from src.utils.logger import get_logger
from src.utils.exceptions import ModelLoadingError

logger = get_logger(__name__)


class ModelType(str, Enum):
    """Types de modèles disponibles dans le registre."""
    ORIENTATION = "orientation"
    DETECTION = "detection"


@dataclass
class ModelConfig:
    """Configuration pour un modèle."""
    model_type: ModelType
    name: str
    source: str  # "huggingface", "doctr", "local", etc.
    identifier: str  # ID du modèle (repo HuggingFace, nom doctr, chemin local)
    version: Optional[str] = None
    lazy_load: bool = True
    description: Optional[str] = None


class ModelRegistry:
    """
    Registre centralisé pour la gestion des modèles ML.
    
    Ce registre permet de :
    - Charger les modèles de manière lazy
    - Gérer les versions de modèles
    - Partager les modèles entre plusieurs services
    - Faciliter les tests en permettant l'injection de modèles mock
    """
    
    def __init__(self, lazy_load: bool = True):
        """
        Initialise le registre de modèles.
        
        Args:
            lazy_load: Si True, les modèles sont chargés à la demande (lazy loading).
                      Si False, tous les modèles sont chargés au démarrage.
        """
        # Cache des modèles chargés
        self._models: Dict[str, Any] = {}
        self._adapters: Dict[str, Any] = {}
        
        # Configuration des modèles par défaut
        self._model_configs: Dict[ModelType, ModelConfig] = {
            ModelType.ORIENTATION: ModelConfig(
                model_type=ModelType.ORIENTATION,
                name="page-orientation",
                source="huggingface",
                identifier="Felix92/onnxtr-mobilenet-v3-small-page-orientation",
                lazy_load=lazy_load,
                description="Modèle de détection d'orientation de page (0°, 90°, 180°, 270°)"
            ),
            ModelType.DETECTION: ModelConfig(
                model_type=ModelType.DETECTION,
                name="page-detection",
                source="doctr",
                identifier="db_resnet50",
                lazy_load=lazy_load,
                description="Modèle de détection de pages pour le crop intelligent"
            )
        }
        
        # Si lazy_load=False, charger tous les modèles maintenant
        if not lazy_load:
            logger.info("Chargement de tous les modèles au démarrage (lazy_load=False)")
            try:
                self.get_orientation_adapter()
            except Exception as e:
                logger.warning(f"Impossible de précharger le modèle d'orientation: {e}")
            try:
                self.get_detection_model()
            except Exception as e:
                logger.warning(f"Impossible de précharger le modèle de détection: {e}")
    
    def register_model(
        self,
        model_type: ModelType,
        config: ModelConfig
    ) -> None:
        """
        Enregistre une configuration de modèle.
        
        Args:
            model_type: Type de modèle
            config: Configuration du modèle
        """
        self._model_configs[model_type] = config
        logger.info(f"Modèle {model_type.value} enregistré: {config.name}")
    
    def get_orientation_adapter(
        self,
        force_reload: bool = False
    ) -> OrientationModelAdapter:
        """
        Récupère l'adaptateur du modèle d'orientation.
        
        Args:
            force_reload: Si True, recharge le modèle même s'il est déjà chargé
            
        Returns:
            OrientationModelAdapter: L'adaptateur du modèle d'orientation
            
        Raises:
            ModelLoadingError: Si le chargement du modèle échoue
        """
        model_key = ModelType.ORIENTATION.value
        
        # Vérifier si l'adaptateur est déjà en cache
        if not force_reload and model_key in self._adapters:
            return self._adapters[model_key]
        
        try:
            config = self._model_configs[ModelType.ORIENTATION]
            
            # Vérifier si le lazy loading est activé
            if config.lazy_load and model_key in self._models and not force_reload:
                # Le modèle est déjà chargé et lazy_load est activé, retourner l'adaptateur existant
                if model_key in self._adapters:
                    return self._adapters[model_key]
            
            logger.info(f"Chargement du modèle d'orientation: {config.identifier}")
            
            # Charger le modèle brut
            if model_key not in self._models or force_reload:
                if config.source == "huggingface":
                    raw_model = from_hub(config.identifier)
                    self._models[model_key] = page_orientation_predictor(arch=raw_model)
                else:
                    raise ModelLoadingError(
                        f"Source de modèle non supportée pour l'orientation: {config.source}"
                    )
            
            # Créer l'adaptateur
            orientation_model = self._models[model_key]
            adapter = OrientationModelAdapter(orientation_model)
            self._adapters[model_key] = adapter
            
            logger.info(f"Modèle d'orientation chargé avec succès")
            return adapter
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle d'orientation: {e}", exc_info=True)
            raise ModelLoadingError(f"Impossible de charger le modèle d'orientation: {e}") from e
    
    def get_detection_model(
        self,
        force_reload: bool = False
    ) -> Any:
        """
        Récupère le modèle de détection de pages.
        
        Args:
            force_reload: Si True, recharge le modèle même s'il est déjà chargé
            
        Returns:
            Modèle de détection doctr
            
        Raises:
            ModelLoadingError: Si le chargement du modèle échoue
        """
        model_key = ModelType.DETECTION.value
        
        # Vérifier si le modèle est déjà en cache
        if not force_reload and model_key in self._models:
            return self._models[model_key]
        
        try:
            config = self._model_configs[ModelType.DETECTION]
            
            # Vérifier si le lazy loading est activé
            if config.lazy_load and model_key in self._models and not force_reload:
                # Le modèle est déjà chargé et lazy_load est activé, retourner le modèle existant
                return self._models[model_key]
            
            logger.info(f"Chargement du modèle de détection: {config.identifier}")
            
            if config.source == "doctr":
                self._models[model_key] = detection.detection_predictor(
                    arch=config.identifier,
                    pretrained=True
                )
            else:
                raise ModelLoadingError(
                    f"Source de modèle non supportée pour la détection: {config.source}"
                )
            
            logger.info(f"Modèle de détection chargé avec succès")
            return self._models[model_key]
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle de détection: {e}", exc_info=True)
            raise ModelLoadingError(f"Impossible de charger le modèle de détection: {e}") from e
    
    def unload_model(self, model_type: ModelType) -> None:
        """
        Décharge un modèle de la mémoire.
        
        Args:
            model_type: Type de modèle à décharger
        """
        model_key = model_type.value
        
        if model_key in self._models:
            del self._models[model_key]
            logger.info(f"Modèle {model_key} déchargé")
        
        if model_key in self._adapters:
            del self._adapters[model_key]
            logger.info(f"Adaptateur {model_key} déchargé")
    
    def unload_all(self) -> None:
        """Décharge tous les modèles de la mémoire."""
        self._models.clear()
        self._adapters.clear()
        logger.info("Tous les modèles ont été déchargés")
    
    def get_model_info(self, model_type: ModelType) -> Dict[str, Any]:
        """
        Récupère les informations sur un modèle.
        
        Args:
            model_type: Type de modèle
            
        Returns:
            dict: Informations sur le modèle
        """
        config = self._model_configs.get(model_type)
        if not config:
            return {}
        
        return {
            "type": model_type.value,
            "name": config.name,
            "source": config.source,
            "identifier": config.identifier,
            "version": config.version,
            "lazy_load": config.lazy_load,
            "description": config.description,
            "loaded": model_type.value in self._models
        }
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Liste tous les modèles enregistrés.
        
        Returns:
            dict: Dictionnaire avec les informations de tous les modèles
        """
        return {
            model_type.value: self.get_model_info(model_type)
            for model_type in self._model_configs.keys()
        }

