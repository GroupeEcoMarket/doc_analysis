"""
Utilitaires partagés pour l'initialisation des workers dans le multiprocessing.

Ce module centralise la logique commune d'initialisation des workers pour éviter
la duplication de code entre différents modules (API routes, geometry pipeline, etc.).
"""

import os
from typing import Dict, Any, Optional
from dataclasses import asdict

from src.utils.logger import get_logger
from src.utils.config_loader import (
    get_config, Config, GeometryConfig, QAConfig, PerformanceConfig, OutputConfig
)

logger = get_logger(__name__)


def get_config_dicts_from_config(config: Config) -> Dict[str, Dict[str, Any]]:
    """
    Convertit un objet Config en dictionnaires sérialisables pour le multiprocessing.
    
    Cette fonction centralise la conversion des configurations en dictionnaires,
    ce qui est nécessaire car les objets de configuration ne sont pas directement
    sérialisables pour le multiprocessing.
    
    Args:
        config: Objet Config à convertir
        
    Returns:
        dict: Dictionnaire contenant les configurations sérialisées:
            - 'geo_config': GeometryConfig en dict
            - 'qa_config': QAConfig en dict
            - 'perf_config': PerformanceConfig en dict
            - 'output_config': OutputConfig en dict
    """
    return {
        'geo_config': asdict(config.geometry),
        'qa_config': asdict(config.qa),
        'perf_config': asdict(config.performance),
        'output_config': asdict(config.output)
    }


def log_worker_initialization(worker_type: str, pid: Optional[int] = None) -> None:
    """
    Log standardisé pour l'initialisation d'un worker.
    
    Args:
        worker_type: Type de worker (ex: 'api', 'geometry')
        pid: Process ID (optionnel, sera récupéré automatiquement si None)
    """
    if pid is None:
        pid = os.getpid()
    logger.debug(f"Worker {worker_type} initialisé avec PID: {pid}")


def create_geometry_normalizer_from_dicts(
    geo_config_dict: Dict[str, Any],
    qa_config_dict: Dict[str, Any],
    perf_config_dict: Dict[str, Any],
    output_config_dict: Dict[str, Any]
) -> "GeometryNormalizer":
    """
    Crée un GeometryNormalizer depuis des dictionnaires de configuration.
    
    Cette fonction centralise la création d'un GeometryNormalizer depuis
    des configurations sérialisées, ce qui est nécessaire pour le multiprocessing.
    
    Args:
        geo_config_dict: Dictionnaire de configuration géométrique
        qa_config_dict: Dictionnaire de configuration QA
        perf_config_dict: Dictionnaire de configuration de performance
        output_config_dict: Dictionnaire de configuration de sortie
        
    Returns:
        GeometryNormalizer: Instance du normalizer initialisée
    """
    # Reconstruire les objets de configuration depuis les dictionnaires
    geo_config = GeometryConfig(**geo_config_dict)
    qa_config = QAConfig(**qa_config_dict)
    perf_config = PerformanceConfig(**perf_config_dict)
    output_config = OutputConfig(**output_config_dict)
    
    # Initialiser le ModelRegistry dans le worker
    # Le ModelRegistry doit être initialisé ici car il n'est pas partageable entre les processus
    from src.models.registry import ModelRegistry
    model_registry = ModelRegistry(lazy_load=perf_config.lazy_load_models)
    
    # Importer GeometryNormalizer ici pour éviter les imports circulaires
    from src.pipeline.geometry import GeometryNormalizer
    
    # Créer le normalizer
    return GeometryNormalizer(
        geo_config=geo_config,
        qa_config=qa_config,
        perf_config=perf_config,
        output_config=output_config,
        model_registry=model_registry
    )


def create_api_dependencies_from_config(config: Optional[Config] = None) -> Dict[str, Any]:
    """
    Crée les dépendances API (FeatureExtractor, DocumentClassifier) depuis une configuration.
    
    Cette fonction centralise la création des dépendances API pour le multiprocessing.
    
    Args:
        config: Configuration (optionnel, sera chargée si None)
        
    Returns:
        dict: Dictionnaire contenant les dépendances:
            - 'feature_extractor': FeatureExtractor
            - 'document_classifier': DocumentClassifier (peut être None si désactivé)
    """
    if config is None:
        config = get_config()
    
    from src.api.dependencies import get_feature_extractor, get_document_classifier
    
    dependencies = {}
    dependencies['feature_extractor'] = get_feature_extractor(config)
    
    try:
        dependencies['document_classifier'] = get_document_classifier(config)
    except ValueError:
        # Gérer le cas où la classification est désactivée
        dependencies['document_classifier'] = None
        logger.debug("Classification désactivée, document_classifier non initialisé")
    
    return dependencies

