"""
Dependency Injection for FastAPI application.

This module provides FastAPI dependencies for injecting configuration
and services into route handlers, following the Dependency Injection pattern.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from src.utils.config_loader import get_config, Config, GeometryConfig, QAConfig
from src.pipeline.preprocessing import PreprocessingNormalizer
from src.pipeline.geometry import GeometryNormalizer
from src.utils.capture_classifier import CaptureClassifier
from src.models.registry import ModelRegistry


# ==========================================
# Configuration Dependencies
# ==========================================

@lru_cache()
def get_app_config() -> Config:
    """
    Récupère la configuration de l'application (singleton).
    Cette fonction est mise en cache pour éviter de recharger la config à chaque requête.
    
    Returns:
        Config: Configuration de l'application
    """
    return get_config()


def get_geometry_config(
    app_config: Annotated[Config, Depends(get_app_config)]
) -> GeometryConfig:
    """
    Récupère la configuration géométrique.
    
    Args:
        app_config: Configuration de l'application (injectée)
        
    Returns:
        GeometryConfig: Configuration géométrique
    """
    return app_config.geometry


def get_qa_config(
    app_config: Annotated[Config, Depends(get_app_config)]
) -> QAConfig:
    """
    Récupère la configuration QA.
    
    Args:
        app_config: Configuration de l'application (injectée)
        
    Returns:
        QAConfig: Configuration QA
    """
    return app_config.qa


# ==========================================
# Service Dependencies
# ==========================================

def get_capture_classifier(
    geo_config: Annotated[GeometryConfig, Depends(get_geometry_config)]
) -> CaptureClassifier:
    """
    Crée et retourne un classificateur de capture.
    
    Args:
        geo_config: Configuration géométrique (injectée)
        
    Returns:
        CaptureClassifier: Instance du classificateur
    """
    return CaptureClassifier(
        white_level_threshold=geo_config.capture_classifier_white_level_threshold,
        white_percentage_threshold=geo_config.capture_classifier_white_percentage_threshold,
        enabled=geo_config.capture_classifier_enabled
    )


def get_preprocessing_normalizer(
    capture_classifier: Annotated[CaptureClassifier, Depends(get_capture_classifier)]
) -> PreprocessingNormalizer:
    """
    Crée et retourne un normaliseur de prétraitement.
    
    Args:
        capture_classifier: Classificateur de capture (injecté)
        
    Returns:
        PreprocessingNormalizer: Instance du normaliseur
    """
    return PreprocessingNormalizer(capture_classifier=capture_classifier)


@lru_cache()
def get_model_registry() -> ModelRegistry:
    """
    Récupère le registre de modèles (singleton).
    Cette fonction est mise en cache pour partager le registre entre toutes les requêtes.
    
    Returns:
        ModelRegistry: Registre de modèles
    """
    return ModelRegistry()


def get_geometry_normalizer(
    geo_config: Annotated[GeometryConfig, Depends(get_geometry_config)],
    qa_config: Annotated[QAConfig, Depends(get_qa_config)],
    model_registry: Annotated[ModelRegistry, Depends(get_model_registry)]
) -> GeometryNormalizer:
    """
    Crée et retourne un normaliseur géométrique.
    
    Args:
        geo_config: Configuration géométrique (injectée)
        qa_config: Configuration QA (injectée)
        model_registry: Registre de modèles (injecté)
        
    Returns:
        GeometryNormalizer: Instance du normaliseur
    """
    return GeometryNormalizer(
        geo_config=geo_config,
        qa_config=qa_config,
        model_registry=model_registry
    )

