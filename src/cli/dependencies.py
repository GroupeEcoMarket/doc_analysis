"""
Dependency Injection for CLI application.

This module provides functions to create and inject dependencies
for CLI commands, following the same pattern as the API layer.
"""

from functools import lru_cache
from typing import Optional

from src.utils.config_loader import get_config, Config, GeometryConfig, QAConfig
from src.pipeline.preprocessing import PreprocessingNormalizer
from src.pipeline.geometry import GeometryNormalizer
from src.utils.capture_classifier import CaptureClassifier
from src.models.registry import ModelRegistry


# ==========================================
# Configuration Dependencies
# ==========================================

def get_app_config(config_path: Optional[str] = None) -> Config:
    """
    Récupère la configuration de l'application.
    
    Args:
        config_path: Chemin optionnel vers le fichier de configuration
        
    Returns:
        Config: Configuration de l'application
    """
    return get_config(config_path)


def get_geometry_config(app_config: Config) -> GeometryConfig:
    """
    Récupère la configuration géométrique.
    
    Args:
        app_config: Configuration de l'application
        
    Returns:
        GeometryConfig: Configuration géométrique
    """
    return app_config.geometry


def get_qa_config(app_config: Config) -> QAConfig:
    """
    Récupère la configuration QA.
    
    Args:
        app_config: Configuration de l'application
        
    Returns:
        QAConfig: Configuration QA
    """
    return app_config.qa


# ==========================================
# Service Dependencies
# ==========================================

def get_capture_classifier(geo_config: GeometryConfig) -> CaptureClassifier:
    """
    Crée et retourne un classificateur de capture.
    
    Args:
        geo_config: Configuration géométrique
        
    Returns:
        CaptureClassifier: Instance du classificateur
    """
    return CaptureClassifier(
        white_level_threshold=geo_config.capture_classifier_white_level_threshold,
        white_percentage_threshold=geo_config.capture_classifier_white_percentage_threshold,
        enabled=geo_config.capture_classifier_enabled
    )


def get_preprocessing_normalizer(
    capture_classifier: Optional[CaptureClassifier] = None,
    config_path: Optional[str] = None
) -> PreprocessingNormalizer:
    """
    Crée et retourne un normaliseur de prétraitement.
    
    Args:
        capture_classifier: Classificateur de capture (optionnel, sera créé si None)
        config_path: Chemin vers le fichier de configuration (optionnel)
        
    Returns:
        PreprocessingNormalizer: Instance du normaliseur
    """
    if capture_classifier is None:
        app_config = get_app_config(config_path)
        geo_config = get_geometry_config(app_config)
        capture_classifier = get_capture_classifier(geo_config)
    
    return PreprocessingNormalizer(capture_classifier=capture_classifier)


def get_model_registry() -> ModelRegistry:
    """
    Crée et retourne un registre de modèles.
    
    Returns:
        ModelRegistry: Instance du registre de modèles
    """
    return ModelRegistry()


def get_geometry_normalizer(
    config_path: Optional[str] = None
) -> GeometryNormalizer:
    """
    Crée et retourne un normaliseur géométrique.
    
    Args:
        config_path: Chemin vers le fichier de configuration (optionnel)
        
    Returns:
        GeometryNormalizer: Instance du normaliseur
    """
    app_config = get_app_config(config_path)
    geo_config = get_geometry_config(app_config)
    qa_config = get_qa_config(app_config)
    model_registry = get_model_registry()
    
    return GeometryNormalizer(
        geo_config=geo_config,
        qa_config=qa_config,
        model_registry=model_registry
    )

