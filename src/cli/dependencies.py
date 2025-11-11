"""
Dependency Injection for CLI application.

This module provides functions to create and inject dependencies
for CLI commands, following the same pattern as the API layer.
"""

from functools import lru_cache
from typing import Optional

from src.utils.config_loader import (
    get_config, Config, GeometryConfig, QAConfig, 
    PerformanceConfig, OutputConfig, PDFConfig
)
from src.pipeline.preprocessing import PreprocessingNormalizer
from src.pipeline.geometry import GeometryNormalizer
from src.pipeline.colometry import ColometryNormalizer
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


def get_performance_config(app_config: Config) -> PerformanceConfig:
    """
    Récupère la configuration de performance.
    
    Args:
        app_config: Configuration de l'application
        
    Returns:
        PerformanceConfig: Configuration de performance
    """
    return app_config.performance


def get_output_config(app_config: Config) -> OutputConfig:
    """
    Récupère la configuration de sortie.
    
    Args:
        app_config: Configuration de l'application
        
    Returns:
        OutputConfig: Configuration de sortie
    """
    return app_config.output


def get_pdf_config(app_config: Config) -> PDFConfig:
    """
    Récupère la configuration PDF.
    
    Args:
        app_config: Configuration de l'application
        
    Returns:
        PDFConfig: Configuration PDF
    """
    return app_config.pdf


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
    app_config = get_app_config(config_path)
    
    if capture_classifier is None:
        geo_config = get_geometry_config(app_config)
        capture_classifier = get_capture_classifier(geo_config)
    
    # Récupérer la configuration PDF
    pdf_config = get_pdf_config(app_config)
    
    return PreprocessingNormalizer(capture_classifier=capture_classifier, pdf_config=pdf_config)


def get_model_registry(config_path: Optional[str] = None) -> ModelRegistry:
    """
    Crée et retourne un registre de modèles.
    
    Args:
        config_path: Chemin vers le fichier de configuration (optionnel)
        
    Returns:
        ModelRegistry: Instance du registre de modèles
    """
    app_config = get_app_config(config_path)
    perf_config = get_performance_config(app_config)
    return ModelRegistry(lazy_load=perf_config.lazy_load_models)


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
    perf_config = get_performance_config(app_config)
    output_config = get_output_config(app_config)
    model_registry = get_model_registry(config_path)
    
    return GeometryNormalizer(
        geo_config=geo_config,
        qa_config=qa_config,
        perf_config=perf_config,
        output_config=output_config,
        model_registry=model_registry
    )


def get_colometry_normalizer(
    config_path: Optional[str] = None
) -> ColometryNormalizer:
    """
    Crée et retourne un normaliseur colométrique.
    
    Args:
        config_path: Chemin vers le fichier de configuration (optionnel)
        
    Returns:
        ColometryNormalizer: Instance du normaliseur
    """
    app_config = get_app_config(config_path)
    
    # Injecter la configuration via DI
    return ColometryNormalizer(app_config=app_config)


def get_feature_extractor(
    config_path: Optional[str] = None
) -> "FeatureExtractor":
    """
    Crée et retourne un extracteur de features.
    
    Args:
        config_path: Chemin vers le fichier de configuration (optionnel)
        
    Returns:
        FeatureExtractor: Instance de l'extracteur
    """
    from src.pipeline.features import FeatureExtractor
    
    app_config = get_app_config(config_path)
    
    # Injecter la configuration via DI
    return FeatureExtractor(app_config=app_config)


def get_document_classifier(
    config_path: Optional[str] = None
) -> "DocumentClassifier":
    """
    Crée et retourne un classifieur de documents.
    
    Args:
        config_path: Chemin vers le fichier de configuration (optionnel)
        
    Returns:
        DocumentClassifier: Instance du classifieur
    """
    from src.classification.classifier_service import DocumentClassifier
    
    app_config = get_app_config(config_path)
    
    # Vérifier si la classification est activée
    classification_config = app_config.get('classification', {})
    if not classification_config.get('enabled', False):
        raise ValueError(
            "La classification de documents n'est pas activée dans la configuration. "
            "Définissez classification.enabled: true dans config.yaml"
        )
    
    # Injecter la configuration via DI
    return DocumentClassifier(app_config=app_config)