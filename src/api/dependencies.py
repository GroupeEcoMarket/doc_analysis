"""
Dependency Injection for FastAPI application.

This module provides FastAPI dependencies for injecting configuration
and services into route handlers, following the Dependency Injection pattern.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from src.utils.config_loader import (
    get_config, Config, GeometryConfig, QAConfig, PDFConfig,
    PerformanceConfig, OutputConfig
)
from src.pipeline.preprocessing import PreprocessingNormalizer
from src.pipeline.geometry import GeometryNormalizer
from src.pipeline.colometry import ColometryNormalizer
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


def get_pdf_config(
    app_config: Annotated[Config, Depends(get_app_config)]
) -> PDFConfig:
    """
    Récupère la configuration PDF.
    
    Args:
        app_config: Configuration de l'application (injectée)
        
    Returns:
        PDFConfig: Configuration PDF
    """
    return app_config.pdf


def get_performance_config(
    app_config: Annotated[Config, Depends(get_app_config)]
) -> PerformanceConfig:
    """
    Récupère la configuration de performance.
    
    Args:
        app_config: Configuration de l'application (injectée)
        
    Returns:
        PerformanceConfig: Configuration de performance
    """
    return app_config.performance


def get_output_config(
    app_config: Annotated[Config, Depends(get_app_config)]
) -> OutputConfig:
    """
    Récupère la configuration de sortie.
    
    Args:
        app_config: Configuration de l'application (injectée)
        
    Returns:
        OutputConfig: Configuration de sortie
    """
    return app_config.output


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
    capture_classifier: Annotated[CaptureClassifier, Depends(get_capture_classifier)],
    pdf_config: Annotated[PDFConfig, Depends(get_pdf_config)]
) -> PreprocessingNormalizer:
    """
    Crée et retourne un normaliseur de prétraitement.
    
    Args:
        capture_classifier: Classificateur de capture (injecté)
        pdf_config: Configuration PDF (injectée)
        
    Returns:
        PreprocessingNormalizer: Instance du normaliseur
    """
    return PreprocessingNormalizer(capture_classifier=capture_classifier, pdf_config=pdf_config)


def get_model_registry(
    perf_config: Annotated[PerformanceConfig, Depends(get_performance_config)]
) -> ModelRegistry:
    """
    Récupère le registre de modèles.
    Le registre est créé avec la configuration de performance injectée.
    
    Args:
        perf_config: Configuration de performance (injectée)
        
    Returns:
        ModelRegistry: Registre de modèles
    """
    return ModelRegistry(lazy_load=perf_config.lazy_load_models)


def get_geometry_normalizer(
    geo_config: Annotated[GeometryConfig, Depends(get_geometry_config)],
    qa_config: Annotated[QAConfig, Depends(get_qa_config)],
    perf_config: Annotated[PerformanceConfig, Depends(get_performance_config)],
    output_config: Annotated[OutputConfig, Depends(get_output_config)],
    model_registry: Annotated[ModelRegistry, Depends(get_model_registry)]
) -> GeometryNormalizer:
    """
    Crée et retourne un normaliseur géométrique.
    
    Args:
        geo_config: Configuration géométrique (injectée)
        qa_config: Configuration QA (injectée)
        perf_config: Configuration de performance (injectée)
        output_config: Configuration de sortie (injectée)
        model_registry: Registre de modèles (injecté)
        
    Returns:
        GeometryNormalizer: Instance du normaliseur
    """
    return GeometryNormalizer(
        geo_config=geo_config,
        qa_config=qa_config,
        perf_config=perf_config,
        output_config=output_config,
        model_registry=model_registry
    )


def get_colometry_normalizer(
    app_config: Annotated[Config, Depends(get_app_config)]
) -> ColometryNormalizer:
    """
    Crée et retourne un normaliseur colométrique.
    
    Args:
        app_config: Configuration de l'application (injectée)
        
    Returns:
        ColometryNormalizer: Instance du normaliseur
    """
    return ColometryNormalizer(app_config=app_config)


@lru_cache()
def get_feature_extractor(
    app_config: Annotated[Config, Depends(get_app_config)]
) -> "FeatureExtractor":
    """
    Crée et retourne un extracteur de features (singleton).
    Le modèle OCR est chargé une seule fois au premier appel.
    
    Args:
        app_config: Configuration de l'application (injectée)
        
    Returns:
        FeatureExtractor: Instance de l'extracteur
    """
    from src.pipeline.features import FeatureExtractor
    
    # Injecter la configuration via DI
    return FeatureExtractor(app_config=app_config)


@lru_cache()
def get_document_classifier(
    app_config: Annotated[Config, Depends(get_app_config)]
) -> "DocumentClassifier":
    """
    Crée et retourne un classifieur de documents (singleton).
    Le modèle de classification est chargé une seule fois au premier appel.
    
    Args:
        app_config: Configuration de l'application (injectée)
        
    Returns:
        DocumentClassifier: Instance du classifieur
    """
    from src.classification.classifier_service import DocumentClassifier
    
    # Vérifier si la classification est activée
    classification_config = app_config.get('classification', {})
    if not classification_config.get('enabled', False):
        raise ValueError(
            "La classification de documents n'est pas activée dans la configuration. "
            "Définissez classification.enabled: true dans config.yaml"
        )
    
    # Injecter la configuration via DI
    return DocumentClassifier(app_config=app_config)