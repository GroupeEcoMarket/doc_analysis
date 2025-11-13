"""
Fixtures communes pour tous les tests
"""

import pytest
from unittest.mock import Mock
from pathlib import Path
from src.utils.config_loader import (
    Config, GeometryConfig, QAConfig, PerformanceConfig, OutputConfig, PDFConfig
)
from src.utils.capture_classifier import CaptureClassifier
from src.models.registry import ModelRegistry


# ==========================================
# Fixture Autouse : Nettoyage du Cache
# ==========================================

@pytest.fixture(autouse=True)
def clear_dependency_cache():
    """
    Fixture qui s'exécute automatiquement pour chaque test.
    
    Elle vide le cache des dépendances FastAPI mises en cache avec @lru_cache
    après l'exécution de chaque test pour garantir l'isolation entre les tests.
    
    Cette fixture est essentielle pour éviter que les mocks d'un test
    n'affectent les tests suivants à cause du cache LRU.
    """
    # yield exécute le test
    yield
    
    # Ce code est exécuté APRÈS chaque test
    # Vider tous les caches des dépendances FastAPI
    try:
        from src.api.dependencies import (
            get_app_config,
            get_feature_extractor,
            get_document_classifier
        )
        get_app_config.cache_clear()
        get_feature_extractor.cache_clear()
        get_document_classifier.cache_clear()
    except ImportError:
        # Si les dépendances ne sont pas disponibles (tests unitaires purs),
        # on ignore silencieusement
        pass


# ==========================================
# Fixture Principale : Configuration Réelle
# ==========================================

@pytest.fixture(scope="session")
def app_config():
    """
    Charge la VRAIE configuration depuis config.yaml.
    
    Idéal pour les tests d'intégration et pour s'assurer que l'application
    est cohérente avec sa configuration.
    
    Cette fixture est mise en cache au niveau de la session pour éviter
    de recharger le fichier à chaque test.
    """
    # Chercher config.yaml à la racine du projet
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config.yaml"
    
    if not config_path.exists():
        pytest.skip(f"Fichier de configuration introuvable: {config_path}")
    
    return Config(str(config_path))


# ==========================================
# Fixtures Mockées : Pour Tests Unitaires
# ==========================================


@pytest.fixture
def mock_geometry_config():
    """Fixture pour GeometryConfig mock"""
    return GeometryConfig(
        capture_classifier_enabled=True,
        capture_classifier_white_level_threshold=245,
        capture_classifier_white_percentage_threshold=0.70,
        capture_classifier_skip_crop_if_scan=True,
        orientation_min_confidence=0.70,
        crop_enabled=True,
        crop_min_area_ratio=0.85,
        crop_max_margin_ratio=0.08,
        deskew_enabled=True,
        deskew_min_confidence=0.20,
        deskew_max_angle=15.0,
        deskew_min_angle=0.5,
        deskew_hough_threshold=100,
        quality_min_contrast=50,
        quality_min_resolution_width=1200,
        quality_min_resolution_height=1600,
    )


@pytest.fixture
def mock_qa_config():
    """Fixture pour QAConfig mock"""
    return QAConfig(
        low_confidence_orientation=0.70,
        overcrop_risk=0.08,
        low_contrast=50,
        too_small_width=1200,
        too_small_height=1600,
    )


@pytest.fixture
def mock_performance_config():
    """Fixture pour PerformanceConfig mock"""
    return PerformanceConfig(
        batch_size=10,
        max_workers=4,
        parallelization_threshold=2,
        lazy_load_models=True,
    )


@pytest.fixture
def mock_output_config():
    """Fixture pour OutputConfig mock"""
    return OutputConfig(
        save_original=True,
        save_transformed=True,
        save_qa_flags=True,
        save_transforms=True,
        image_format='png',
        jpeg_quality=95,
    )


@pytest.fixture
def mock_pdf_config():
    """Fixture pour PDFConfig mock"""
    return PDFConfig(
        dpi=300,
        min_dpi=300,
    )


@pytest.fixture
def mock_app_config(mock_geometry_config, mock_qa_config, mock_performance_config, mock_output_config, mock_pdf_config):
    """
    Fixture pour un objet Config COMPLÈTEMENT MOCKÉ.
    
    Idéal pour les tests unitaires purs où l'on veut un contrôle total
    et une isolation du système de fichiers.
    
    Cette fixture utilise Mock(spec=Config) pour s'assurer que le mock
    respecte l'interface de la classe Config.
    """
    # Utilise Mock(spec=Config) pour s'assurer que le mock respecte l'interface
    config = Mock(spec=Config)
    
    # Attache directement les sous-configurations mockées
    config.geometry = mock_geometry_config
    config.qa = mock_qa_config
    config.performance = mock_performance_config
    config.output = mock_output_config
    config.pdf = mock_pdf_config
    
    # Simule le dictionnaire interne _config_data pour la méthode .get()
    # C'est ici qu'on centralise les valeurs pour les sections non-structurées
    config._config_data = {
        'colometry': {},
        'features': {
            'ocr_filtering': {
                'enabled': False,
                'min_confidence': 0.70,
            }
        },
        'ocr_service': {
            'queue_name': 'ocr-queue',
            'timeout_ms': 30000,
            'max_retries': 3
        },
        'classification': {
            'enabled': True,
            'model_path': 'models/document_classifier.joblib',
            'embedding_model': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            'min_confidence': 0.70,
            'classification_confidence_threshold': 0.60,
        }
    }
    
    # Fais en sorte que la méthode .get() du mock utilise ce dictionnaire
    # C'est plus simple et plus proche du fonctionnement réel
    def mock_get(key, default=None):
        keys = key.split('.')
        value = config._config_data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value if value is not None else default
    
    config.get.side_effect = mock_get
    
    return config


@pytest.fixture
def mock_capture_classifier():
    """Fixture pour CaptureClassifier mock"""
    return CaptureClassifier(
        white_level_threshold=245,
        white_percentage_threshold=0.70,
        enabled=True,
    )


@pytest.fixture
def mock_model_registry():
    """Fixture pour ModelRegistry mock"""
    return ModelRegistry(lazy_load=True)

