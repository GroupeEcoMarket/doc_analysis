"""
Tests unitaires pour les scripts d'entraînement.

Ces tests vérifient que les fonctions des scripts d'entraînement
fonctionnent correctement de manière isolée.
"""

import pytest
from unittest.mock import patch, MagicMock

from training.train_classifier import init_worker


@pytest.fixture(autouse=True)
def reset_worker_feature_engineer():
    """
    Réinitialise la variable globale worker_feature_engineer avant chaque test.
    """
    import training.train_classifier as train_module
    train_module.worker_feature_engineer = None
    yield
    train_module.worker_feature_engineer = None


@patch('training.train_classifier.get_logger')
@patch('training.train_classifier.FeatureEngineer')
def test_init_worker_initializes_feature_engineer(mock_feature_engineer_class, mock_get_logger):
    """
    Vérifie que init_worker appelle le constructeur de FeatureEngineer
    avec les bons arguments.
    """
    # Arrange
    model_name = "test-model-name"
    min_conf = 0.75
    
    # Mock du logger
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    
    # Mock de l'instance FeatureEngineer
    mock_feature_engineer_instance = MagicMock()
    mock_feature_engineer_class.return_value = mock_feature_engineer_instance
    
    # Act
    init_worker(model_name, min_conf)
    
    # Assert
    # Vérifie que la classe FeatureEngineer a été instanciée une fois
    mock_feature_engineer_class.assert_called_once_with(
        semantic_model_name=model_name,
        min_confidence=min_conf
    )
    
    # Vérifie que le logger a été appelé pour les messages de debug
    assert mock_logger.debug.call_count == 2
    
    # Vérifie que la variable globale worker_feature_engineer a été définie
    import training.train_classifier as train_module
    assert train_module.worker_feature_engineer == mock_feature_engineer_instance


@patch('training.train_classifier.get_logger')
@patch('training.train_classifier.FeatureEngineer')
def test_init_worker_handles_different_parameters(mock_feature_engineer_class, mock_get_logger):
    """
    Vérifie que init_worker fonctionne avec différents paramètres.
    """
    # Arrange
    model_name = "another-model"
    min_conf = 0.90
    
    # Mock du logger
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    
    # Mock de l'instance FeatureEngineer
    mock_feature_engineer_instance = MagicMock()
    mock_feature_engineer_class.return_value = mock_feature_engineer_instance
    
    # Act
    init_worker(model_name, min_conf)
    
    # Assert
    mock_feature_engineer_class.assert_called_once_with(
        semantic_model_name=model_name,
        min_confidence=min_conf
    )

