"""
Tests unitaires pour les fonctions worker du multiprocessing

Ce module teste les fonctions de traitement de pages dans src/workers.py :
- init_worker() : Initialisation des modèles dans les workers
- process_single_page() : Traitement d'une page unique
"""

import pytest
import numpy as np
from unittest.mock import patch, Mock, MagicMock
import sys

# Importer les fonctions à tester depuis src/workers
from src.workers import init_worker, process_single_page


# Fixture pour créer un mock de configuration réaliste
@pytest.fixture
def mock_app_config_for_worker():
    """
    Crée un mock de configuration réaliste pour les tests des workers.
    Ce mock retourne des dictionnaires concrets pour les clés attendues.
    """
    mock_config = MagicMock()
    
    # Configurer le mock pour qu'il retourne un dictionnaire réel pour 'features'
    features_config = {
        'ocr': {
            'enabled': True,
            'min_confidence': 0.70
        }
    }
    
    # Configuration du service OCR (microservice)
    ocr_service_config = {
        'queue_name': 'ocr-queue',
        'timeout_ms': 30000,
        'max_retries': 3
    }
    
    # Configurer aussi la section classification pour get_document_classifier
    classification_config = {
        'enabled': True,
        'device': 'cpu'  # Device PyTorch depuis config.yaml
    }
    
    # Simuler la méthode .get() pour retourner des valeurs concrètes
    def get_side_effect(key, default=None):
        if key == 'features':
            return features_config
        elif key == 'classification':
            return classification_config
        elif key == 'ocr_service':
            return ocr_service_config
        # Retourner un autre mock pour les clés non spécifiées
        return MagicMock()
    
    mock_config.get.side_effect = get_side_effect
    
    return mock_config


class TestInitWorker:
    """Tests pour la fonction init_worker"""
    
    def setup_method(self):
        """Réinitialise les variables globales avant chaque test"""
        # Réinitialiser les variables globales dans src.workers
        import src.workers
        src.workers._feature_extractor = None
        src.workers._document_classifier = None
    
    @patch('src.workers.get_config')
    @patch('src.workers.init_classification_worker')
    def test_init_worker_success(
        self,
        mock_init_classification_worker,
        mock_get_config,
        mock_app_config_for_worker
    ):
        """Test que init_worker initialise correctement les modèles."""
        # 1. Configurer les mocks
        mock_get_config.return_value = mock_app_config_for_worker
        
        mock_feature_extractor = Mock()
        mock_classifier = Mock()
        mock_init_classification_worker.return_value = (mock_feature_extractor, mock_classifier)
        
        # 2. Appeler la fonction à tester
        result_fe, result_dc = init_worker()
        
        # 3. Vérifications
        mock_get_config.assert_called_once()
        mock_init_classification_worker.assert_called_once_with(mock_app_config_for_worker)
        
        assert result_fe == mock_feature_extractor
        assert result_dc == mock_classifier
        
        # Vérifier que les variables globales sont bien définies
        import src.workers
        assert src.workers._feature_extractor == mock_feature_extractor
        assert src.workers._document_classifier == mock_classifier
    
    @patch('src.workers.get_config')
    @patch('src.workers.init_classification_worker')
    def test_init_worker_classifier_disabled(
        self,
        mock_init_classification_worker,
        mock_get_config,
        mock_app_config_for_worker
    ):
        """Test que init_worker gère le cas où la classification est désactivée."""
        mock_get_config.return_value = mock_app_config_for_worker
        
        mock_feature_extractor = Mock()
        mock_init_classification_worker.return_value = (mock_feature_extractor, None)
        
        result_fe, result_dc = init_worker()
        
        mock_get_config.assert_called_once()
        mock_init_classification_worker.assert_called_once_with(mock_app_config_for_worker)
        
        assert result_fe == mock_feature_extractor
        assert result_dc is None
        
        # Vérifier que les variables globales sont bien définies
        import src.workers
        assert src.workers._feature_extractor == mock_feature_extractor
        assert src.workers._document_classifier is None
    
    @patch('src.workers.get_config')
    @patch('src.workers.init_classification_worker')
    def test_init_worker_idempotent(
        self,
        mock_init_classification_worker,
        mock_get_config,
        mock_app_config_for_worker
    ):
        """Test que init_worker est idempotent (peut être appelé plusieurs fois)."""
        mock_get_config.return_value = mock_app_config_for_worker
        
        mock_feature_extractor = Mock()
        mock_classifier = Mock()
        mock_init_classification_worker.return_value = (mock_feature_extractor, mock_classifier)
        
        # Premier appel
        init_worker()
        
        # Réinitialiser les mocks pour compter les appels
        mock_get_config.reset_mock()
        mock_init_classification_worker.reset_mock()
        
        # Deuxième appel
        init_worker()
        
        # Vérifier que les fonctions ne sont pas appelées à nouveau
        # (car _feature_extractor existe déjà)
        mock_get_config.assert_not_called()
        mock_init_classification_worker.assert_not_called()
        
        # Vérifier que les variables globales sont toujours définies
        import src.workers
        assert src.workers._feature_extractor == mock_feature_extractor
        assert src.workers._document_classifier == mock_classifier


class TestProcessSinglePage:
    """Tests pour la fonction process_single_page"""
    
    def setup_method(self):
        """Prépare les variables globales avec des mocks avant chaque test"""
        # Réinitialiser les variables globales dans src.workers
        import src.workers
        src.workers._feature_extractor = None
        src.workers._document_classifier = None
    
    def test_process_single_page_success(self):
        """Test que process_single_page traite correctement une page"""
        import src.workers
        
        # Préparer les mocks
        mock_feature_extractor = Mock()
        mock_ocr_lines = [
            {
                'text': 'Test text',
                'confidence': 0.95,
                'bounding_box': [10, 20, 100, 50]
            }
        ]
        mock_feature_extractor.extract_ocr.return_value = mock_ocr_lines
        
        mock_classifier = Mock()
        mock_classification_result = {
            'document_type': 'Attestation_CEE',
            'confidence': 0.92
        }
        mock_classifier.predict.return_value = mock_classification_result
        
        # Remplir les variables globales
        src.workers._feature_extractor = mock_feature_extractor
        src.workers._document_classifier = mock_classifier
        
        # Créer une image de test
        image_np = np.ones((100, 200, 3), dtype=np.uint8) * 255
        page_index = 0
        
        # Appel de la fonction
        result_index, result_data = process_single_page((page_index, image_np))
        
        # Vérifications
        assert result_index == page_index
        assert result_data == mock_classification_result
        
        # Vérifier que les méthodes ont été appelées
        mock_feature_extractor.extract_ocr.assert_called_once_with(image_np)
        mock_classifier.predict.assert_called_once_with({'ocr_lines': mock_ocr_lines})
    
    def test_process_single_page_missing_classifier(self):
        """Test que process_single_page gère le cas où le classifier n'est pas disponible"""
        import src.workers
        
        # Préparer seulement le feature_extractor
        mock_feature_extractor = Mock()
        src.workers._feature_extractor = mock_feature_extractor
        src.workers._document_classifier = None
        
        # Créer une image de test
        image_np = np.ones((100, 200, 3), dtype=np.uint8) * 255
        page_index = 1
        
        # Appel de la fonction
        result_index, result_data = process_single_page((page_index, image_np))
        
        # Vérifications
        assert result_index == page_index
        assert result_data['document_type'] is None
        assert result_data['confidence'] == 0.0
        assert 'error' in result_data
        assert "Le service de classification n'est pas activé ou configuré." in result_data['error']
        
        # Vérifier que extract_ocr n'a pas été appelé
        mock_feature_extractor.extract_ocr.assert_not_called()
    
    def test_process_single_page_multiple_pages(self):
        """Test que process_single_page peut traiter plusieurs pages avec différents index"""
        import src.workers
        
        # Préparer les mocks
        mock_feature_extractor = Mock()
        mock_feature_extractor.extract_ocr.return_value = [{'text': 'test', 'confidence': 0.9, 'bounding_box': [0, 0, 1, 1]}]
        
        mock_classifier = Mock()
        mock_classifier.predict.return_value = {'document_type': 'Test', 'confidence': 0.85}
        
        src.workers._feature_extractor = mock_feature_extractor
        src.workers._document_classifier = mock_classifier
        
        # Créer plusieurs images de test
        image1 = np.ones((100, 100, 3), dtype=np.uint8) * 255
        image2 = np.ones((150, 150, 3), dtype=np.uint8) * 128
        
        # Traiter la première page
        result_index1, result_data1 = process_single_page((0, image1))
        assert result_index1 == 0
        
        # Traiter la deuxième page
        result_index2, result_data2 = process_single_page((1, image2))
        assert result_index2 == 1
        
        # Vérifier que extract_ocr a été appelé deux fois
        assert mock_feature_extractor.extract_ocr.call_count == 2
        assert mock_classifier.predict.call_count == 2
    
    def test_process_single_page_feature_extraction_error(self):
        """Test que process_single_page gère correctement FeatureExtractionError"""
        import src.workers
        from src.utils.exceptions import FeatureExtractionError
        
        # Préparer les mocks
        mock_feature_extractor = Mock()
        mock_feature_extractor.extract_ocr.side_effect = FeatureExtractionError("OCR failed")
        
        mock_classifier = Mock()
        src.workers._feature_extractor = mock_feature_extractor
        src.workers._document_classifier = mock_classifier
        
        # Créer une image de test
        image_np = np.ones((100, 200, 3), dtype=np.uint8) * 255
        page_index = 0
        
        # Appel de la fonction
        result_index, result_data = process_single_page((page_index, image_np))
        
        # Vérifications
        assert result_index == page_index
        assert result_data['document_type'] is None
        assert result_data['confidence'] == 0.0
        assert 'error' in result_data
        assert "OCR processing failed" in result_data['error']
        
        # Vérifier que predict n'a pas été appelé
        mock_classifier.predict.assert_not_called()
    
    def test_process_single_page_unexpected_error(self):
        """Test que process_single_page gère correctement les erreurs inattendues"""
        import src.workers
        
        # Préparer les mocks
        mock_feature_extractor = Mock()
        mock_feature_extractor.extract_ocr.side_effect = ValueError("Unexpected error")
        
        mock_classifier = Mock()
        src.workers._feature_extractor = mock_feature_extractor
        src.workers._document_classifier = mock_classifier
        
        # Créer une image de test
        image_np = np.ones((100, 200, 3), dtype=np.uint8) * 255
        page_index = 0
        
        # Appel de la fonction
        result_index, result_data = process_single_page((page_index, image_np))
        
        # Vérifications
        assert result_index == page_index
        assert result_data['document_type'] is None
        assert result_data['confidence'] == 0.0
        assert 'error' in result_data
        assert "An unexpected error occurred" in result_data['error']
        
        # Vérifier que predict n'a pas été appelé
        mock_classifier.predict.assert_not_called()
