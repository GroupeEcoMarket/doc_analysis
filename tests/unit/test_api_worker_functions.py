"""
Tests unitaires pour les fonctions worker du multiprocessing
"""

import pytest
import numpy as np
from unittest.mock import patch, Mock, MagicMock
import sys

# Importer les fonctions à tester
from src.api.routes import init_api_worker, process_page_worker, worker_dependencies


class TestInitApiWorker:
    """Tests pour la fonction init_api_worker"""
    
    def setup_method(self):
        """Réinitialise le dictionnaire worker_dependencies avant chaque test"""
        # Réinitialiser le dictionnaire global
        from src.api.routes import worker_dependencies
        worker_dependencies.clear()
    
    @patch('src.api.routes.get_document_classifier')
    @patch('src.api.routes.get_feature_extractor')
    @patch('src.api.routes.get_app_config')
    def test_init_api_worker_success(
        self,
        mock_get_app_config,
        mock_get_feature_extractor,
        mock_get_document_classifier
    ):
        """Test que init_api_worker initialise correctement les dépendances"""
        # Configuration des mocks
        mock_config = Mock()
        mock_get_app_config.return_value = mock_config
        
        mock_feature_extractor = Mock()
        mock_get_feature_extractor.return_value = mock_feature_extractor
        
        mock_classifier = Mock()
        mock_get_document_classifier.return_value = mock_classifier
        
        # Appel de la fonction
        init_api_worker()
        
        # Vérifications
        mock_get_app_config.assert_called_once()
        mock_get_feature_extractor.assert_called_once_with(mock_config)
        mock_get_document_classifier.assert_called_once_with(mock_config)
        
        # Vérifier que worker_dependencies est rempli
        assert 'feature_extractor' in worker_dependencies
        assert 'document_classifier' in worker_dependencies
        assert worker_dependencies['feature_extractor'] == mock_feature_extractor
        assert worker_dependencies['document_classifier'] == mock_classifier
    
    @patch('src.api.routes.get_document_classifier')
    @patch('src.api.routes.get_feature_extractor')
    @patch('src.api.routes.get_app_config')
    def test_init_api_worker_classifier_disabled(
        self,
        mock_get_app_config,
        mock_get_feature_extractor,
        mock_get_document_classifier
    ):
        """Test que init_api_worker gère le cas où la classification est désactivée"""
        # Configuration des mocks
        mock_config = Mock()
        mock_get_app_config.return_value = mock_config
        
        mock_feature_extractor = Mock()
        mock_get_feature_extractor.return_value = mock_feature_extractor
        
        # Simuler une ValueError quand la classification est désactivée
        mock_get_document_classifier.side_effect = ValueError("Classification désactivée")
        
        # Appel de la fonction
        init_api_worker()
        
        # Vérifications
        mock_get_app_config.assert_called_once()
        mock_get_feature_extractor.assert_called_once_with(mock_config)
        mock_get_document_classifier.assert_called_once_with(mock_config)
        
        # Vérifier que worker_dependencies est rempli avec classifier=None
        assert 'feature_extractor' in worker_dependencies
        assert 'document_classifier' in worker_dependencies
        assert worker_dependencies['feature_extractor'] == mock_feature_extractor
        assert worker_dependencies['document_classifier'] is None
    
    @patch('src.api.routes.get_document_classifier')
    @patch('src.api.routes.get_feature_extractor')
    @patch('src.api.routes.get_app_config')
    def test_init_api_worker_idempotent(
        self,
        mock_get_app_config,
        mock_get_feature_extractor,
        mock_get_document_classifier
    ):
        """Test que init_api_worker est idempotent (peut être appelé plusieurs fois)"""
        # Configuration des mocks
        mock_config = Mock()
        mock_get_app_config.return_value = mock_config
        
        mock_feature_extractor = Mock()
        mock_get_feature_extractor.return_value = mock_feature_extractor
        
        mock_classifier = Mock()
        mock_get_document_classifier.return_value = mock_classifier
        
        # Premier appel
        init_api_worker()
        
        # Réinitialiser les mocks pour compter les appels
        mock_get_app_config.reset_mock()
        mock_get_feature_extractor.reset_mock()
        mock_get_document_classifier.reset_mock()
        
        # Deuxième appel
        init_api_worker()
        
        # Vérifier que les fonctions ne sont pas appelées à nouveau
        mock_get_app_config.assert_not_called()
        mock_get_feature_extractor.assert_not_called()
        mock_get_document_classifier.assert_not_called()
        
        # Vérifier que worker_dependencies est toujours rempli
        assert 'feature_extractor' in worker_dependencies
        assert 'document_classifier' in worker_dependencies


class TestProcessPageWorker:
    """Tests pour la fonction process_page_worker"""
    
    def setup_method(self):
        """Prépare worker_dependencies avec des mocks avant chaque test"""
        # Réinitialiser le dictionnaire global
        from src.api.routes import worker_dependencies
        worker_dependencies.clear()
    
    def test_process_page_worker_success(self):
        """Test que process_page_worker traite correctement une page"""
        # Importer worker_dependencies pour le modifier
        from src.api.routes import worker_dependencies
        
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
        
        # Remplir worker_dependencies
        worker_dependencies['feature_extractor'] = mock_feature_extractor
        worker_dependencies['document_classifier'] = mock_classifier
        
        # Créer une image de test
        image_np = np.ones((100, 200, 3), dtype=np.uint8) * 255
        page_index = 0
        
        # Appel de la fonction
        result_index, result_data = process_page_worker((page_index, image_np))
        
        # Vérifications
        assert result_index == page_index
        assert result_data == mock_classification_result
        
        # Vérifier que les méthodes ont été appelées
        mock_feature_extractor.extract_ocr.assert_called_once_with(image_np)
        mock_classifier.predict.assert_called_once_with({'ocr_lines': mock_ocr_lines})
    
    def test_process_page_worker_missing_classifier(self):
        """Test que process_page_worker gère le cas où le classifier n'est pas disponible"""
        # Importer worker_dependencies pour le modifier
        from src.api.routes import worker_dependencies
        
        # Préparer seulement le feature_extractor
        mock_feature_extractor = Mock()
        worker_dependencies['feature_extractor'] = mock_feature_extractor
        worker_dependencies['document_classifier'] = None
        
        # Créer une image de test
        image_np = np.ones((100, 200, 3), dtype=np.uint8) * 255
        page_index = 1
        
        # Appel de la fonction
        result_index, result_data = process_page_worker((page_index, image_np))
        
        # Vérifications
        assert result_index == page_index
        assert result_data['document_type'] is None
        assert result_data['confidence'] == 0.0
        assert 'error' in result_data
        assert "Classifier non disponible" in result_data['error']
        
        # Vérifier que extract_ocr n'a pas été appelé
        mock_feature_extractor.extract_ocr.assert_not_called()
    
    def test_process_page_worker_missing_feature_extractor(self):
        """Test que process_page_worker gère le cas où le feature_extractor n'est pas disponible"""
        # Importer worker_dependencies pour le modifier
        from src.api.routes import worker_dependencies
        
        # Préparer seulement le classifier
        mock_classifier = Mock()
        worker_dependencies['feature_extractor'] = None
        worker_dependencies['document_classifier'] = mock_classifier
        
        # Créer une image de test
        image_np = np.ones((100, 200, 3), dtype=np.uint8) * 255
        page_index = 2
        
        # Appel de la fonction
        result_index, result_data = process_page_worker((page_index, image_np))
        
        # Vérifications
        assert result_index == page_index
        assert result_data['document_type'] is None
        assert result_data['confidence'] == 0.0
        assert 'error' in result_data
        
        # Vérifier que predict n'a pas été appelé
        mock_classifier.predict.assert_not_called()
    
    def test_process_page_worker_multiple_pages(self):
        """Test que process_page_worker peut traiter plusieurs pages avec différents index"""
        # Importer worker_dependencies pour le modifier
        from src.api.routes import worker_dependencies
        
        # Préparer les mocks
        mock_feature_extractor = Mock()
        mock_feature_extractor.extract_ocr.return_value = [{'text': 'test', 'confidence': 0.9, 'bounding_box': [0, 0, 1, 1]}]
        
        mock_classifier = Mock()
        mock_classifier.predict.return_value = {'document_type': 'Test', 'confidence': 0.85}
        
        worker_dependencies['feature_extractor'] = mock_feature_extractor
        worker_dependencies['document_classifier'] = mock_classifier
        
        # Créer plusieurs images de test
        image1 = np.ones((100, 100, 3), dtype=np.uint8) * 255
        image2 = np.ones((150, 150, 3), dtype=np.uint8) * 128
        
        # Traiter la première page
        result_index1, result_data1 = process_page_worker((0, image1))
        assert result_index1 == 0
        
        # Traiter la deuxième page
        result_index2, result_data2 = process_page_worker((1, image2))
        assert result_index2 == 1
        
        # Vérifier que extract_ocr a été appelé deux fois
        assert mock_feature_extractor.extract_ocr.call_count == 2
        assert mock_classifier.predict.call_count == 2

