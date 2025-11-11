"""
Tests unitaires pour les fonctions worker du multiprocessing
"""

import pytest
import numpy as np
from unittest.mock import patch, Mock, MagicMock
import sys

# Importer les fonctions à tester
from src.api.routes import init_api_worker, process_page_worker, worker_dependencies


# Fixture pour créer un mock de configuration réaliste
@pytest.fixture
def mock_app_config_for_worker():
    """
    Crée un mock de configuration réaliste pour les tests des workers.
    Ce mock retourne des dictionnaires concrets pour les clés attendues.
    """
    mock_config = MagicMock()
    
    # Configurer le mock pour qu'il retourne un dictionnaire réel pour 'features'
    # C'est la clé de la correction.
    features_config = {
        'ocr': {
            'enabled': True,
            'default_language': 'fr',  # <-- La valeur concrète attendue !
            'use_gpu': False,
            'runtime_options': {},
            'max_image_dimension': 3500,
            'min_confidence': 0.70
        }
    }
    
    # Configurer aussi la section classification pour get_document_classifier
    classification_config = {
        'enabled': True
    }
    
    # Simuler la méthode .get() pour retourner des valeurs concrètes
    def get_side_effect(key, default=None):
        if key == 'features':
            return features_config
        elif key == 'classification':
            return classification_config
        # Retourner un autre mock pour les clés non spécifiées
        return MagicMock()
    
    mock_config.get.side_effect = get_side_effect
    
    return mock_config


class TestInitApiWorker:
    """Tests pour la fonction init_api_worker"""
    
    def setup_method(self):
        """Réinitialise le dictionnaire worker_dependencies avant chaque test"""
        # Réinitialiser le dictionnaire global
        from src.api.routes import worker_dependencies
        worker_dependencies.clear()
    
    # L'ordre des patchs est inversé par rapport à l'ordre des arguments.
    # Le patch du bas est appliqué en premier.
    @patch('src.utils.worker_init.create_api_dependencies_from_config')  # Patch #2
    @patch('src.api.routes.get_app_config')  # Patch #1
    def test_init_api_worker_success(
        self,
        mock_get_app_config,      # Correspond à @patch #1
        mock_create_deps_func,    # Correspond à @patch #2
        mock_app_config_for_worker
    ):
        """Test que init_api_worker initialise correctement les dépendances."""
        # 1. Configurer les mocks
        mock_get_app_config.return_value = mock_app_config_for_worker
        
        mock_feature_extractor = Mock()
        mock_classifier = Mock()
        mock_create_deps_func.return_value = {
            'feature_extractor': mock_feature_extractor,
            'document_classifier': mock_classifier
        }
        
        # 2. Appeler la fonction à tester
        init_api_worker()
        
        from src.api.routes import worker_dependencies
        
        # 3. Vérifications
        mock_get_app_config.assert_called_once()
        mock_create_deps_func.assert_called_once_with(mock_app_config_for_worker)
        
        assert 'feature_extractor' in worker_dependencies
        assert 'document_classifier' in worker_dependencies
        assert worker_dependencies['feature_extractor'] == mock_feature_extractor
        assert worker_dependencies['document_classifier'] == mock_classifier
    
    @patch('src.utils.worker_init.create_api_dependencies_from_config')
    @patch('src.api.routes.get_app_config')
    def test_init_api_worker_classifier_disabled(
        self,
        mock_get_app_config,
        mock_create_deps_func,
        mock_app_config_for_worker
    ):
        """Test que init_api_worker gère le cas où la classification est désactivée."""
        mock_get_app_config.return_value = mock_app_config_for_worker
        
        mock_create_deps_func.return_value = {
            'feature_extractor': Mock(),
            'document_classifier': None
        }
        
        init_api_worker()
        
        from src.api.routes import worker_dependencies
        
        mock_get_app_config.assert_called_once()
        mock_create_deps_func.assert_called_once_with(mock_app_config_for_worker)
        
        assert 'feature_extractor' in worker_dependencies
        assert 'document_classifier' in worker_dependencies
        assert worker_dependencies['document_classifier'] is None
    
    @patch('src.utils.worker_init.create_api_dependencies_from_config')
    @patch('src.api.routes.get_app_config')
    def test_init_api_worker_idempotent(
        self,
        mock_get_app_config,
        mock_create_deps_func,
        mock_app_config_for_worker
    ):
        """Test que init_api_worker est idempotent (peut être appelé plusieurs fois)."""
        mock_get_app_config.return_value = mock_app_config_for_worker
        
        mock_feature_extractor = Mock()
        mock_classifier = Mock()
        mock_create_deps_func.return_value = {
            'feature_extractor': mock_feature_extractor,
            'document_classifier': mock_classifier
        }
        
        # Premier appel
        init_api_worker()
        
        # Réinitialiser les mocks pour compter les appels
        mock_get_app_config.reset_mock()
        mock_create_deps_func.reset_mock()
        
        # Deuxième appel
        init_api_worker()
        
        # Vérifier que les fonctions ne sont pas appelées à nouveau
        # (car worker_dependencies['feature_extractor'] existe déjà)
        mock_get_app_config.assert_not_called()
        mock_create_deps_func.assert_not_called()
        
        # Vérifier que worker_dependencies est toujours rempli
        from src.api.routes import worker_dependencies
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

