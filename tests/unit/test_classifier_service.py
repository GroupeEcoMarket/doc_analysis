"""
Tests unitaires pour le module classifier_service
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import joblib
import tempfile
import os

from src.classification.classifier_service import DocumentClassifier
from src.pipeline.models import FeaturesOutput, OCRLine
from src.utils.config_loader import Config


@pytest.fixture
def mock_model():
    """Mock d'un modèle de classification"""
    model = Mock()
    model.predict.return_value = np.array([1])  # Classe 1
    model.predict_proba.return_value = np.array([[0.1, 0.9]])  # Probabilités
    return model


@pytest.fixture
def mock_feature_engineer():
    """Mock du FeatureEngineer"""
    engineer = Mock()
    engineer.extract_document_embedding.return_value = np.random.rand(388).astype(np.float32)
    engineer.semantic_model_name = 'test-model'
    return engineer


@pytest.fixture
def sample_model_file(mock_model, tmp_path):
    """Crée un fichier modèle temporaire pour les tests"""
    model_path = tmp_path / "test_model.joblib"
    model_data = {
        'model': mock_model,
        'class_names': ['Attestation_CEE', 'Facture', 'Contrat']
    }
    joblib.dump(model_data, model_path)
    return str(model_path)


@pytest.fixture
def mock_config():
    """Mock de la configuration"""
    config = Mock(spec=Config)
    config.get.return_value = {
        'enabled': True,
        'model_path': 'models/document_classifier.joblib',
        'embedding_model': 'sentence-transformers/test-model',
        'min_confidence': 0.70,
        'classification_confidence_threshold': 0.60
    }
    return config


@pytest.fixture
def sample_ocr_data():
    """Exemple de données OCR pour les tests"""
    return {
        'ocr_lines': [
            {
                'text': 'Attestation de conformité CEE',
                'confidence': 0.95,
                'bounding_box': [100, 200, 500, 250]
            },
            {
                'text': 'Document certifié',
                'confidence': 0.88,
                'bounding_box': [100, 300, 400, 350]
            }
        ]
    }


class TestDocumentClassifier:
    """Tests pour la classe DocumentClassifier"""
    
    @patch('src.classification.classifier_service.joblib.load')
    @patch('src.classification.classifier_service.FeatureEngineer')
    @patch('src.classification.classifier_service.get_config')
    def test_init_from_config(
        self,
        mock_get_config,
        mock_feature_engineer_class,
        mock_joblib_load,
        mock_config,
        mock_model,
        sample_model_file
    ):
        """Test de l'initialisation depuis la configuration"""
        mock_get_config.return_value = mock_config
        mock_joblib_load.return_value = {
            'model': mock_model,
            'class_names': ['Attestation_CEE', 'Facture']
        }
        mock_feature_engineer_class.return_value = Mock()
        
        # Mock du Path.exists pour que le modèle soit trouvé
        with patch('src.classification.classifier_service.Path.exists', return_value=True):
            classifier = DocumentClassifier(app_config=mock_config)
        
        assert classifier.model is not None
        assert classifier.class_names == ['Attestation_CEE', 'Facture']
        mock_joblib_load.assert_called_once()
    
    @patch('src.classification.classifier_service.joblib.load')
    @patch('src.classification.classifier_service.FeatureEngineer')
    def test_init_with_custom_params(
        self,
        mock_feature_engineer_class,
        mock_joblib_load,
        mock_model,
        sample_model_file
    ):
        """Test de l'initialisation avec paramètres personnalisés"""
        mock_joblib_load.return_value = {
            'model': mock_model,
            'class_names': ['Type1', 'Type2']
        }
        mock_feature_engineer_class.return_value = Mock()
        
        with patch('src.classification.classifier_service.Path.exists', return_value=True):
            classifier = DocumentClassifier(
                model_path=sample_model_file,
                semantic_model_name='custom-model',
                min_confidence=0.80
            )
        
        assert classifier.model is not None
        assert classifier.class_names == ['Type1', 'Type2']
    
    @patch('src.classification.classifier_service.joblib.load')
    @patch('src.classification.classifier_service.FeatureEngineer')
    def test_load_model_file_not_found(
        self,
        mock_feature_engineer_class,
        mock_joblib_load
    ):
        """Test que FileNotFoundError est levée si le modèle n'existe pas"""
        mock_feature_engineer_class.return_value = Mock()
        
        with patch('src.classification.classifier_service.Path.exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                DocumentClassifier(model_path='nonexistent.joblib')
    
    @patch('src.classification.classifier_service.joblib.load')
    @patch('src.classification.classifier_service.FeatureEngineer')
    def test_predict_with_proba(
        self,
        mock_feature_engineer_class,
        mock_joblib_load,
        mock_model,
        mock_feature_engineer,
        sample_ocr_data,
        sample_model_file
    ):
        """Test de la prédiction avec probabilités"""
        mock_joblib_load.return_value = {
            'model': mock_model,
            'class_names': ['Attestation_CEE', 'Facture']
        }
        mock_feature_engineer_class.return_value = mock_feature_engineer
        
        with patch('src.classification.classifier_service.Path.exists', return_value=True):
            classifier = DocumentClassifier(model_path=sample_model_file)
        
        result = classifier.predict(sample_ocr_data)
        
        # Vérifier le format de la réponse
        assert 'document_type' in result
        assert 'confidence' in result
        assert result['document_type'] == 'Facture'  # Classe 1
        assert result['confidence'] == pytest.approx(0.9)  # Probabilité max
        
        # Vérifier que les dépendances ont été appelées
        mock_feature_engineer.extract_document_embedding.assert_called_once()
        mock_model.predict.assert_called_once()
        mock_model.predict_proba.assert_called_once()
    
    @patch('src.classification.classifier_service.joblib.load')
    @patch('src.classification.classifier_service.FeatureEngineer')
    def test_predict_without_proba(
        self,
        mock_feature_engineer_class,
        mock_joblib_load,
        mock_model,
        mock_feature_engineer,
        sample_ocr_data,
        sample_model_file
    ):
        """Test de la prédiction sans probabilités"""
        # Modèle sans predict_proba
        model_no_proba = Mock()
        model_no_proba.predict.return_value = np.array([0])
        del model_no_proba.predict_proba
        
        mock_joblib_load.return_value = {
            'model': model_no_proba,
            'class_names': ['Attestation_CEE', 'Facture']
        }
        mock_feature_engineer_class.return_value = mock_feature_engineer
        
        with patch('src.classification.classifier_service.Path.exists', return_value=True):
            classifier = DocumentClassifier(model_path=sample_model_file)
        
        result = classifier.predict(sample_ocr_data)
        
        assert result['document_type'] == 'Attestation_CEE'  # Classe 0
        assert result['confidence'] == 1.0  # Par défaut si pas de probabilités
    
    @patch('src.classification.classifier_service.joblib.load')
    @patch('src.classification.classifier_service.FeatureEngineer')
    def test_predict_low_confidence(
        self,
        mock_feature_engineer_class,
        mock_joblib_load,
        mock_model,
        mock_feature_engineer,
        sample_ocr_data,
        sample_model_file
    ):
        """Test que document_type est None si confiance trop faible"""
        # Modèle avec probabilité faible
        model_low_conf = Mock()
        model_low_conf.predict.return_value = np.array([1])
        model_low_conf.predict_proba.return_value = np.array([[0.4, 0.5]])  # Confiance 0.5 < 0.6
        
        mock_joblib_load.return_value = {
            'model': model_low_conf,
            'class_names': ['Attestation_CEE', 'Facture']
        }
        mock_feature_engineer_class.return_value = mock_feature_engineer
        
        with patch('src.classification.classifier_service.Path.exists', return_value=True):
            classifier = DocumentClassifier(
                model_path=sample_model_file,
                app_config=Mock(get=lambda k, d=None: {
                    'classification': {
                        'enabled': True,
                        'classification_confidence_threshold': 0.60
                    }
                }.get(k, d))
            )
            # Override le threshold
            classifier.classification_confidence_threshold = 0.60
        
        result = classifier.predict(sample_ocr_data)
        
        assert result['document_type'] is None  # Confiance trop faible
        assert result['confidence'] == pytest.approx(0.5)
    
    @patch('src.classification.classifier_service.joblib.load')
    @patch('src.classification.classifier_service.FeatureEngineer')
    def test_predict_with_features_output(
        self,
        mock_feature_engineer_class,
        mock_joblib_load,
        mock_model,
        mock_feature_engineer,
        sample_model_file
    ):
        """Test de la prédiction avec FeaturesOutput"""
        mock_joblib_load.return_value = {
            'model': mock_model,
            'class_names': ['Attestation_CEE', 'Facture']
        }
        mock_feature_engineer_class.return_value = mock_feature_engineer
        
        features_output = FeaturesOutput(
            status='success',
            input_path='test.jpg',
            output_path='test.json',
            processing_time=1.0,
            ocr_lines=[
                OCRLine(
                    text='Test',
                    confidence=0.9,
                    bounding_box=[100, 200, 300, 250]
                )
            ]
        )
        
        with patch('src.classification.classifier_service.Path.exists', return_value=True):
            classifier = DocumentClassifier(model_path=sample_model_file)
        
        result = classifier.predict(features_output)
        
        assert 'document_type' in result
        assert 'confidence' in result
    
    @patch('src.classification.classifier_service.joblib.load')
    @patch('src.classification.classifier_service.FeatureEngineer')
    def test_normalize_ocr_input_dict(
        self,
        mock_feature_engineer_class,
        mock_joblib_load,
        mock_model,
        sample_model_file
    ):
        """Test de la normalisation d'entrée OCR (dict)"""
        mock_joblib_load.return_value = {
            'model': mock_model,
            'class_names': ['Type1']
        }
        mock_feature_engineer_class.return_value = Mock()
        
        with patch('src.classification.classifier_service.Path.exists', return_value=True):
            classifier = DocumentClassifier(model_path=sample_model_file)
        
        # Test avec dict direct
        ocr_dict = {'ocr_lines': []}
        normalized = classifier._normalize_ocr_input(ocr_dict)
        assert normalized == ocr_dict
        
        # Test avec dict contenant 'features'
        ocr_dict_features = {'features': {'ocr_lines': []}}
        normalized = classifier._normalize_ocr_input(ocr_dict_features)
        assert 'ocr_lines' in normalized or normalized == {'ocr_lines': []}

