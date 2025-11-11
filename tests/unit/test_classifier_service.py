"""
Tests unitaires pour le module classifier_service
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

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
    def test_init_from_config(
        self,
        mock_feature_engineer_class,
        mock_joblib_load,
        mock_config,
        mock_model
    ):
        """Test de l'initialisation depuis la configuration"""
        mock_joblib_load.return_value = {
            'model': mock_model,
            'class_names': ['Attestation_CEE', 'Facture']
        }
        mock_feature_engineer_class.return_value = Mock()
        
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
        mock_config
    ):
        """Test de l'initialisation avec paramètres personnalisés"""
        mock_joblib_load.return_value = {
            'model': mock_model,
            'class_names': ['Type1', 'Type2']
        }
        mock_feature_engineer_class.return_value = Mock()
        
        # Le constructeur s'attend à un chemin, on lui en donne un,
        # mais il ne sera jamais utilisé car joblib.load est intercepté.
        classifier = DocumentClassifier(
            app_config=mock_config,
            model_path='un/chemin/qui/n_existe/pas',
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
        mock_joblib_load,
        mock_config
    ):
        """Test que FileNotFoundError est levée si le modèle n'existe pas"""
        mock_feature_engineer_class.return_value = Mock()
        
        # Pattern EAFP : joblib.load lève directement FileNotFoundError si le fichier n'existe pas
        mock_joblib_load.side_effect = FileNotFoundError("No such file or directory")
        
        with pytest.raises(FileNotFoundError):
            DocumentClassifier(
                app_config=mock_config,
                model_path='nonexistent.joblib'
            )
    
    @patch('src.classification.classifier_service.joblib.load')
    @patch('src.classification.classifier_service.FeatureEngineer')
    def test_predict_with_proba(
        self,
        mock_feature_engineer_class,
        mock_joblib_load,
        mock_model,
        mock_feature_engineer,
        mock_config,
        sample_ocr_data
    ):
        """Test de la prédiction avec probabilités"""
        mock_joblib_load.return_value = {
            'model': mock_model,
            'class_names': ['Attestation_CEE', 'Facture']
        }
        mock_feature_engineer_class.return_value = mock_feature_engineer
        
        # Le constructeur s'attend à un chemin, on lui en donne un,
        # mais il ne sera jamais utilisé car joblib.load est intercepté.
        classifier = DocumentClassifier(
            app_config=mock_config,
            model_path='un/chemin/qui/n_existe/pas'
        )
        
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
        mock_config,
        sample_ocr_data
    ):
        """Test de la prédiction sans predict_proba"""
        # Le code détecte un modèle comme LightGBM si: hasattr(predict) and not hasattr(predict_proba)
        # Si détecté comme LightGBM, predict() est traité comme retournant des probabilités
        # Pour ce test, on accepte que le modèle sera traité comme LightGBM
        # et on s'assure que predict() retourne des probabilités valides (> threshold)
        model_no_proba = Mock(spec=['predict'])  # Pas de predict_proba
        # Retourner des probabilités (comme LightGBM) avec confidence > threshold
        model_no_proba.predict.return_value = np.array([[0.8, 0.2]])  # Probabilités pour 2 classes
        
        mock_joblib_load.return_value = {
            'model': model_no_proba,
            'class_names': ['Attestation_CEE', 'Facture']
        }
        mock_feature_engineer_class.return_value = mock_feature_engineer
        
        # Le constructeur s'attend à un chemin, on lui en donne un,
        # mais il ne sera jamais utilisé car joblib.load est intercepté.
        classifier = DocumentClassifier(
            app_config=mock_config,
            model_path='un/chemin/qui/n_existe/pas'
        )
        
        result = classifier.predict(sample_ocr_data)
        
        # Le modèle sera détecté comme LightGBM, donc predict() retourne des probabilités
        # [0.8, 0.2] -> argmax = 0 (Attestation_CEE), confidence = 0.8
        assert result['document_type'] == 'Attestation_CEE'  # Classe 0 (argmax de [0.8, 0.2])
        assert result['confidence'] == pytest.approx(0.8)  # max([0.8, 0.2])
    
    @patch('src.classification.classifier_service.joblib.load')
    @patch('src.classification.classifier_service.FeatureEngineer')
    def test_predict_low_confidence(
        self,
        mock_feature_engineer_class,
        mock_joblib_load,
        mock_model,
        mock_feature_engineer,
        sample_ocr_data
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
        
        # Créer un mock_config avec le bon threshold
        custom_config = Mock(spec=Config)
        custom_config.get.return_value = {
            'enabled': True,
            'model_path': 'un/chemin/qui/n_existe/pas',
            'embedding_model': 'test-model',
            'min_confidence': 0.70,
            'classification_confidence_threshold': 0.60
        }
        
        # Le constructeur s'attend à un chemin, on lui en donne un,
        # mais il ne sera jamais utilisé car joblib.load est intercepté.
        classifier = DocumentClassifier(
            app_config=custom_config,
            model_path='un/chemin/qui/n_existe/pas'
        )
        
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
        mock_config
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
        
        # joblib.load est mocké, donc pas besoin de patcher Path.exists
        # Le constructeur s'attend à un chemin, on lui en donne un,
        # mais il ne sera jamais utilisé car joblib.load est intercepté.
        classifier = DocumentClassifier(
            app_config=mock_config,
            model_path='un/chemin/qui/n_existe/pas'
        )
        
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
        mock_config
    ):
        """Test de la normalisation d'entrée OCR (dict)"""
        mock_joblib_load.return_value = {
            'model': mock_model,
            'class_names': ['Type1']
        }
        mock_feature_engineer_class.return_value = Mock()
        
        # Le constructeur s'attend à un chemin, on lui en donne un,
        # mais il ne sera jamais utilisé car joblib.load est intercepté.
        classifier = DocumentClassifier(
            app_config=mock_config,
            model_path='un/chemin/qui/n_existe/pas'
        )
        
        # Test avec dict direct
        ocr_dict = {'ocr_lines': []}
        normalized = classifier._normalize_ocr_input(ocr_dict)
        assert normalized == ocr_dict
        
        # Test avec dict contenant 'features'
        ocr_dict_features = {'features': {'ocr_lines': []}}
        normalized = classifier._normalize_ocr_input(ocr_dict_features)
        assert 'ocr_lines' in normalized or normalized == {'ocr_lines': []}

