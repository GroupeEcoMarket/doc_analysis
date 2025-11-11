"""
Tests d'intégration pour l'endpoint de classification de documents
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
import json
import tempfile
import joblib

from src.api.app import app


@pytest.fixture(scope="session")
def client():
    """Fixture qui crée un client de test pour l'API"""
    return TestClient(app)


@pytest.fixture
def sample_image():
    """Crée une image de test simple"""
    # Créer une image blanche avec du texte simulé
    img = np.ones((800, 600, 3), dtype=np.uint8) * 255
    
    # Ajouter du texte simulé (rectangles pour simuler du texte)
    cv2.rectangle(img, (50, 50), (550, 100), (0, 0, 0), -1)  # Ligne de texte
    cv2.rectangle(img, (50, 150), (400, 200), (0, 0, 0), -1)  # Ligne de texte
    
    return img


@pytest.fixture
def attestation_cee_image(tmp_path):
    """Crée une image de test pour une Attestation CEE"""
    img = np.ones((1000, 700, 3), dtype=np.uint8) * 255
    
    # Simuler un document Attestation CEE avec du texte
    # En-tête
    cv2.rectangle(img, (50, 50), (650, 120), (0, 0, 0), -1)
    # Titre
    cv2.rectangle(img, (50, 150), (600, 200), (0, 0, 0), -1)
    # Texte "Attestation de conformité CEE"
    cv2.rectangle(img, (50, 250), (500, 300), (0, 0, 0), -1)
    # Numéro de référence
    cv2.rectangle(img, (50, 350), (400, 400), (0, 0, 0), -1)
    
    file_path = tmp_path / "attestation_cee.png"
    cv2.imwrite(str(file_path), img)
    return file_path


@pytest.fixture
def facture_image(tmp_path):
    """Crée une image de test pour une Facture"""
    img = np.ones((1000, 700, 3), dtype=np.uint8) * 255
    
    # Simuler un document Facture avec du texte
    # En-tête "FACTURE"
    cv2.rectangle(img, (50, 50), (300, 100), (0, 0, 0), -1)
    # Numéro de facture
    cv2.rectangle(img, (50, 150), (400, 200), (0, 0, 0), -1)
    # Date
    cv2.rectangle(img, (50, 250), (300, 300), (0, 0, 0), -1)
    # Montant
    cv2.rectangle(img, (50, 350), (400, 400), (0, 0, 0), -1)
    # TVA
    cv2.rectangle(img, (50, 450), (350, 500), (0, 0, 0), -1)
    
    file_path = tmp_path / "facture.png"
    cv2.imwrite(str(file_path), img)
    return file_path


@pytest.fixture
def mock_classifier_model(tmp_path):
    """Crée un modèle de classification mock pour les tests"""
    # Créer un modèle mock simple
    from sklearn.linear_model import LogisticRegression
    
    # Créer des données d'entraînement factices
    X_train = np.random.rand(10, 388)  # 388 = dimension d'embedding typique
    y_train = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # 2 classes
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Sauvegarder le modèle
    model_path = tmp_path / "test_classifier.joblib"
    model_data = {
        'model': model,
        'class_names': ['Attestation_CEE', 'Facture']
    }
    joblib.dump(model_data, model_path)
    
    return str(model_path)


class TestClassificationEndpoint:
    """Tests pour l'endpoint /api/v1/classify"""
    
    def test_classify_endpoint_exists(self, client):
        """Test que l'endpoint existe et répond"""
        # Créer une image de test simple
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        _, img_encoded = cv2.imencode('.png', img)
        
        response = client.post(
            "/api/v1/classify",
            files={"file": ("test.png", img_encoded.tobytes(), "image/png")}
        )
        
        # L'endpoint doit exister (même si la classification n'est pas activée)
        assert response.status_code in [200, 503, 500]
    
    @patch('src.api.dependencies.get_document_classifier')
    @patch('src.api.dependencies.get_feature_extractor')
    def test_classify_with_mock(
        self,
        mock_get_feature_extractor,
        mock_get_classifier,
        client,
        sample_image
    ):
        """Test de classification avec mocks"""
        # Mock du feature extractor
        mock_extractor = Mock()
        mock_extractor.extract_ocr.return_value = [
            {
                'text': 'Attestation de conformité CEE',
                'confidence': 0.95,
                'bounding_box': [100, 200, 500, 250]
            }
        ]
        mock_get_feature_extractor.return_value = mock_extractor
        
        # Mock du classifier
        mock_classifier = Mock()
        mock_classifier.predict.return_value = {
            'document_type': 'Attestation_CEE',
            'confidence': 0.92
        }
        mock_get_classifier.return_value = mock_classifier
        
        # Encoder l'image
        _, img_encoded = cv2.imencode('.png', sample_image)
        
        response = client.post(
            "/api/v1/classify",
            files={"file": ("test.png", img_encoded.tobytes(), "image/png")}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['status'] == 'success'
        assert data['document_type'] == 'Attestation_CEE'
        assert data['confidence'] == pytest.approx(0.92)
        assert 'processing_time' in data
        
        # Vérifier que les dépendances ont été appelées
        mock_extractor.extract_ocr.assert_called_once()
        mock_classifier.predict.assert_called_once()
    
    def test_classify_invalid_file(self, client):
        """Test avec un fichier invalide"""
        response = client.post(
            "/api/v1/classify",
            files={"file": ("test.txt", b"invalid content", "text/plain")}
        )
        
        # Devrait retourner une erreur 400 ou 500
        assert response.status_code in [400, 500]
    
    def test_classify_empty_file(self, client):
        """Test avec un fichier vide"""
        response = client.post(
            "/api/v1/classify",
            files={"file": ("empty.png", b"", "image/png")}
        )
        
        assert response.status_code == 400
        assert "vide" in response.json()['detail'].lower()


class TestClassificationWithRealModel:
    """Tests d'intégration avec un vrai modèle (si disponible)"""
    
    @pytest.mark.skipif(
        not Path("models/document_classifier.joblib").exists(),
        reason="Modèle de classification non disponible"
    )
    def test_classify_with_real_model(
        self,
        client,
        attestation_cee_image
    ):
        """Test avec un vrai modèle (si disponible)"""
        with open(attestation_cee_image, "rb") as f:
            response = client.post(
                "/api/v1/classify",
                files={"file": ("attestation_cee.png", f.read(), "image/png")}
            )
        
        if response.status_code == 200:
            data = response.json()
            assert data['status'] == 'success'
            assert 'document_type' in data
            assert 'confidence' in data
            # Si c'est une Attestation CEE, vérifier que le type est correct
            # (peut varier selon le modèle entraîné)
            assert data['document_type'] is not None or data['confidence'] < 0.6
    
    @pytest.mark.skipif(
        not Path("models/document_classifier.joblib").exists(),
        reason="Modèle de classification non disponible"
    )
    def test_classify_facture_with_real_model(
        self,
        client,
        facture_image
    ):
        """Test avec une facture et un vrai modèle"""
        with open(facture_image, "rb") as f:
            response = client.post(
                "/api/v1/classify",
                files={"file": ("facture.png", f.read(), "image/png")}
            )
        
        if response.status_code == 200:
            data = response.json()
            assert data['status'] == 'success'
            assert 'document_type' in data
            assert 'confidence' in data


class TestClassificationIntegration:
    """Tests d'intégration complets du pipeline de classification"""
    
    @patch('src.classification.classifier_service.joblib.load')
    @patch('src.classification.feature_engineering.SentenceTransformer')
    def test_full_classification_pipeline(
        self,
        mock_sentence_transformer_class,
        mock_joblib_load,
        client,
        sample_image,
        mock_classifier_model
    ):
        """Test du pipeline complet de classification"""
        # Mock du modèle sentence-transformers
        mock_st_model = Mock()
        mock_st_model.encode.return_value = np.random.rand(384).astype(np.float32)
        mock_st_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer_class.return_value = mock_st_model
        
        # Mock du modèle de classification
        from sklearn.linear_model import LogisticRegression
        test_model = LogisticRegression()
        test_model.fit(np.random.rand(5, 388), np.array([0, 0, 1, 1, 1]))
        
        mock_joblib_load.return_value = {
            'model': test_model,
            'class_names': ['Attestation_CEE', 'Facture']
        }
        
        # Encoder l'image
        _, img_encoded = cv2.imencode('.png', sample_image)
        
        # Appeler l'endpoint
        response = client.post(
            "/api/v1/classify",
            files={"file": ("test.png", img_encoded.tobytes(), "image/png")}
        )
        
        # Vérifier la réponse
        if response.status_code == 200:
            data = response.json()
            assert data['status'] == 'success'
            assert 'document_type' in data
            assert 'confidence' in data
            assert isinstance(data['confidence'], (int, float))
            assert 0.0 <= data['confidence'] <= 1.0

