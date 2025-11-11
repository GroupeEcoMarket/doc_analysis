"""
Tests unitaires pour le module feature_engineering
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.classification.feature_engineering import (
    FeatureEngineer,
    filter_ocr_lines,
    create_multimodal_embedding,
    extract_positional_features,
    aggregate_line_embeddings,
    extract_document_embedding
)
from src.pipeline.models import OCRLine, FeaturesOutput


@pytest.fixture
def mock_sentence_transformer():
    """Mock du modèle sentence-transformers"""
    mock_model = Mock()
    mock_model.encode.return_value = np.random.rand(384).astype(np.float32)  # Dimension typique
    mock_model.get_sentence_embedding_dimension.return_value = 384
    return mock_model


@pytest.fixture
def sample_ocr_lines():
    """Exemple de lignes OCR pour les tests"""
    return [
        OCRLine(
            text="Attestation de conformité",
            confidence=0.95,
            bounding_box=[100, 200, 500, 250]
        ),
        OCRLine(
            text="Document CEE",
            confidence=0.88,
            bounding_box=[100, 300, 400, 350]
        ),
        OCRLine(
            text="Texte avec faible confiance",
            confidence=0.50,  # En dessous du seuil par défaut (0.70)
            bounding_box=[100, 400, 500, 450]
        ),
        OCRLine(
            text="Numéro de référence: 12345",
            confidence=0.92,
            bounding_box=[100, 500, 600, 550]
        )
    ]


@pytest.fixture
def sample_ocr_dict():
    """Exemple de dictionnaire OCR pour les tests"""
    return {
        'ocr_lines': [
            {
                'text': 'Attestation de conformité',
                'confidence': 0.95,
                'bounding_box': [100, 200, 500, 250]
            },
            {
                'text': 'Document CEE',
                'confidence': 0.88,
                'bounding_box': [100, 300, 400, 350]
            }
        ]
    }


class TestFilterOCRLines:
    """Tests pour la fonction filter_ocr_lines"""
    
    def test_filter_by_confidence(self, sample_ocr_lines):
        """Test que les lignes sont filtrées selon la confiance"""
        filtered = filter_ocr_lines(sample_ocr_lines, min_confidence=0.70)
        
        # Seules les lignes avec confiance >= 0.70 doivent être conservées
        assert len(filtered) == 3
        assert all(line.confidence >= 0.70 for line in filtered)
    
    def test_filter_all_kept(self, sample_ocr_lines):
        """Test avec un seuil bas qui garde toutes les lignes"""
        filtered = filter_ocr_lines(sample_ocr_lines, min_confidence=0.40)
        assert len(filtered) == 4
    
    def test_filter_all_removed(self, sample_ocr_lines):
        """Test avec un seuil élevé qui supprime toutes les lignes"""
        filtered = filter_ocr_lines(sample_ocr_lines, min_confidence=0.99)
        assert len(filtered) == 0


class TestExtractPositionalFeatures:
    """Tests pour extract_positional_features"""
    
    def test_extract_from_simple_bbox(self):
        """Test avec un bounding box simple [x1, y1, x2, y2]"""
        bbox = [100, 200, 500, 300]
        features = extract_positional_features(bbox, image_width=1000, image_height=2000)
        
        assert features.shape == (4,)
        assert features.dtype == np.float32
        # Vérifier la normalisation
        assert features[0] == pytest.approx(0.1)  # 100/1000
        assert features[1] == pytest.approx(0.1)  # 200/2000
        assert features[2] == pytest.approx(0.5)  # 500/1000
        assert features[3] == pytest.approx(0.15)  # 300/2000
    
    def test_extract_from_point_list(self):
        """Test avec un bounding box sous forme de points"""
        bbox = [[100, 200], [500, 200], [500, 300], [100, 300]]
        features = extract_positional_features(bbox, image_width=1000, image_height=2000)
        
        assert features.shape == (4,)
        # Vérifier que les coordonnées sont normalisées
        assert 0.0 <= features[0] <= 1.0
        assert 0.0 <= features[1] <= 1.0
        assert 0.0 <= features[2] <= 1.0
        assert 0.0 <= features[3] <= 1.0


class TestCreateMultimodalEmbedding:
    """Tests pour create_multimodal_embedding"""
    
    def test_create_embedding(self, mock_sentence_transformer):
        """Test de création d'embedding multi-modal"""
        ocr_line = OCRLine(
            text="Test text",
            confidence=0.90,
            bounding_box=[100, 200, 500, 300]
        )
        
        embedding = create_multimodal_embedding(
            ocr_line,
            mock_sentence_transformer,
            image_width=1000,
            image_height=2000
        )
        
        # Vérifier la dimension (sémantique + positionnel)
        semantic_dim = mock_sentence_transformer.get_sentence_embedding_dimension()
        expected_dim = semantic_dim + 4  # +4 pour les coordonnées
        assert embedding.shape == (expected_dim,)
        assert embedding.dtype == np.float32
        
        # Vérifier que le modèle a été appelé
        mock_sentence_transformer.encode.assert_called_once_with(
            "Test text",
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )


class TestAggregateLineEmbeddings:
    """Tests pour aggregate_line_embeddings"""
    
    def test_weighted_mean(self):
        """Test de l'agrégation par moyenne pondérée"""
        embeddings = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
            np.array([7.0, 8.0, 9.0])
        ]
        confidences = [0.9, 0.8, 0.7]
        
        aggregated = aggregate_line_embeddings(
            embeddings,
            confidences,
            aggregation_method='weighted_mean'
        )
        
        assert aggregated.shape == (3,)
        # Vérifier que les valeurs sont raisonnables
        assert np.all(aggregated >= 1.0)
        assert np.all(aggregated <= 9.0)
    
    def test_mean(self):
        """Test de l'agrégation par moyenne simple"""
        embeddings = [
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0])
        ]
        
        aggregated = aggregate_line_embeddings(
            embeddings,
            aggregation_method='mean'
        )
        
        assert aggregated.shape == (2,)
        assert aggregated[0] == pytest.approx(2.0)  # (1+3)/2
        assert aggregated[1] == pytest.approx(3.0)  # (2+4)/2
    
    def test_max(self):
        """Test de l'agrégation par maximum"""
        embeddings = [
            np.array([1.0, 5.0]),
            np.array([3.0, 2.0])
        ]
        
        aggregated = aggregate_line_embeddings(
            embeddings,
            aggregation_method='max'
        )
        
        assert aggregated.shape == (2,)
        assert aggregated[0] == pytest.approx(3.0)  # max(1, 3)
        assert aggregated[1] == pytest.approx(5.0)  # max(5, 2)


class TestFeatureEngineer:
    """Tests pour la classe FeatureEngineer"""
    
    @patch('src.classification.feature_engineering.SentenceTransformer')
    def test_init(self, mock_sentence_transformer_class, mock_sentence_transformer):
        """Test de l'initialisation"""
        mock_sentence_transformer_class.return_value = mock_sentence_transformer
        
        engineer = FeatureEngineer(
            semantic_model_name='test-model',
            min_confidence=0.75
        )
        
        assert engineer.semantic_model_name == 'test-model'
        assert engineer.min_confidence == 0.75
        mock_sentence_transformer_class.assert_called_once_with('test-model')
    
    @patch('src.classification.feature_engineering.SentenceTransformer')
    def test_extract_document_embedding_from_features_output(
        self,
        mock_sentence_transformer_class,
        mock_sentence_transformer,
        sample_ocr_lines
    ):
        """Test extraction depuis FeaturesOutput"""
        mock_sentence_transformer_class.return_value = mock_sentence_transformer
        
        engineer = FeatureEngineer(
            semantic_model_name='test-model',
            min_confidence=0.70,
            image_width=1000,
            image_height=2000
        )
        
        features_output = FeaturesOutput(
            status='success',
            input_path='test.jpg',
            output_path='test.json',
            processing_time=1.0,
            ocr_lines=sample_ocr_lines
        )
        
        embedding = engineer.extract_document_embedding(features_output)
        
        # Vérifier la dimension
        semantic_dim = mock_sentence_transformer.get_sentence_embedding_dimension()
        expected_dim = semantic_dim + 4
        assert embedding.shape == (expected_dim,)
        
        # Vérifier que le modèle a été appelé (pour les lignes filtrées)
        assert mock_sentence_transformer.encode.call_count == 3  # 3 lignes avec conf >= 0.70
    
    @patch('src.classification.feature_engineering.SentenceTransformer')
    def test_extract_document_embedding_from_dict(
        self,
        mock_sentence_transformer_class,
        mock_sentence_transformer,
        sample_ocr_dict
    ):
        """Test extraction depuis un dictionnaire"""
        mock_sentence_transformer_class.return_value = mock_sentence_transformer
        
        engineer = FeatureEngineer(
            semantic_model_name='test-model',
            min_confidence=0.70,
            image_width=1000,
            image_height=2000
        )
        
        embedding = engineer.extract_document_embedding(sample_ocr_dict)
        
        # Vérifier la dimension
        semantic_dim = mock_sentence_transformer.get_sentence_embedding_dimension()
        expected_dim = semantic_dim + 4
        assert embedding.shape == (expected_dim,)
    
    @patch('src.classification.feature_engineering.SentenceTransformer')
    def test_extract_document_embedding_empty(
        self,
        mock_sentence_transformer_class,
        mock_sentence_transformer
    ):
        """Test avec des données vides"""
        mock_sentence_transformer_class.return_value = mock_sentence_transformer
        
        engineer = FeatureEngineer(
            semantic_model_name='test-model',
            min_confidence=0.99  # Seuil très élevé
        )
        
        features_output = FeaturesOutput(
            status='success',
            input_path='test.jpg',
            output_path='test.json',
            processing_time=1.0,
            ocr_lines=[]  # Aucune ligne
        )
        
        embedding = engineer.extract_document_embedding(features_output)
        
        # Devrait retourner un vecteur zéro
        semantic_dim = mock_sentence_transformer.get_sentence_embedding_dimension()
        expected_dim = semantic_dim + 4
        assert embedding.shape == (expected_dim,)
        assert np.allclose(embedding, 0.0)


class TestExtractDocumentEmbeddingFunction:
    """Tests pour la fonction de convenance extract_document_embedding"""
    
    @patch('src.classification.feature_engineering.SentenceTransformer')
    def test_function_wrapper(
        self,
        mock_sentence_transformer_class,
        mock_sentence_transformer,
        sample_ocr_dict
    ):
        """Test de la fonction de convenance"""
        mock_sentence_transformer_class.return_value = mock_sentence_transformer
        
        embedding = extract_document_embedding(
            sample_ocr_dict,
            semantic_model_name='test-model',
            min_confidence=0.70,
            image_width=1000,
            image_height=2000
        )
        
        # Vérifier que ça fonctionne
        semantic_dim = mock_sentence_transformer.get_sentence_embedding_dimension()
        expected_dim = semantic_dim + 4
        assert embedding.shape == (expected_dim,)

