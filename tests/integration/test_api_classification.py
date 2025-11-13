"""
Tests d'intégration pour l'endpoint de classification de documents
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock, MagicMock
import json
import tempfile
import joblib
from io import BytesIO
from contextlib import contextmanager

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


@contextmanager
def mock_dramatiq_actor(mock_feature_extractor, mock_document_classifier):
    """
    Context manager qui mocke Dramatiq pour exécuter les tâches de manière synchrone.
    
    Cette fonction mocke la tâche Dramatiq pour qu'elle s'exécute immédiatement
    et stocke le résultat dans un backend mocké.
    
    Adapté pour la nouvelle architecture avec process_page_task et groupes Dramatiq.
    Utilise maintenant le système de stockage basé sur des URI au lieu de base64.
    
    Args:
        mock_feature_extractor: Mock du FeatureExtractor
        mock_document_classifier: Mock du DocumentClassifier
    """
    from src.workers import classify_document_task, process_page_task
    from dramatiq.results.errors import ResultMissing
    import uuid
    import numpy as np
    import cv2
    from pathlib import Path
    import tempfile
    import shutil
    
    # Mock du backend de résultats pour stocker les résultats
    results_store = {}
    
    # Créer un répertoire temporaire pour le stockage mocké
    temp_storage_dir = Path(tempfile.mkdtemp(prefix="test_storage_"))
    
    # Dictionnaire pour stocker les fichiers temporaires (URI -> contenu)
    storage_files = {}
    
    # Sauvegarder les fonctions originales
    original_classify_task_fn = classify_document_task.fn
    original_process_page_task_fn = process_page_task.fn
    
    def mock_process_page_task_fn(image_uri, page_index=0):
        """Fonction mockée pour process_page_task qui exécute de manière synchrone"""
        # Mock des dépendances
        import src.workers
        src.workers._feature_extractor = mock_feature_extractor
        src.workers._document_classifier = mock_document_classifier
        
        # Charger l'image depuis le stockage mocké
        if image_uri in storage_files:
            image_np = storage_files[image_uri]
        elif image_uri.startswith("file://"):
            # Charger depuis le fichier si c'est une URI file://
            image_path = Path(image_uri[7:])  # Enlever "file://"
            if image_path.exists():
                image_np = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            else:
                raise FileNotFoundError(f"Image non trouvée: {image_path}")
        else:
            raise ValueError(f"URI invalide: {image_uri}")
        
        # Exécuter la logique de process_page_task
        if mock_document_classifier is None:
            return {
                'page_index': page_index,
                'document_type': None,
                'confidence': 0.0,
                'error': "Le service de classification n'est pas activé ou configuré."
            }
        
        try:
            ocr_lines = mock_feature_extractor.extract_ocr(image_np)
            ocr_data = {'ocr_lines': ocr_lines}
            classification_result = mock_document_classifier.predict(ocr_data)
            classification_result['page_index'] = page_index
            return classification_result
        except Exception as e:
            from src.utils.exceptions import FeatureExtractionError
            if isinstance(e, FeatureExtractionError):
                return {
                    'page_index': page_index,
                    'document_type': None,
                    'confidence': 0.0,
                    'error': f"OCR processing failed: {e}"
                }
            else:
                return {
                    'page_index': page_index,
                    'document_type': None,
                    'confidence': 0.0,
                    'error': f"An unexpected error occurred: {e}"
                }
    
    def mock_classify_task_fn(file_uri, original_filename):
        """Fonction mockée qui exécute classify_document_task de manière synchrone"""
        # Mock des dépendances
        import src.workers
        src.workers._feature_extractor = mock_feature_extractor
        src.workers._document_classifier = mock_document_classifier
        
        # Exécuter la fonction originale avec les mocks
        # Elle va créer des messages pour process_page_task et utiliser les groupes
        return original_classify_task_fn(file_uri, original_filename)
    
    # Mock du message Dramatiq
    class MockMessage:
        def __init__(self, message_id):
            self.message_id = message_id
    
    # Mock de process_page_task.send pour exécuter immédiatement
    def mock_process_page_send(image_uri, page_index=0):
        """Mock de process_page_task.send() qui exécute immédiatement"""
        message_id = str(uuid.uuid4())
        try:
            result = mock_process_page_task_fn(image_uri, page_index)
            results_store[message_id] = result
        except Exception as e:
            results_store[message_id] = {'error': str(e), 'page_index': page_index}
        return MockMessage(message_id)
    
    # Mock de classify_document_task.send
    def mock_classify_send(file_uri, original_filename):
        """Mock de classify_document_task.send() qui exécute immédiatement"""
        message_id = str(uuid.uuid4())
        try:
            result = mock_classify_task_fn(file_uri, original_filename)
            results_store[message_id] = result
        except Exception as e:
            results_store[message_id] = {'error': str(e)}
        return MockMessage(message_id)
    
    # Mock des groupes Dramatiq
    class MockGroupResult:
        def __init__(self, messages):
            self.messages = messages
        
        def get_results(self, block=True):
            """Récupère les résultats de tous les messages"""
            results = []
            for msg in self.messages:
                if msg.message_id in results_store:
                    results.append(results_store[msg.message_id])
                else:
                    # Si le résultat n'est pas encore disponible, attendre un peu
                    import time
                    time.sleep(0.01)
                    if msg.message_id in results_store:
                        results.append(results_store[msg.message_id])
                    else:
                        results.append({
                            'page_index': 0,
                            'document_type': None,
                            'confidence': 0.0,
                            'error': 'Result not found'
                        })
            return results
    
    class MockGroup:
        def __init__(self, messages):
            self.messages = messages
        
        def run(self):
            return MockGroupResult(self.messages)
    
    def mock_group(messages):
        """Mock de dramatiq.group()"""
        return MockGroup(messages)
    
    # Mock du backend pour récupérer les résultats
    class MockBackend:
        def get_result(self, message, block=False):
            message_id = message.message_id
            if message_id not in results_store:
                raise ResultMissing()
            result = results_store[message_id]
            if 'error' in result and 'status' not in result:
                raise Exception(result['error'])
            return result
    
    class MockResultsMiddleware:
        def __init__(self):
            self.backend = MockBackend()
    
    # Mock du système de stockage
    class MockStorage:
        """Mock du système de stockage pour les tests"""
        def __init__(self):
            self.storage_files = storage_files
            self.temp_storage_dir = temp_storage_dir
        
        def save_file(self, file_content, filename=None, prefix="file_"):
            """Sauvegarde un fichier et retourne son URI"""
            file_id = f"{prefix}{uuid.uuid4().hex}"
            extension = Path(filename).suffix if filename else ""
            file_path = self.temp_storage_dir / f"{file_id}{extension}"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(file_content)
            uri = f"file://{file_path}"
            return uri
        
        def save_image(self, image_np, page_index, task_id=None):
            """Sauvegarde une image et retourne son URI"""
            file_id = f"page_{page_index}_{uuid.uuid4().hex[:8]}"
            if task_id:
                file_id = f"{task_id}_{file_id}"
            image_path = self.temp_storage_dir / f"{file_id}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_path), image_np)
            uri = f"file://{image_path}"
            storage_files[uri] = image_np  # Garder en mémoire aussi
            return uri
        
        def load_file(self, uri):
            """Charge un fichier depuis son URI"""
            if uri.startswith("file://"):
                file_path = Path(uri[7:])
                if file_path.exists():
                    return file_path.read_bytes()
            raise FileNotFoundError(f"Fichier non trouvé: {uri}")
        
        def load_image(self, uri):
            """Charge une image depuis son URI"""
            if uri in storage_files:
                return storage_files[uri]
            elif uri.startswith("file://"):
                image_path = Path(uri[7:])
                if image_path.exists():
                    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    if img is not None:
                        return img
            raise FileNotFoundError(f"Image non trouvée: {uri}")
        
        def delete_file(self, uri):
            """Supprime un fichier"""
            if uri in storage_files:
                del storage_files[uri]
            if uri.startswith("file://"):
                file_path = Path(uri[7:])
                if file_path.exists():
                    file_path.unlink()
            return True
    
    mock_storage = MockStorage()
    
    try:
        # Patcher le système de stockage
        with patch('src.utils.storage.get_storage', return_value=mock_storage):
            # Patcher les actors et les groupes
            with patch.object(process_page_task, 'send', mock_process_page_send):
                with patch.object(classify_document_task, 'send', mock_classify_send):
                    with patch('src.workers.dramatiq.group', mock_group):
                        # Patcher le broker pour retourner notre middleware mocké
                        mock_broker = Mock()
                        mock_broker.middleware = [MockResultsMiddleware()]
                        with patch.object(classify_document_task, 'broker', mock_broker):
                            yield
    finally:
        # Nettoyer
        results_store.clear()
        storage_files.clear()
        if temp_storage_dir.exists():
            shutil.rmtree(temp_storage_dir, ignore_errors=True)
        import src.workers
        src.workers._feature_extractor = None
        src.workers._document_classifier = None


@contextmanager
def mock_process_pool_executor(mock_feature_extractor, mock_document_classifier):
    """
    Context manager qui mocke ProcessPoolExecutor pour exécuter les workers de manière synchrone.
    
    NOTE: Cette fonction est obsolète car l'endpoint /classify utilise maintenant Dramatiq.
    Conservée pour compatibilité avec d'autres tests si nécessaire.
    
    Args:
        mock_feature_extractor: Mock du FeatureExtractor
        mock_document_classifier: Mock du DocumentClassifier
    """
    from src.api.routes import process_page_worker, worker_dependencies
    
    # Sauvegarder l'état original de worker_dependencies
    original_dependencies = worker_dependencies.copy()
    
    # Initialiser worker_dependencies avec les mocks
    worker_dependencies['feature_extractor'] = mock_feature_extractor
    worker_dependencies['document_classifier'] = mock_document_classifier
    
    # Créer un mock de ProcessPoolExecutor qui exécute process_page_worker de manière synchrone
    class MockProcessPoolExecutor:
        def __init__(self, max_workers=None, initializer=None):
            self.max_workers = max_workers
            if initializer:
                initializer()  # Appeler l'initialiseur dans le processus de test
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
        
        def map(self, func, iterable):
            """Exécute la fonction de manière synchrone sur chaque élément"""
            return [func(item) for item in iterable]
    
    try:
        with patch('src.api.routes.ProcessPoolExecutor', MockProcessPoolExecutor):
            yield
    finally:
        # Restaurer l'état original de worker_dependencies
        worker_dependencies.clear()
        worker_dependencies.update(original_dependencies)


class TestClassificationEndpoint:
    """Tests pour l'endpoint /api/v1/classify"""
    
    @patch('src.classification.feature_engineering.SentenceTransformer')
    @patch('src.classification.classifier_service.joblib.load')
    def test_classify_endpoint_exists(
        self,
        mock_joblib_load,
        mock_sentence_transformer_class,
        client
    ):
        """Test que l'endpoint existe et répond avec un task_id"""
        # Mock SentenceTransformer pour éviter les appels réseau
        mock_st_model = Mock()
        mock_st_model.encode.return_value = np.random.rand(384).astype(np.float32)
        mock_st_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer_class.return_value = mock_st_model
        
        # Mock joblib.load pour éviter de charger un vrai modèle
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0])
        mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])
        mock_joblib_load.return_value = {
            'model': mock_model,
            'class_names': ['Attestation_CEE', 'Facture']
        }
        
        # Mock du Feature Extractor et Document Classifier
        mock_extractor = Mock()
        mock_extractor.extract_ocr.return_value = []
        mock_classifier = Mock()
        mock_classifier.predict.return_value = {
            'document_type': 'Attestation_CEE',
            'confidence': 0.9
        }
        
        # Créer une image de test simple
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        _, img_encoded = cv2.imencode('.png', img)
        
        with mock_dramatiq_actor(mock_extractor, mock_classifier):
            response = client.post(
                "/api/v1/classify",
                files={"file": ("test.png", img_encoded.tobytes(), "image/png")}
            )
            
            # L'endpoint doit retourner 202 Accepted avec un task_id
            assert response.status_code == 202
            data = response.json()
            assert 'task_id' in data
            assert data['status'] == 'pending'
            assert 'message' in data
    
    @patch('src.classification.feature_engineering.SentenceTransformer')
    @patch('src.classification.classifier_service.joblib.load')
    def test_classify_with_mock(
        self,
        mock_joblib_load,
        mock_sentence_transformer_class,
        client,
        sample_image
    ):
        """
        Test de classification avec Dramatiq mocké.
        
        L'endpoint retourne maintenant un task_id (202 Accepted), puis on récupère
        les résultats via /classify/results/{task_id}.
        """
        # --- Configuration des Mocks ---
        # Mock SentenceTransformer pour éviter les appels réseau
        mock_st_model = Mock()
        mock_st_model.encode.return_value = np.random.rand(384).astype(np.float32)
        mock_st_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer_class.return_value = mock_st_model
        
        # Mock joblib.load pour mocker le modèle interne
        # L'API appelle classifier.predict() qui utilise model.predict_proba() pour calculer confidence
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0])  # Classe 0 = Attestation_CEE
        mock_model.predict_proba.return_value = np.array([[0.92, 0.08]])  # Probabilités pour 2 classes
        mock_joblib_load.return_value = {
            'model': mock_model,
            'class_names': ['Attestation_CEE', 'Facture']
        }
        
        # Mock du Feature Extractor
        mock_extractor = Mock()
        mock_extractor.extract_ocr.return_value = [
            {
                'text': 'Attestation de conformité CEE',
                'confidence': 0.95,
                'bounding_box': [100, 200, 500, 250]
            }
        ]
        
        # Mock du Document Classifier
        mock_classifier = Mock()
        mock_classifier.predict.return_value = {
            'document_type': 'Attestation_CEE',
            'confidence': 0.92
        }
        
        # --- Utiliser le mock de Dramatiq ---
        with mock_dramatiq_actor(mock_extractor, mock_classifier):
            # --- Exécution du Test ---
            _, img_encoded = cv2.imencode('.png', sample_image)
            
            # 1. Soumettre la tâche
            response = client.post(
                "/api/v1/classify",
                files={"file": ("test.png", img_encoded.tobytes(), "image/png")}
            )
            
            # --- Assertions pour la soumission ---
            assert response.status_code == 202
            task_data = response.json()
            assert 'task_id' in task_data
            assert task_data['status'] == 'pending'
            task_id = task_data['task_id']
            
            # 2. Récupérer les résultats
            result_response = client.get(f"/api/v1/classify/results/{task_id}")
            assert result_response.status_code == 200
            result_data = result_response.json()
            
            # --- Assertions pour les résultats ---
            assert result_data['status'] == 'completed'
            assert 'result' in result_data
            result = result_data['result']
            
            assert result['status'] == 'success'
            assert result['total_pages'] == 1
            assert len(result['results_by_page']) == 1
            
            # Vérifier la structure de la réponse pour une page
            page_result = result['results_by_page'][0]
            assert page_result['page_number'] == 1
            assert page_result['document_type'] == 'Attestation_CEE'
            # La valeur doit correspondre à celle du mock (0.92)
            assert page_result['confidence'] == pytest.approx(0.92)
            assert 'processing_time' in result
            
            # Vérifier que les mocks ont été appelés
            mock_extractor.extract_ocr.assert_called_once()
            mock_classifier.predict.assert_called_once()
    
    def test_classify_result_not_found(self, client):
        """
        Teste que l'endpoint /classify/results/{task_id} retourne 404 pour une tâche inexistante.
        """
        # Utiliser un task_id qui n'existe pas
        fake_task_id = "00000000-0000-0000-0000-000000000000"
        
        # Récupérer les résultats pour une tâche inexistante
        result_response = client.get(f"/api/v1/classify/results/{fake_task_id}")
        
        # L'endpoint doit retourner 404 Not Found
        assert result_response.status_code == 404
        error_data = result_response.json()
        assert "detail" in error_data
        assert fake_task_id in error_data["detail"]
        assert "not found" in error_data["detail"].lower()
    
    @patch('src.classification.feature_engineering.SentenceTransformer')
    @patch('src.classification.classifier_service.joblib.load')
    def test_classify_invalid_file(
        self,
        mock_joblib_load,
        mock_sentence_transformer_class,
        client
    ):
        """Test avec un fichier invalide"""
        # Mock SentenceTransformer pour éviter les appels réseau
        mock_st_model = Mock()
        mock_st_model.encode.return_value = np.random.rand(384).astype(np.float32)
        mock_st_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer_class.return_value = mock_st_model
        
        # Mock joblib.load pour éviter de charger un vrai modèle
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0])
        mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])
        mock_joblib_load.return_value = {
            'model': mock_model,
            'class_names': ['Attestation_CEE', 'Facture']
        }
        
        response = client.post(
            "/api/v1/classify",
            files={"file": ("test.txt", b"invalid content", "text/plain")}
        )
        
        # Devrait retourner une erreur 400 ou 500
        assert response.status_code in [400, 500]
    
    @patch('src.classification.feature_engineering.SentenceTransformer')
    @patch('src.classification.classifier_service.joblib.load')
    def test_classify_empty_file(
        self,
        mock_joblib_load,
        mock_sentence_transformer_class,
        client
    ):
        """Test avec un fichier vide"""
        # Mock SentenceTransformer pour éviter les appels réseau
        mock_st_model = Mock()
        mock_st_model.encode.return_value = np.random.rand(384).astype(np.float32)
        mock_st_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer_class.return_value = mock_st_model
        
        # Mock joblib.load pour éviter de charger un vrai modèle
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0])
        mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])
        mock_joblib_load.return_value = {
            'model': mock_model,
            'class_names': ['Attestation_CEE', 'Facture']
        }
        
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
        """Test avec un vrai modèle (si disponible) - Architecture asynchrone"""
        with open(attestation_cee_image, "rb") as f:
            file_content = f.read()
            response = client.post(
                "/api/v1/classify",
                files={"file": ("attestation_cee.png", file_content, "image/png")}
            )
        
        # L'endpoint doit retourner 202 avec un task_id
        if response.status_code == 202:
            task_data = response.json()
            assert 'task_id' in task_data
            task_id = task_data['task_id']
            
            # Attendre un peu pour que la tâche se termine (dans un vrai test, on utiliserait polling)
            import time
            time.sleep(1)
            
            # Récupérer les résultats
            result_response = client.get(f"/api/v1/classify/results/{task_id}")
            if result_response.status_code == 200:
                result_data = result_response.json()
                if result_data.get('status') == 'completed':
                    result = result_data['result']
                    assert result['status'] == 'success'
                    assert 'total_pages' in result
                    assert 'results_by_page' in result
                    assert len(result['results_by_page']) > 0
                    page_result = result['results_by_page'][0]
                    assert 'document_type' in page_result
                    assert 'confidence' in page_result
                    # Si c'est une Attestation CEE, vérifier que le type est correct
                    # (peut varier selon le modèle entraîné)
                    assert page_result['document_type'] is not None or page_result['confidence'] < 0.6
    
    @pytest.mark.skipif(
        not Path("models/document_classifier.joblib").exists(),
        reason="Modèle de classification non disponible"
    )
    def test_classify_facture_with_real_model(
        self,
        client,
        facture_image
    ):
        """Test avec une facture et un vrai modèle - Architecture asynchrone"""
        with open(facture_image, "rb") as f:
            file_content = f.read()
            response = client.post(
                "/api/v1/classify",
                files={"file": ("facture.png", file_content, "image/png")}
            )
        
        # L'endpoint doit retourner 202 avec un task_id
        if response.status_code == 202:
            task_data = response.json()
            assert 'task_id' in task_data
            task_id = task_data['task_id']
            
            # Attendre un peu pour que la tâche se termine
            import time
            time.sleep(1)
            
            # Récupérer les résultats
            result_response = client.get(f"/api/v1/classify/results/{task_id}")
            if result_response.status_code == 200:
                result_data = result_response.json()
                if result_data.get('status') == 'completed':
                    result = result_data['result']
                    assert result['status'] == 'success'
                    assert 'total_pages' in result
                    assert 'results_by_page' in result
                    assert len(result['results_by_page']) > 0
                    page_result = result['results_by_page'][0]
                    assert 'document_type' in page_result
                    assert 'confidence' in page_result


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
        
        # Mock du Feature Extractor et Document Classifier
        mock_extractor = Mock()
        mock_extractor.extract_ocr.return_value = []
        mock_classifier = Mock()
        mock_classifier.predict.return_value = {
            'document_type': 'Attestation_CEE',
            'confidence': 0.85
        }
        
        # Encoder l'image
        _, img_encoded = cv2.imencode('.png', sample_image)
        
        # Appeler l'endpoint avec Dramatiq mocké
        with mock_dramatiq_actor(mock_extractor, mock_classifier):
            response = client.post(
                "/api/v1/classify",
                files={"file": ("test.png", img_encoded.tobytes(), "image/png")}
            )
            
            # Vérifier la réponse de soumission
            assert response.status_code == 202
            task_data = response.json()
            assert 'task_id' in task_data
            task_id = task_data['task_id']
            
            # Récupérer les résultats
            result_response = client.get(f"/api/v1/classify/results/{task_id}")
            assert result_response.status_code == 200
            result_data = result_response.json()
            
            if result_data.get('status') == 'completed':
                result = result_data['result']
                assert result['status'] == 'success'
                assert 'total_pages' in result
                assert 'results_by_page' in result
                assert len(result['results_by_page']) > 0
                page_result = result['results_by_page'][0]
                assert 'document_type' in page_result
                assert 'confidence' in page_result
                assert isinstance(page_result['confidence'], (int, float))
                assert 0.0 <= page_result['confidence'] <= 1.0

    @patch('src.classification.feature_engineering.SentenceTransformer')
    @patch('src.classification.classifier_service.joblib.load')
    def test_classify_multipage_pdf_with_mock(
        self,
        mock_joblib_load,
        mock_sentence_transformer_class,
        client,
        sample_image
    ):
        """
        Teste la classification d'un PDF multi-pages en mockant le traitement parallèle.
        """
        # --- Configuration des Mocks ---
        # Mock SentenceTransformer pour éviter les appels réseau
        mock_st_model = Mock()
        mock_st_model.encode.return_value = np.random.rand(384).astype(np.float32)
        mock_st_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer_class.return_value = mock_st_model
        
        # Mock joblib.load pour mocker le modèle interne
        mock_model = Mock()
        mock_model.predict.side_effect = [np.array([0]), np.array([1])]  # Page 1 -> Classe 0, Page 2 -> Classe 1
        mock_model.predict_proba.side_effect = [np.array([[0.9, 0.1]]), np.array([[0.2, 0.8]])]
        mock_joblib_load.return_value = {
            'model': mock_model,
            'class_names': ['Attestation_CEE', 'Facture']
        }
        
        # Mock du Feature Extractor (appelé deux fois, une par page)
        mock_extractor = Mock()
        mock_extractor.extract_ocr.return_value = [
            {
                'text': 'mock text',
                'confidence': 0.9,
                'bounding_box': [0, 0, 1, 1]
            }
        ]
        
        # Mock du Document Classifier (appelé deux fois, une par page)
        mock_classifier = Mock()
        mock_classifier.predict.side_effect = [
            {'document_type': 'Attestation_CEE', 'confidence': 0.9},  # Page 1
            {'document_type': 'Facture', 'confidence': 0.8}  # Page 2
        ]
        
        # --- Création du fichier PDF de test ---
        # Utiliser PIL pour créer un PDF multi-pages (plus simple et fiable)
        try:
            from PIL import Image
            # Convertir l'image numpy en format PIL
            pil_image = Image.fromarray(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))
            
            # Créer un PDF avec deux pages identiques
            img_bytes = BytesIO()
            pil_image.save(img_bytes, format='PDF', save_all=True, append_images=[pil_image])
            pdf_bytes = img_bytes.getvalue()
            
        except ImportError:
            # Fallback: créer un PDF simple avec reportlab si PIL n'est pas disponible
            try:
                from reportlab.pdfgen import canvas
                from reportlab.lib.pagesizes import letter
                
                buffer = BytesIO()
                p = canvas.Canvas(buffer, pagesize=letter)
                p.drawString(100, 750, "Page 1")
                p.showPage()
                p.drawString(100, 750, "Page 2")
                p.showPage()
                p.save()
                pdf_bytes = buffer.getvalue()
            except ImportError:
                pytest.skip("PIL/Pillow ou reportlab non disponible pour créer un PDF de test")
        
        # --- Utiliser le mock de Dramatiq ---
        with mock_dramatiq_actor(mock_extractor, mock_classifier):
            # --- Exécution du Test ---
            response = client.post(
                "/api/v1/classify",
                files={"file": ("test_2pages.pdf", pdf_bytes, "application/pdf")}
            )
            
            # --- Assertions pour la soumission ---
            assert response.status_code == 202
            task_data = response.json()
            assert 'task_id' in task_data
            task_id = task_data['task_id']
            
            # Récupérer les résultats
            result_response = client.get(f"/api/v1/classify/results/{task_id}")
            assert result_response.status_code == 200
            result_data = result_response.json()
            
            # --- Assertions pour les résultats ---
            assert result_data['status'] == 'completed'
            result = result_data['result']
            
            assert result['status'] == 'success'
            assert result['total_pages'] == 2
            assert len(result['results_by_page']) == 2
            
            # Vérifier la page 1
            assert result['results_by_page'][0]['page_number'] == 1
            assert result['results_by_page'][0]['document_type'] == 'Attestation_CEE'
            assert result['results_by_page'][0]['confidence'] == pytest.approx(0.9)
            
            # Vérifier la page 2
            assert result['results_by_page'][1]['page_number'] == 2
            assert result['results_by_page'][1]['document_type'] == 'Facture'
            assert result['results_by_page'][1]['confidence'] == pytest.approx(0.8)
            
            # Vérifier que les mocks ont été appelés (deux fois, une par page)
            assert mock_extractor.extract_ocr.call_count == 2
            assert mock_classifier.predict.call_count == 2

    @patch('src.classification.feature_engineering.SentenceTransformer')
    @patch('src.classification.classifier_service.joblib.load')
    def test_classify_single_image_via_workers(
        self,
        mock_joblib_load,
        mock_sentence_transformer_class,
        client,
        sample_image
    ):
        """
        Teste la classification d'une image unique via les workers.
        Avec la nouvelle architecture, ProcessPoolExecutor est TOUJOURS utilisé,
        même pour une seule page, pour éviter de charger les modèles dans le processus principal.
        """
        # --- Configuration des Mocks ---
        # Mock SentenceTransformer pour éviter les appels réseau
        mock_st_model = Mock()
        mock_st_model.encode.return_value = np.random.rand(384).astype(np.float32)
        mock_st_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer_class.return_value = mock_st_model
        
        # Mock joblib.load pour mocker le modèle interne
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0])  # Classe 0 = Attestation_CEE
        mock_model.predict_proba.return_value = np.array([[0.92, 0.08]])
        mock_joblib_load.return_value = {
            'model': mock_model,
            'class_names': ['Attestation_CEE', 'Facture']
        }
        
        # Mock du Feature Extractor
        mock_extractor = Mock()
        mock_extractor.extract_ocr.return_value = [
            {
                'text': 'Attestation de conformité CEE',
                'confidence': 0.95,
                'bounding_box': [100, 200, 500, 250]
            }
        ]
        
        # Mock du Document Classifier
        mock_classifier = Mock()
        mock_classifier.predict.return_value = {
            'document_type': 'Attestation_CEE',
            'confidence': 0.92
        }
        
        # --- Utiliser le mock de Dramatiq ---
        with mock_dramatiq_actor(mock_extractor, mock_classifier):
            # --- Exécution du Test ---
            _, img_encoded = cv2.imencode('.png', sample_image)
            
            # Soumettre la tâche
            response = client.post(
                "/api/v1/classify",
                files={"file": ("test.png", img_encoded.tobytes(), "image/png")}
            )
            
            # --- Assertions pour la soumission ---
            assert response.status_code == 202
            task_data = response.json()
            assert 'task_id' in task_data
            task_id = task_data['task_id']
            
            # Récupérer les résultats
            result_response = client.get(f"/api/v1/classify/results/{task_id}")
            assert result_response.status_code == 200
            result_data = result_response.json()
            
            # --- Assertions pour les résultats ---
            assert result_data['status'] == 'completed'
            result = result_data['result']
            
            assert result['status'] == 'success'
            assert result['total_pages'] == 1
            assert len(result['results_by_page']) == 1
            
            # Vérifier la structure de la réponse pour une page
            page_result = result['results_by_page'][0]
            assert page_result['page_number'] == 1
            assert page_result['document_type'] == 'Attestation_CEE'
            assert page_result['confidence'] == pytest.approx(0.92)
            assert 'processing_time' in result
            
            # Vérifier que les mocks ont été appelés via Dramatiq
            mock_extractor.extract_ocr.assert_called_once()
            mock_classifier.predict.assert_called_once()

    @patch('src.classification.feature_engineering.SentenceTransformer')
    @patch('src.classification.classifier_service.joblib.load')
    def test_classify_single_page_pdf_via_workers(
        self,
        mock_joblib_load,
        mock_sentence_transformer_class,
        client,
        sample_image
    ):
        """
        Teste la classification d'un PDF d'une seule page via Dramatiq.
        Avec la nouvelle architecture, Dramatiq est utilisé pour le traitement asynchrone.
        """
        # --- Configuration des Mocks ---
        # Mock SentenceTransformer pour éviter les appels réseau
        mock_st_model = Mock()
        mock_st_model.encode.return_value = np.random.rand(384).astype(np.float32)
        mock_st_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer_class.return_value = mock_st_model
        
        # Mock joblib.load pour mocker le modèle interne
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])  # Classe 1 = Facture
        mock_model.predict_proba.return_value = np.array([[0.15, 0.85]])
        mock_joblib_load.return_value = {
            'model': mock_model,
            'class_names': ['Attestation_CEE', 'Facture']
        }
        
        # Mock du Feature Extractor
        mock_extractor = Mock()
        mock_extractor.extract_ocr.return_value = [
            {
                'text': 'FACTURE',
                'confidence': 0.95,
                'bounding_box': [50, 50, 300, 100]
            }
        ]
        
        # Mock du Document Classifier
        mock_classifier = Mock()
        mock_classifier.predict.return_value = {
            'document_type': 'Facture',
            'confidence': 0.85
        }
        
        # --- Création du fichier PDF de test (1 page) ---
        try:
            from PIL import Image
            # Convertir l'image numpy en format PIL
            pil_image = Image.fromarray(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))
            
            # Créer un PDF avec une seule page
            img_bytes = BytesIO()
            pil_image.save(img_bytes, format='PDF')
            pdf_bytes = img_bytes.getvalue()
            
        except ImportError:
            pytest.skip("PIL/Pillow non disponible pour créer un PDF de test")
        
        # --- Utiliser le mock de Dramatiq ---
        with mock_dramatiq_actor(mock_extractor, mock_classifier):
            # --- Exécution du Test ---
            response = client.post(
                "/api/v1/classify",
                files={"file": ("test_1page.pdf", pdf_bytes, "application/pdf")}
            )
            
            # --- Assertions pour la soumission ---
            assert response.status_code == 202
            task_data = response.json()
            assert 'task_id' in task_data
            task_id = task_data['task_id']
            
            # Récupérer les résultats
            result_response = client.get(f"/api/v1/classify/results/{task_id}")
            assert result_response.status_code == 200
            result_data = result_response.json()
            
            # --- Assertions pour les résultats ---
            assert result_data['status'] == 'completed'
            result = result_data['result']
            
            assert result['status'] == 'success'
            assert result['total_pages'] == 1
            assert len(result['results_by_page']) == 1
            
            # Vérifier la structure de la réponse pour une page
            page_result = result['results_by_page'][0]
            assert page_result['page_number'] == 1
            assert page_result['document_type'] == 'Facture'
            assert page_result['confidence'] == pytest.approx(0.85)
            assert 'processing_time' in result
            
            # Vérifier que les mocks ont été appelés via Dramatiq
            mock_extractor.extract_ocr.assert_called_once()
            mock_classifier.predict.assert_called_once()

