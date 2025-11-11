"""
Tests d'intégration pour le pipeline d'entraînement complet.

Ces tests vérifient que le refactoring des chemins d'entraînement fonctionne correctement :
- Les chemins par défaut de config.yaml sont utilisés quand aucun argument CLI n'est fourni
- Les arguments CLI surchargent correctement les chemins par défaut
"""

import pytest
import json
import yaml
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Machine Learning
from sklearn.linear_model import LogisticRegression

# Importer les fonctions main des scripts
from training.prepare_training_data import main as prepare_main
from training.train_classifier import main as train_main


def create_dummy_image(path: Path):
    """Crée une image PNG blanche de 100x100 avec des rectangles noirs."""
    image = np.ones((100, 100, 3), dtype=np.uint8) * 255  # Fond blanc
    # Ajoute quelques rectangles pour simuler des lignes de texte
    cv2.rectangle(image, (10, 10), (90, 20), (0, 0, 0), -1)
    cv2.rectangle(image, (10, 30), (70, 40), (0, 0, 0), -1)
    cv2.imwrite(str(path), image)


def get_base_config_content():
    """Retourne le contenu de base pour config.yaml (sans section paths)."""
    return {
        "features": {
            "ocr": {
                "enabled": True,
                "default_language": "fr",
                "use_gpu": False,
                "min_confidence": 0.70
            }
        },
        "classification": {
            "enabled": True,
            "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "min_confidence": 0.70
        }
    }


def get_dummy_model(X, y):
    """
    Entraîne un vrai modèle sklearn simple qui est sérialisable.
    
    Args:
        X: Features d'entraînement
        y: Labels d'entraînement
    
    Returns:
        Modèle LogisticRegression entraîné
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    # sklearn a besoin d'au moins 1 sample par classe pour le fit
    if len(np.unique(y)) > 1 and all(np.sum(y == c) > 0 for c in np.unique(y)):
        model.fit(X, y)
    return model


def setup_mocks(mock_sentence_transformer_class, mock_feature_extractor_class, 
                mock_feature_engineer_class, mock_train_model_func, 
                mock_feature_extractor_process):
    """Configure tous les mocks nécessaires pour les tests."""
    # --- Mock de SentenceTransformer ---
    # On simule l'objet retourné par SentenceTransformer('model-name')
    mock_st_instance = MagicMock()
    # La méthode .encode() doit retourner un vecteur numpy
    mock_st_instance.encode.return_value = np.random.rand(384).astype(np.float32)
    # La méthode .get_sentence_embedding_dimension() doit retourner la taille du vecteur
    mock_st_instance.get_sentence_embedding_dimension.return_value = 384
    # On configure la classe mockée pour qu'elle retourne notre instance
    mock_sentence_transformer_class.return_value = mock_st_instance
    
    # Mock FeatureExtractor
    mock_extractor_instance = MagicMock()
    mock_extractor_instance.process = mock_feature_extractor_process
    mock_feature_extractor_class.return_value = mock_extractor_instance
    
    # Mock FeatureEngineer
    # On va le laisser s'initialiser, car il va maintenant utiliser notre SentenceTransformer mocké.
    # On peut simplifier son mock.
    from src.classification.feature_engineering import FeatureEngineer
    mock_feature_engineer_class.side_effect = lambda **kwargs: FeatureEngineer(**kwargs)
    
    # Mock train_model : utiliser un vrai modèle sklearn au lieu d'un MagicMock
    # Configure le mock pour appeler notre fonction qui retourne un vrai modèle
    mock_train_model_func.side_effect = lambda X_train, y_train, model_type, **kwargs: get_dummy_model(X_train, y_train)


def reset_config_cache():
    """Réinitialise le cache global de configuration."""
    import src.utils.config_loader as config_module
    config_module._global_config = None


@pytest.fixture
def mock_feature_extractor_process():
    """Mock de FeatureExtractor.process() pour éviter l'OCR réel."""
    def mock_process(input_path, output_path):
        from src.pipeline.models import FeaturesOutput, OCRLine
        
        # Créer un résultat mock avec des données OCR minimales
        ocr_lines = [
            OCRLine(
                text="Test document",
                confidence=0.95,
                bounding_box=[0.1, 0.1, 0.5, 0.2]
            )
        ]
        
        # Sauvegarder le JSON comme le vrai process() le ferait
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        features_dict = {
            'ocr_lines': [line.model_dump() for line in ocr_lines],
            'checkboxes': []
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(features_dict, f, indent=2, default=str)
        
        return FeaturesOutput(
            status='success',
            input_path=input_path,
            output_path=output_path,
            processing_time=0.1,
            ocr_lines=ocr_lines,
            checkboxes=[]
        )
    
    return mock_process


@pytest.fixture
def training_setup_default_paths(tmp_path):
    """
    Crée l'arborescence de fichiers et la config pour un test avec chemins par défaut.
    
    Returns:
        dict: Dictionnaire avec les chemins utiles (root, raw, processed, artifacts)
    """
    project_root = tmp_path
    
    # Définir les chemins avec deux classes pour permettre stratify dans train_test_split
    raw_dir_class_a = project_root / "training_data" / "raw" / "ClassA"
    raw_dir_class_b = project_root / "training_data" / "raw" / "ClassB"
    processed_dir = project_root / "training_data" / "processed"
    artifacts_dir = project_root / "training_data" / "artifacts"
    
    raw_dir_class_a.mkdir(parents=True)
    raw_dir_class_b.mkdir(parents=True)
    
    # Créer 5 images pour ClassA
    for i in range(5):
        create_dummy_image(raw_dir_class_a / f"doc{i}_A.png")
    
    # Créer 5 images pour ClassB
    for i in range(5):
        create_dummy_image(raw_dir_class_b / f"doc{i}_B.png")
    
    # Créer le config.yaml avec chemins par défaut
    config_content = get_base_config_content()
    config_content["paths"] = {
        "training_raw_dir": "training_data/raw",
        "training_processed_dir": "training_data/processed",
        "training_artifacts_dir": "training_data/artifacts"
    }
    
    with open(project_root / "config.yaml", "w", encoding='utf-8') as f:
        yaml.dump(config_content, f)
    
    return {
        "root": project_root,
        "raw": raw_dir_class_a.parent,  # training_data/raw
        "raw_class_a": raw_dir_class_a,  # training_data/raw/ClassA
        "raw_class_b": raw_dir_class_b,  # training_data/raw/ClassB
        "processed": processed_dir,
        "artifacts": artifacts_dir
    }


@pytest.fixture
def training_setup_cli_overrides(tmp_path):
    """
    Crée l'arborescence pour tester la surcharge CLI.
    
    Returns:
        dict: Dictionnaire avec les chemins (default et external)
    """
    project_root = tmp_path
    
    # Chemins par défaut (qui ne doivent PAS être utilisés)
    default_raw_dir = project_root / "training_data" / "raw"
    default_raw_dir.mkdir(parents=True)
    
    # Chemins externes (qui DOIVENT être utilisés)
    external_data_path = project_root / "external_drive"
    external_raw_dir_class_a = external_data_path / "my_images" / "ClassA"
    external_raw_dir_class_b = external_data_path / "my_images" / "ClassB"
    external_processed_dir = external_data_path / "my_json_data"
    external_artifacts_dir = external_data_path / "my_models"
    
    external_raw_dir_class_a.mkdir(parents=True)
    external_raw_dir_class_b.mkdir(parents=True)
    
    # Créer 5 images pour ClassA
    for i in range(5):
        create_dummy_image(external_raw_dir_class_a / f"doc{i}_A.png")
    
    # Créer 5 images pour ClassB
    for i in range(5):
        create_dummy_image(external_raw_dir_class_b / f"doc{i}_B.png")
    
    # Le config.yaml pointe vers les chemins par défaut
    config_content = get_base_config_content()
    config_content["paths"] = {
        "training_raw_dir": str(default_raw_dir),
        "training_processed_dir": str(project_root / "training_data" / "processed"),
        "training_artifacts_dir": str(project_root / "training_data" / "artifacts")
    }
    
    with open(project_root / "config.yaml", "w", encoding='utf-8') as f:
        yaml.dump(config_content, f)
    
    return {
        "root": project_root,
        "default": {
            "raw": default_raw_dir,
            "processed": project_root / "training_data" / "processed",
            "artifacts": project_root / "training_data" / "artifacts"
        },
        "external": {
            "raw": external_raw_dir_class_a.parent,  # my_images
            "raw_class_a": external_raw_dir_class_a,  # my_images/ClassA
            "raw_class_b": external_raw_dir_class_b,  # my_images/ClassB
            "processed": external_processed_dir,
            "artifacts": external_artifacts_dir
        }
    }


@pytest.fixture
def training_setup_no_paths(tmp_path):
    """
    Crée l'arborescence pour tester sans section paths dans config.yaml.
    
    Returns:
        dict: Dictionnaire avec les chemins
    """
    project_root = tmp_path
    
    # Définir les chemins avec deux classes pour permettre stratify dans train_test_split
    raw_dir_class_a = project_root / "training_data" / "raw" / "ClassA"
    raw_dir_class_b = project_root / "training_data" / "raw" / "ClassB"
    
    raw_dir_class_a.mkdir(parents=True)
    raw_dir_class_b.mkdir(parents=True)
    
    # Créer 5 images pour ClassA
    for i in range(5):
        create_dummy_image(raw_dir_class_a / f"doc{i}_A.png")
    
    # Créer 5 images pour ClassB
    for i in range(5):
        create_dummy_image(raw_dir_class_b / f"doc{i}_B.png")
    
    # Config sans section paths
    config_content = get_base_config_content()
    
    with open(project_root / "config.yaml", "w", encoding='utf-8') as f:
        yaml.dump(config_content, f)
    
    return {
        "root": project_root,
        "raw": raw_dir_class_a.parent,  # training_data/raw
        "raw_class_a": raw_dir_class_a,  # training_data/raw/ClassA
        "raw_class_b": raw_dir_class_b,  # training_data/raw/ClassB
        "processed": project_root / "training_data" / "processed"
    }


@patch('training.train_classifier.train_model')
@patch('src.classification.feature_engineering.FeatureEngineer')
@patch('training.prepare_training_data.FeatureExtractor')
@patch('src.classification.feature_engineering.SentenceTransformer')
def test_full_pipeline_with_default_paths(
    mock_sentence_transformer_class,
    mock_feature_extractor_class,
    mock_feature_engineer_class,
    mock_train_model_func,
    training_setup_default_paths,
    monkeypatch,
    mock_feature_extractor_process
):
    """
    Vérifie que le pipeline complet s'exécute correctement en utilisant
    uniquement les chemins par défaut de config.yaml.
    """
    setup_mocks(mock_sentence_transformer_class, mock_feature_extractor_class, 
                mock_feature_engineer_class, mock_train_model_func, 
                mock_feature_extractor_process)
    
    setup = training_setup_default_paths
    monkeypatch.chdir(setup["root"])
    reset_config_cache()
    
    # EXÉCUTION
    with patch('sys.argv', ['prepare_training_data.py', '--workers', '2']):
        result = prepare_main()
        assert result == 0, "prepare_training_data.py devrait retourner 0"
    
    with patch('sys.argv', ['train_classifier.py', '--workers', '2']):
        result = train_main()
        assert result == 0, "train_classifier.py devrait retourner 0"
    
    # VÉRIFICATION
    # Vérifier les fichiers JSON pour ClassA
    assert (setup["processed"] / "ClassA" / "doc1_A.json").exists(), \
        f"Le fichier JSON devrait être créé dans {setup['processed'] / 'ClassA' / 'doc1_A.json'}"
    assert (setup["processed"] / "ClassA" / "doc2_A.json").exists(), \
        f"Le fichier JSON devrait être créé dans {setup['processed'] / 'ClassA' / 'doc2_A.json'}"
    # Vérifier les fichiers JSON pour ClassB
    assert (setup["processed"] / "ClassB" / "doc1_B.json").exists(), \
        f"Le fichier JSON devrait être créé dans {setup['processed'] / 'ClassB' / 'doc1_B.json'}"
    assert (setup["processed"] / "ClassB" / "doc2_B.json").exists(), \
        f"Le fichier JSON devrait être créé dans {setup['processed'] / 'ClassB' / 'doc2_B.json'}"
    assert (setup["artifacts"] / "document_classifier.joblib").exists(), \
        f"Le modèle devrait être sauvegardé dans {setup['artifacts'] / 'document_classifier.joblib'}"
    assert (setup["artifacts"] / "training_report.json").exists(), \
        f"Le rapport devrait être sauvegardé dans {setup['artifacts'] / 'training_report.json'}"


@patch('training.train_classifier.train_model')
@patch('src.classification.feature_engineering.FeatureEngineer')
@patch('training.prepare_training_data.FeatureExtractor')
@patch('src.classification.feature_engineering.SentenceTransformer')
def test_full_pipeline_with_cli_overrides(
    mock_sentence_transformer_class,
    mock_feature_extractor_class,
    mock_feature_engineer_class,
    mock_train_model_func,
    training_setup_cli_overrides,
    monkeypatch,
    mock_feature_extractor_process
):
    """
    Vérifie que les arguments de ligne de commande surchargent correctement
    les chemins par défaut de config.yaml.
    """
    setup_mocks(mock_sentence_transformer_class, mock_feature_extractor_class, 
                mock_feature_engineer_class, mock_train_model_func, 
                mock_feature_extractor_process)
    
    setup = training_setup_cli_overrides
    monkeypatch.chdir(setup["root"])
    reset_config_cache()
    
    # EXÉCUTION avec arguments
    prepare_args = [
        'prepare_training_data.py',
        '--input-dir', str(setup["external"]["raw"]),
        '--output-dir', str(setup["external"]["processed"])
    ]
    
    with patch('sys.argv', prepare_args):
        result = prepare_main()
        assert result == 0, "prepare_training_data.py devrait retourner 0"
    
    train_args = [
        'train_classifier.py',
        '--data-dir', str(setup["external"]["processed"]),
        '--model-path', str(setup["external"]["artifacts"] / "model.joblib"),
        '--report-path', str(setup["external"]["artifacts"] / "report.json")
    ]
    
    with patch('sys.argv', train_args):
        result = train_main()
        assert result == 0, "train_classifier.py devrait retourner 0"
    
    # VÉRIFICATION
    # Vérifier que les fichiers ont été créés dans les dossiers EXTERNES
    # Vérifier les fichiers JSON pour ClassA
    assert (setup["external"]["processed"] / "ClassA" / "doc1_A.json").exists(), \
        f"Le fichier JSON devrait être créé dans {setup['external']['processed'] / 'ClassA' / 'doc1_A.json'}"
    assert (setup["external"]["processed"] / "ClassA" / "doc2_A.json").exists(), \
        f"Le fichier JSON devrait être créé dans {setup['external']['processed'] / 'ClassA' / 'doc2_A.json'}"
    # Vérifier les fichiers JSON pour ClassB
    assert (setup["external"]["processed"] / "ClassB" / "doc1_B.json").exists(), \
        f"Le fichier JSON devrait être créé dans {setup['external']['processed'] / 'ClassB' / 'doc1_B.json'}"
    assert (setup["external"]["processed"] / "ClassB" / "doc2_B.json").exists(), \
        f"Le fichier JSON devrait être créé dans {setup['external']['processed'] / 'ClassB' / 'doc2_B.json'}"
    assert (setup["external"]["artifacts"] / "model.joblib").exists(), \
        f"Le modèle devrait être sauvegardé dans {setup['external']['artifacts'] / 'model.joblib'}"
    assert (setup["external"]["artifacts"] / "report.json").exists(), \
        f"Le rapport devrait être sauvegardé dans {setup['external']['artifacts'] / 'report.json'}"
    
    # Vérifier qu'AUCUN fichier n'a été créé dans les dossiers par défaut
    if setup["default"]["processed"].exists():
        assert not any(setup["default"]["processed"].iterdir()), \
            "Aucun fichier ne devrait être créé dans le dossier processed par défaut"
    
    if setup["default"]["artifacts"].exists():
        assert not any(setup["default"]["artifacts"].iterdir()), \
            "Aucun fichier ne devrait être créé dans le dossier artifacts par défaut"


@patch('training.train_classifier.train_model')
@patch('src.classification.feature_engineering.FeatureEngineer')
@patch('training.prepare_training_data.FeatureExtractor')
@patch('src.classification.feature_engineering.SentenceTransformer')
def test_pipeline_with_missing_config_paths(
    mock_sentence_transformer_class,
    mock_feature_extractor_class,
    mock_feature_engineer_class,
    mock_train_model_func,
    training_setup_no_paths,
    monkeypatch,
    mock_feature_extractor_process
):
    """
    Vérifie que le système ne crash pas si la configuration des chemins est absente.
    Les scripts doivent utiliser des valeurs par défaut hardcodées.
    """
    setup_mocks(mock_sentence_transformer_class, mock_feature_extractor_class, 
                mock_feature_engineer_class, mock_train_model_func, 
                mock_feature_extractor_process)
    
    setup = training_setup_no_paths
    monkeypatch.chdir(setup["root"])
    reset_config_cache()
    
    # EXÉCUTION : Les scripts doivent utiliser les valeurs par défaut hardcodées
    with patch('sys.argv', ['prepare_training_data.py', '--workers', '2']):
        result = prepare_main()
        assert result == 0, "prepare_training_data.py devrait fonctionner même sans paths dans config"
    
    # Vérifier que les fichiers ont été créés avec les chemins par défaut hardcodés
    # Vérifier les fichiers JSON pour ClassA
    assert (setup["processed"] / "ClassA" / "doc1_A.json").exists(), \
        "Le fichier devrait être créé avec les chemins par défaut hardcodés"
    assert (setup["processed"] / "ClassA" / "doc2_A.json").exists(), \
        "Le fichier devrait être créé avec les chemins par défaut hardcodés"
    # Vérifier les fichiers JSON pour ClassB
    assert (setup["processed"] / "ClassB" / "doc1_B.json").exists(), \
        "Le fichier devrait être créé avec les chemins par défaut hardcodés"
    assert (setup["processed"] / "ClassB" / "doc2_B.json").exists(), \
        "Le fichier devrait être créé avec les chemins par défaut hardcodés"
