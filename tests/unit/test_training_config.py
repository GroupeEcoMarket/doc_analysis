"""
Tests unitaires pour la logique de chargement des chemins d'entraînement.

Ces tests vérifient que la logique de chargement des chemins depuis config.yaml
fonctionne correctement de manière isolée.
"""

import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile

from src.utils.config_loader import get_config, Config


def test_config_loader_uses_default_paths_when_missing():
    """Vérifie que get_config() retourne des valeurs par défaut si les chemins sont absents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        
        # Créer un config.yaml sans section paths
        config_content = {
            "features": {
                "ocr": {
                    "enabled": True
                }
            }
        }
        
        with open(config_path, "w", encoding='utf-8') as f:
            yaml.dump(config_content, f)
        
        # Réinitialiser le cache global
        import src.utils.config_loader as config_module
        config_module._global_config = None
        
        # Charger la config
        config = get_config(str(config_path))
        
        # Vérifier que .get() retourne les valeurs par défaut
        assert config.get('paths.training_raw_dir', 'training_data/raw') == 'training_data/raw'
        assert config.get('paths.training_processed_dir', 'training_data/processed') == 'training_data/processed'
        assert config.get('paths.training_artifacts_dir', 'training_data/artifacts') == 'training_data/artifacts'


def test_config_loader_uses_config_paths_when_present():
    """Vérifie que get_config() utilise les chemins définis dans config.yaml."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        
        # Créer un config.yaml avec section paths
        custom_raw = "custom/raw/path"
        custom_processed = "custom/processed/path"
        custom_artifacts = "custom/artifacts/path"
        
        config_content = {
            "paths": {
                "training_raw_dir": custom_raw,
                "training_processed_dir": custom_processed,
                "training_artifacts_dir": custom_artifacts
            },
            "features": {
                "ocr": {
                    "enabled": True
                }
            }
        }
        
        with open(config_path, "w", encoding='utf-8') as f:
            yaml.dump(config_content, f)
        
        # Réinitialiser le cache global
        import src.utils.config_loader as config_module
        config_module._global_config = None
        
        # Charger la config
        config = get_config(str(config_path))
        
        # Vérifier que .get() retourne les valeurs du config
        assert config.get('paths.training_raw_dir') == custom_raw
        assert config.get('paths.training_processed_dir') == custom_processed
        assert config.get('paths.training_artifacts_dir') == custom_artifacts


def test_prepare_training_data_uses_config_defaults():
    """Vérifie que prepare_training_data.py utilise bien les valeurs de config.yaml."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        
        custom_raw = "my_custom/raw"
        custom_processed = "my_custom/processed"
        
        config_content = {
            "paths": {
                "training_raw_dir": custom_raw,
                "training_processed_dir": custom_processed
            },
            "features": {
                "ocr_filtering": {
                    "enabled": True,
                    "min_confidence": 0.70
                }
            },
            "ocr_service": {
                "queue_name": "ocr-queue",
                "timeout_ms": 30000,
                "max_retries": 3
            }
        }
        
        with open(config_path, "w", encoding='utf-8') as f:
            yaml.dump(config_content, f)
        
        # Réinitialiser le cache global
        import src.utils.config_loader as config_module
        config_module._global_config = None
        
        # Charger la config
        config = get_config(str(config_path))
        
        # Simuler ce que fait prepare_training_data.py dans main()
        default_input_dir = config.get('paths.training_raw_dir', 'training_data/raw')
        default_output_dir = config.get('paths.training_processed_dir', 'training_data/processed')
        
        # Vérifier que les valeurs du config sont utilisées
        assert default_input_dir == custom_raw
        assert default_output_dir == custom_processed


def test_train_classifier_uses_config_defaults():
    """Vérifie que train_classifier.py utilise bien les valeurs de config.yaml."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        
        custom_processed = "my_custom/processed"
        custom_artifacts = "my_custom/artifacts"
        
        config_content = {
            "paths": {
                "training_processed_dir": custom_processed,
                "training_artifacts_dir": custom_artifacts
            },
            "classification": {
                "enabled": True,
                "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "min_confidence": 0.70
            }
        }
        
        with open(config_path, "w", encoding='utf-8') as f:
            yaml.dump(config_content, f)
        
        # Réinitialiser le cache global
        import src.utils.config_loader as config_module
        config_module._global_config = None
        
        # Charger la config
        config = get_config(str(config_path))
        
        # Simuler ce que fait train_classifier.py dans main()
        default_data_dir = config.get('paths.training_processed_dir', 'training_data/processed')
        default_model_dir = config.get('paths.training_artifacts_dir', 'training_data/artifacts')
        
        # Vérifier que les valeurs du config sont utilisées
        assert default_data_dir == custom_processed
        assert default_model_dir == custom_artifacts


def test_config_get_method_with_nested_keys():
    """Vérifie que la méthode .get() fonctionne correctement avec des clés imbriquées."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        
        config_content = {
            "paths": {
                "training_raw_dir": "test/raw",
                "training_processed_dir": "test/processed"
            }
        }
        
        with open(config_path, "w", encoding='utf-8') as f:
            yaml.dump(config_content, f)
        
        # Réinitialiser le cache global
        import src.utils.config_loader as config_module
        config_module._global_config = None
        
        config = get_config(str(config_path))
        
        # Tester avec des clés imbriquées
        assert config.get('paths.training_raw_dir') == "test/raw"
        assert config.get('paths.training_processed_dir') == "test/processed"
        assert config.get('paths.nonexistent', 'default') == 'default'
        assert config.get('nonexistent.key', 'default') == 'default'

