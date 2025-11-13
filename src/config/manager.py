"""
Configuration Manager - Centralized configuration management

This module provides a unified ConfigManager that:
1. Loads .env (environment variables) - highest priority
2. Loads config.yaml (business logic parameters) - default values
3. Merges both (env > yaml)
4. Provides unified access to all configuration

Architecture:
- .env: Environment, infrastructure, deployment (machine-specific)
- config.yaml: Business logic, ML parameters, thresholds (versioned, reproducible)
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class ConfigManager:
    """
    Centralized configuration manager that merges .env and config.yaml.
    
    Priority: Environment variables (.env) > config.yaml > defaults
    """
    
    def __init__(self, yaml_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            yaml_path: Path to config.yaml (default: config.yaml at project root)
        """
        # 1️⃣ Load environment variables
        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        else:
            load_dotenv()  # Try loading from current directory
        
        self.env = dict(os.environ)
        
        # 2️⃣ Load YAML functional configuration
        if yaml_path is None:
            project_root = Path(__file__).parent.parent.parent
            yaml_path = project_root / "config.yaml"
        
        self.yaml_path = Path(yaml_path)
        if not self.yaml_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.yaml_path}\n"
                f"Please create a config.yaml file at the project root."
            )
        
        with open(self.yaml_path, "r", encoding="utf-8") as f:
            self.yaml = yaml.safe_load(f) or {}
        
        # 3️⃣ Merge (env > yaml)
        self.config = self._merge_config()
    
    def _merge_config(self) -> Dict[str, Any]:
        """
        Merge environment variables and YAML configuration.
        
        Environment variables have priority over YAML values.
        """
        merged = self.yaml.copy()
        
        # Inject environment variables into config structure
        merged["env"] = {
            "api": {
                "host": self.env.get("API_HOST", "0.0.0.0"),
                "port": int(self.env.get("API_PORT", "8000")),
                "debug": self.env.get("API_DEBUG", "False").lower() == "true",
            },
            "redis": {
                "host": self.env.get("REDIS_HOST", "localhost"),
                "port": int(self.env.get("REDIS_PORT", "6379")),
            },
            "paths": {
                "input": self.env.get("INPUT_DIR", "data/input"),
                "output": self.env.get("OUTPUT_DIR", "data/output"),
                "processed": self.env.get("PROCESSED_DIR", "data/processed"),
                "temp": self.env.get("TEMP_STORAGE_DIR", "data/temp_storage"),
                "models": self.env.get("MODEL_PATH", "models/"),
                "classification_model": self.env.get("CLASSIFICATION_MODEL_PATH", "training_data/artifacts/document_classifier.joblib"),
            },
            "training": {
                "raw": self.env.get("TRAINING_RAW_DIR", "training_data/raw"),
                "processed": self.env.get("TRAINING_PROCESSED_DIR", "training_data/processed"),
                "artifacts": self.env.get("TRAINING_ARTIFACTS_DIR", "training_data/artifacts"),
            },
            "workers": {
                "dramatiq_processes": int(self.env.get("DRAMATIQ_PROCESSES", "4")),
                "dramatiq_threads": int(self.env.get("DRAMATIQ_THREADS", "1")),
                "ocr_processes": int(self.env.get("OCR_WORKER_PROCESSES", "2")),
                "ocr_threads": int(self.env.get("OCR_WORKER_THREADS", "1")),
            },
            "logging": {
                "level": self.env.get("LOG_LEVEL", "INFO"),
                "file": self.env.get("LOG_FILE", "logs/app.log"),
            },
            "storage": {
                "backend": self.env.get("STORAGE_BACKEND", "local"),
            },
            "models": {
                "embedding": self.env.get("CLASSIFICATION_EMBEDDING_MODEL", "antoinelouis/french-me5-base"),
            }
        }
        
        return merged
    
    def get(self, *keys, default: Any = None) -> Any:
        """
        Get configuration value by nested keys.
        
        Usage:
            config.get('geometry', 'deskew', 'min_angle')
            config.get('env', 'api', 'host')
            config.get('classification', 'min_confidence')
        
        Args:
            *keys: Nested keys to access (e.g., 'geometry', 'deskew', 'min_angle')
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        """
        node = self.config
        
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        
        return node
    
    def reload(self) -> None:
        """Reload configuration from files."""
        # Reload .env
        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=True)
        else:
            load_dotenv(override=True)
        
        self.env = dict(os.environ)
        
        # Reload YAML
        with open(self.yaml_path, "r", encoding="utf-8") as f:
            self.yaml = yaml.safe_load(f) or {}
        
        # Re-merge
        self.config = self._merge_config()


# Global singleton instance
_global_config_manager: Optional[ConfigManager] = None


def get_config_manager(yaml_path: Optional[str] = None) -> ConfigManager:
    """
    Get the global ConfigManager instance (singleton).
    
    Args:
        yaml_path: Path to config.yaml (used only on first call)
    
    Returns:
        ConfigManager instance
    """
    global _global_config_manager
    
    if _global_config_manager is None:
        _global_config_manager = ConfigManager(yaml_path)
    
    return _global_config_manager


# Convenience alias for backward compatibility
get_config = get_config_manager

