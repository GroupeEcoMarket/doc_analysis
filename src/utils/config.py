"""
Configuration management
"""

import os
from dotenv import load_dotenv
from pathlib import Path


def get_config(config_file=None):
    """
    Charge la configuration depuis les variables d'environnement
    
    Args:
        config_file: Chemin vers un fichier .env optionnel
        
    Returns:
        dict: Configuration chargée
    """
    if config_file:
        load_dotenv(config_file)
    else:
        # Charge .env depuis la racine du projet
        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        else:
            load_dotenv()
    
    config = {
        "API_HOST": os.getenv("API_HOST", "0.0.0.0"),
        "API_PORT": os.getenv("API_PORT", "8000"),
        "API_DEBUG": os.getenv("API_DEBUG", "False").lower() == "true",
        "INPUT_DIR": os.getenv("INPUT_DIR", "data/input"),
        "OUTPUT_DIR": os.getenv("OUTPUT_DIR", "data/output"),
        "PROCESSED_DIR": os.getenv("PROCESSED_DIR", "data/processed"),
        "TESSERACT_CMD": os.getenv("TESSERACT_CMD", ""),
        "MODEL_PATH": os.getenv("MODEL_PATH", "models/"),
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
        "LOG_FILE": os.getenv("LOG_FILE", "logs/app.log"),
    }
    
    return config

