# src/utils/bootstrap.py

import os
import yaml
from pathlib import Path


def configure_paddle_environment():
    """
    Lit la configuration du projet et configure l'environnement pour PaddleOCR.
    
    Cette fonction doit être appelée AU TOUT DÉBUT du processus, avant
    toute importation de 'paddle' ou 'paddleocr'.
    
    Elle désactive l'optimisation MKL-DNN si 'enable_mkldnn' est à 'false'
    dans config.yaml, ce qui prévient les crashs sur certains CPU.
    """
    try:
        # Trouver le fichier config.yaml à la racine du projet
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config.yaml"

        if not config_path.exists():
            print("AVERTISSEMENT [bootstrap]: config.yaml non trouvé. MKL-DNN pourrait être activé par défaut.")
            return

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Vérifier la configuration et désactiver MKL-DNN si nécessaire
        enable_mkldnn = config.get('features', {}).get('ocr', {}).get('enable_mkldnn', False)

        if not enable_mkldnn:
            # Définir la variable d'environnement pour désactiver MKL-DNN
            os.environ["FLAGS_use_mkldnn"] = "0"
            print("INFO [bootstrap]: MKL-DNN a été désactivé via la variable d'environnement FLAGS_use_mkldnn.")
        else:
            print("INFO [bootstrap]: MKL-DNN est activé selon la configuration.")

    except Exception as e:
        print(f"ERREUR [bootstrap]: Impossible de configurer l'environnement Paddle. Erreur : {e}")
        # Par sécurité, on désactive MKL-DNN en cas d'erreur de lecture de la config
        os.environ["FLAGS_use_mkldnn"] = "0"
        print("INFO [bootstrap]: MKL-DNN a été désactivé par mesure de sécurité.")

