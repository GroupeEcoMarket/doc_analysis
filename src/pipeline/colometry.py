"""
Colometry Normalization
Normalise les documents par colométrie (analyse des colonnes)
"""

from typing import Dict, List, Optional, Any, Union
import time

from src.utils.config_loader import get_config, Config
from src.pipeline.models import ColometryOutput


class ColometryNormalizer:
    """
    Normalise les documents en analysant et standardisant la structure colométrique
    """
    
    def __init__(
        self,
        app_config: Optional[Config] = None,
        config: Optional[Union[Dict[str, Any], str]] = None
    ) -> None:
        """
        Initialise le normaliseur colométrique
        
        Args:
            app_config: Configuration de l'application (injectée via DI).
                       Si None, charge la config depuis config.yaml (pour compatibilité).
            config: Configuration optionnelle (dict ou chemin de fichier).
                   Si app_config est fourni, config est ignoré.
                   Si les deux sont None, charge depuis config.yaml (pour compatibilité).
        """
        # Priorité: app_config (injection de dépendances) > config (dict/str) > get_config() (fallback)
        if app_config is not None:
            # Extraire la section colometry de la config injectée
            self.config: Dict[str, Any] = app_config.get('colometry', {})
        elif config is not None:
            # Si config est un chemin de fichier (str), charger la config
            if isinstance(config, str):
                config_obj = get_config(config)
                self.config = config_obj.get('colometry', {})
            else:
                # config est un dict
                self.config = config
        else:
            # Fallback final: charger depuis config.yaml
            config_obj = get_config()
            self.config = config_obj.get('colometry', {})
    
    def process(self, input_path: str, output_path: str) -> ColometryOutput:
        """
        Traite un document pour normaliser sa colométrie
        
        Args:
            input_path: Chemin vers le document d'entrée
            output_path: Chemin de sortie pour le document normalisé
            
        Returns:
            ColometryOutput: Résultats du traitement structurés
        """
        start_time = time.time()
        
        # TODO: Implémenter la normalisation colométrique
        # Pour l'instant, retourner un résultat vide mais structuré
        processing_time = time.time() - start_time
        
        return ColometryOutput(
            status='success',
            input_path=input_path,
            output_path=output_path,
            processing_time=processing_time
        )
    
    def process_batch(self, input_dir: str, output_dir: str) -> List[ColometryOutput]:
        """
        Traite un lot de documents
        
        Args:
            input_dir: Répertoire contenant les documents d'entrée
            output_dir: Répertoire de sortie
            
        Returns:
            List[ColometryOutput]: Liste des résultats pour chaque document
        """
        # TODO: Implémenter le traitement par lot
        # Pour l'instant, retourner une liste vide
        return []

