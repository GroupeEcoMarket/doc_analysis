"""
Feature Extraction
Extrait les features des documents (checkboxes, OCR, etc.)
"""

from typing import Dict, List, Optional, Any
import numpy as np
import time
import os
from pathlib import Path

from src.utils.ocr_engines import PaddleOCREngine
from src.utils.config_loader import get_config, Config
from src.pipeline.models import FeaturesOutput, OCRLine, CheckboxDetection
from src.utils.file_handler import ensure_dir, get_files, get_output_path


class FeatureExtractor:
    """
    Extrait les features des documents normalisés
    - Détection de checkboxes
    - OCR (reconnaissance de texte)
    - Autres features à définir
    """
    
    def __init__(
        self, 
        app_config: Optional[Config] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialise l'extracteur de features
        
        Args:
            app_config: Configuration de l'application (injectée via DI).
                       Si None, charge la config depuis config.yaml (pour compatibilité).
            config: Configuration optionnelle (dict). 
                   Si app_config est fourni, config est ignoré.
                   Si les deux sont None, charge depuis config.yaml (pour compatibilité).
        """
        # Priorité: app_config (injection de dépendances) > config (dict) > get_config() (fallback)
        if app_config is not None:
            # Extraire la section features de la config injectée
            self.config = app_config.get('features', {})
        elif config is not None:
            # Fallback pour compatibilité avec l'ancien code
            self.config = config
        else:
            # Fallback final: charger depuis config.yaml
            config_obj = get_config()
            self.config = config_obj.get('features', {})
        
        # Initialiser le moteur OCR si activé
        self.ocr_engine: Optional[PaddleOCREngine] = None
        ocr_config = self.config.get('ocr', {})
        if ocr_config.get('enabled', False):
            ocr_lang = ocr_config.get('default_language', 'fr')
            use_gpu = ocr_config.get('use_gpu', False)
            runtime_options_cfg = ocr_config.get('runtime_options')
            runtime_options = (
                runtime_options_cfg
                if isinstance(runtime_options_cfg, dict)
                else None
            )
            
            self.ocr_engine = PaddleOCREngine(
                language=ocr_lang,
                use_gpu=use_gpu,
                runtime_options=runtime_options
            )
    
    def process(self, input_path: str, output_path: str) -> FeaturesOutput:
        """
        Traite un document pour extraire ses features
        
        Args:
            input_path: Chemin vers le document d'entrée
            output_path: Chemin de sortie pour les features extraites
            
        Returns:
            FeaturesOutput: Features extraites structurées (checkboxes, OCR, etc.)
        """
        start_time = time.time()
        
        # Charger l'image
        import cv2
        img = cv2.imread(input_path)
        if img is None:
            return FeaturesOutput(
                status='error',
                input_path=input_path,
                output_path=output_path,
                processing_time=time.time() - start_time,
                error=f"Impossible de charger l'image: {input_path}"
            )
        
        # Extraire les features
        ocr_lines = self.extract_ocr(img)
        checkboxes = self.extract_checkboxes(input_path)
        
        # Avertir si aucune ligne OCR n'a été détectée
        if not ocr_lines:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Aucune ligne OCR détectée pour {input_path}")
        
        # Convertir les dictionnaires OCR en objets OCRLine
        ocr_line_objects = [
            OCRLine(
                text=line.get('text', ''),
                confidence=line.get('confidence', 0.0),
                bounding_box=line.get('bounding_box', [0.0, 0.0, 0.0, 0.0])
            ) for line in ocr_lines
        ]
        
        # Convertir les dictionnaires checkboxes en objets CheckboxDetection
        checkbox_objects = [
            CheckboxDetection(
                position=box.get('position', [0.0, 0.0, 0.0, 0.0]),
                checked=box.get('checked', False),
                confidence=box.get('confidence', 0.0)
            ) for box in checkboxes
        ]
        
        # Sauvegarder les features dans un fichier JSON
        ensure_dir(os.path.dirname(output_path))
        import json
        features_dict = {
            'ocr_lines': [line.model_dump() for line in ocr_line_objects],
            'checkboxes': [box.model_dump() for box in checkbox_objects]
        }
        with open(output_path, 'w') as f:
            json.dump(features_dict, f, indent=2, default=str)
        
        processing_time = time.time() - start_time
        
        return FeaturesOutput(
            status='success',
            input_path=input_path,
            output_path=output_path,
            processing_time=processing_time,
            ocr_lines=ocr_line_objects,
            checkboxes=checkbox_objects
        )
    
    def process_batch(self, input_dir: str, output_dir: str) -> List[FeaturesOutput]:
        """
        Traite un lot de documents
        
        Args:
            input_dir: Répertoire contenant les documents d'entrée
            output_dir: Répertoire de sortie
            
        Returns:
            List[FeaturesOutput]: Liste des résultats pour chaque document
        """
        ensure_dir(output_dir)
        files = get_files(input_dir, extensions=['.png', '.jpg', '.jpeg', '.tiff', '.bmp'])
        results = []
        
        for file_path in files:
            output_path = get_output_path(file_path, output_dir).replace(Path(file_path).suffix, '.json')
            try:
                result = self.process(file_path, output_path)
                results.append(result)
            except Exception as e:
                # Créer un FeaturesOutput avec status='error'
                error_output = FeaturesOutput(
                    status='error',
                    input_path=file_path,
                    output_path=output_path,
                    processing_time=0.0,
                    error=str(e)
                )
                results.append(error_output)
        
        return results
    
    def extract_checkboxes(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Détecte et extrait les checkboxes du document
        
        Args:
            image_path: Chemin vers l'image du document
            
        Returns:
            list: Liste des checkboxes détectées avec leur état
        """
        # TODO: Implémenter la détection de checkboxes
        return []  # Retourner une liste vide au lieu de None
    
    def extract_ocr(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extrait le texte du document via OCR, ligne par ligne.
        
        Args:
            image: Image sous forme de tableau NumPy (format OpenCV BGR).
        
        Returns:
            Une liste de dictionnaires, chaque dictionnaire représentant une ligne de texte.
            Chaque dictionnaire contient: text, confidence, bounding_box
        """
        if self.ocr_engine is None:
            # OCR non activé dans la config
            return []
        
        # Extraire les lignes de texte avec redimensionnement automatique si nécessaire
        ocr_config = self.config.get('ocr', {})
        max_dimension = ocr_config.get('max_image_dimension', 3500)
        lines = self.ocr_engine.recognize(image, max_dimension=max_dimension)
        
        # Filtrer selon le seuil de confiance minimum
        min_confidence = ocr_config.get('min_confidence', 0.70)
        
        filtered_lines = [
            line for line in lines 
            if line.confidence >= min_confidence
        ]
        
        # Convertir les objets Pydantic en dictionnaires pour la sortie
        # et convertir le format bounding box de quadrilatère vers rectangle
        result = []
        for line in filtered_lines:
            line_dict = line.model_dump()
            # Convertir bounding_box de [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] vers [x_min, y_min, x_max, y_max]
            if 'bounding_box' in line_dict and line_dict['bounding_box'] is not None:
                bbox = line_dict['bounding_box']
                try:
                    # Si c'est déjà une liste de 4 nombres, on la garde telle quelle
                    if len(bbox) == 4 and isinstance(bbox[0], (int, float)):
                        pass  # Déjà au bon format
                    else:
                        # Convertir de liste de tuples vers rectangle
                        all_x = [point[0] for point in bbox]
                        all_y = [point[1] for point in bbox]
                        line_dict['bounding_box'] = [
                            float(min(all_x)), 
                            float(min(all_y)), 
                            float(max(all_x)), 
                            float(max(all_y))
                        ]
                except (TypeError, IndexError, KeyError) as e:
                    # En cas d'erreur, utiliser une bbox par défaut
                    print(f"[WARNING] Could not convert bounding box: {bbox}, error: {e}")
                    line_dict['bounding_box'] = [0.0, 0.0, 0.0, 0.0]
            result.append(line_dict)
        
        return result

