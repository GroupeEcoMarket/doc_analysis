"""
Feature Extraction
Extrait les features des documents (checkboxes, OCR, etc.)
"""

from typing import Dict, List, Optional, Any
import numpy as np
import time
import os
from pathlib import Path

from src.utils.config_loader import Config
from src.pipeline.models import FeaturesOutput, OCRLine, CheckboxDetection
from src.utils.exceptions import FeatureExtractionError
from src.utils.file_handler import ensure_dir, get_files, get_output_path
from src.utils.storage import get_storage
from src.utils.ocr_client import perform_ocr_task
from dramatiq.results import ResultTimeout


class FeatureExtractor:
    """
    Extrait les features des documents normalisés
    - Détection de checkboxes
    - OCR (reconnaissance de texte)
    - Autres features à définir
    """
    
    def __init__(self, app_config: Config) -> None:
        """
        Initialise l'extracteur de features
        
        Args:
            app_config: Configuration de l'application (injectée via DI, obligatoire)
        """
        # Extraire la section features de la config injectée
        self.config = app_config.get('features', default={})
        
        # Vérifier si l'OCR est activé (configuration de filtrage post-traitement)
        # self.config est maintenant un dict Python normal, utiliser la syntaxe dict.get()
        ocr_config = self.config.get('ocr_filtering', {})
        self.ocr_enabled = ocr_config.get('enabled', False)
        
        # Configuration du microservice OCR
        ocr_service_config = app_config.get('ocr_service', default={})
        # ocr_service_config est un dict Python normal
        self.ocr_timeout_ms = ocr_service_config.get('timeout_ms', 30000)  # 30 secondes par défaut
        
        # Initialiser le stockage pour les fichiers temporaires
        self.storage = get_storage()
    
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
        try:
            ocr_lines = self.extract_ocr(img)
        except FeatureExtractionError as e:
            # Si l'OCR échoue, retourner un résultat d'erreur
            return FeaturesOutput(
                status='error',
                input_path=input_path,
                output_path=output_path,
                processing_time=time.time() - start_time,
                error=f"Échec de l'extraction OCR: {str(e)}"
            )
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
        
        Cette méthode utilise maintenant le microservice OCR isolé au lieu d'appeler
        PaddleOCR directement. Elle :
        1. Sauvegarde l'image temporairement
        2. Envoie une tâche au microservice OCR via Dramatiq
        3. Attend le résultat de manière bloquante
        4. Nettoie le fichier temporaire
        5. Retourne le résultat dans le format attendu
        
        Args:
            image: Image sous forme de tableau NumPy (format OpenCV BGR).
        
        Returns:
            Une liste de dictionnaires, chaque dictionnaire représentant une ligne de texte.
            Chaque dictionnaire contient: text, confidence, bounding_box
        
        Raises:
            FeatureExtractionError: Si l'OCR échoue ou si le microservice n'est pas disponible
        """
        if not self.ocr_enabled:
            # OCR non activé dans la config
            return []
        
        image_uri = None
        try:
            # 1. Sauvegarder l'image temporairement
            # Utiliser un page_index unique pour éviter les collisions
            import uuid
            unique_id = uuid.uuid4().hex[:8]
            image_uri = self.storage.save_image(image, page_index=0, task_id=f"ocr_{unique_id}")
            
            # 2. Envoyer la tâche au microservice OCR
            message = perform_ocr_task.send(image_uri, page_index=0)
            
            # 3. Attendre le résultat de manière bloquante
            try:
                result = message.get_result(block=True, timeout=self.ocr_timeout_ms)
            except ResultTimeout:
                raise FeatureExtractionError(
                    f"Timeout lors de l'attente du résultat OCR (timeout: {self.ocr_timeout_ms}ms)"
                )
            except Exception as e:
                raise FeatureExtractionError(
                    f"Erreur lors de la récupération du résultat OCR: {str(e)}"
                ) from e
            
            # 4. Vérifier le statut du résultat
            if result.get('status') == 'error':
                error_msg = result.get('error', 'Erreur inconnue lors du traitement OCR')
                raise FeatureExtractionError(f"Le microservice OCR a retourné une erreur: {error_msg}")
            
            # 5. Extraire les lignes OCR du résultat
            ocr_lines = result.get('ocr_lines', [])
            
            # 6. Filtrer selon le seuil de confiance minimum (filtre post-traitement)
            ocr_config = self.config.get('ocr_filtering', {})
            min_confidence = ocr_config.get('min_confidence', 0.70)
            
            filtered_lines = [
                line for line in ocr_lines
                if line.get('confidence', 0.0) >= min_confidence
            ]
            
            # 7. Convertir le format bounding_box de quadrilatère vers rectangle
            # Le microservice retourne bounding_box comme liste de tuples [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
            result_lines = []
            for line in filtered_lines:
                line_dict = line.copy()
                
                # Convertir bounding_box de quadrilatère vers rectangle [x_min, y_min, x_max, y_max]
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
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning(f"Could not convert bounding box: {bbox}, error: {e}")
                        line_dict['bounding_box'] = [0.0, 0.0, 0.0, 0.0]
                
                result_lines.append(line_dict)
            
            return result_lines
            
        except FeatureExtractionError:
            # Re-raise les erreurs FeatureExtractionError
            raise
        except Exception as e:
            # Transformer toute autre exception en FeatureExtractionError
            raise FeatureExtractionError(f"Erreur lors de l'extraction OCR: {str(e)}") from e
        finally:
            # 8. Nettoyer le fichier temporaire
            if image_uri is not None:
                try:
                    self.storage.delete_file(image_uri)
                except Exception as e:
                    # Logger l'erreur mais ne pas faire échouer la méthode
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Impossible de supprimer le fichier temporaire {image_uri}: {e}")

