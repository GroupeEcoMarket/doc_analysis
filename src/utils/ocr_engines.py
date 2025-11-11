# src/utils/ocr_engines.py

import os
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from paddleocr import PaddleOCR
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OCRLine(BaseModel):
    """Modèle pour représenter une ligne de texte détectée par OCR"""
    text: str = Field(..., description="Le texte reconnu de la ligne")
    confidence: float = Field(..., description="Le score de confiance de la reconnaissance")
    bounding_box: List[Tuple[int, int]] = Field(
        ...,
        description="Les 4 points de la boîte englobante [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]"
    )


class PaddleOCREngine:
    """Wrapper pour le moteur PaddleOCR"""

    def __init__(
        self,
        language: str = 'fr',
        *,
        use_gpu: bool = False,
        runtime_options: Optional[Dict[str, Any]] = None
    ):
        """
        Initialise le moteur PaddleOCR en utilisant la nouvelle API de configuration.
        """
        device_str = 'gpu' if use_gpu else 'cpu'

        init_kwargs: Dict[str, Any] = {
            "lang": language,
            "use_angle_cls": True,
            "device": device_str
        }

        if runtime_options:
            init_kwargs.update(runtime_options)
        
        logger.info(f"Initialisation de PaddleOCR avec les paramètres : {init_kwargs}")
        self.ocr_engine = PaddleOCR(**init_kwargs)

    def recognize(self, image: np.ndarray, max_dimension: int = None) -> List[OCRLine]:
        """
        Prend une image (numpy array) et retourne une liste de lignes de texte structurées.
        
        Args:
            image: Image numpy array
            max_dimension: Dimension maximale (optionnelle, par défaut lit depuis la config)
        """
        # Pré-traitement : Redimensionner si l'image est trop grande
        # PaddleOCR a une limite max_side_len de 4000 pixels par défaut
        if max_dimension is None:
            max_dimension = 3500  # Valeur par défaut si non spécifiée
        
        if image is not None and len(image.shape) >= 2:
            height, width = image.shape[:2]
            max_side = max(height, width)
            
            if max_side > max_dimension:
                # Calculer le ratio de redimensionnement
                scale = max_dimension / max_side
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                logger.info(
                    f"Image trop grande ({width}x{height}), redimensionnement à "
                    f"({new_width}x{new_height}) pour éviter un crash PaddleOCR"
                )
                
                import cv2
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        try:
            result = self.ocr_engine.ocr(image)
        except Exception as e:
            # PaddleOCR peut planter sur certaines images (corrompues, format invalide, etc.)
            logger.error(f"PaddleOCR crashed during inference: {type(e).__name__}: {e}")
            logger.info(f"Image shape: {image.shape if image is not None else 'None'}")
            # Retourner une liste vide au lieu de faire planter tout le script
            return []

        if not result or not result[0]:
            return []

        lines = []
        
        # --- Vérifier le type de result[0] ---
        ocr_data = result[0]
        
        # Debug: afficher le type et un échantillon
        logger.debug(f"type(result): {type(result)}, len: {len(result)}")
        logger.debug(f"type(result[0]): {type(ocr_data)}")
        if isinstance(ocr_data, list) and len(ocr_data) > 0:
            logger.debug(f"Premier élément: {ocr_data[0] if ocr_data else 'vide'}")
        elif isinstance(ocr_data, dict):
            logger.debug(f"Clés du dict: {ocr_data.keys()}")
        
        # Vérifier si c'est un objet OCRResult (avec attributs) ou un dict/list classique
        if hasattr(ocr_data, 'dt_polys') or hasattr(ocr_data, 'rec_polys'):
            # Format OCRResult de PaddleX (objet avec attributs)
            try:
                boxes = getattr(ocr_data, 'dt_polys', None) or getattr(ocr_data, 'rec_polys', None)
                texts = getattr(ocr_data, 'rec_texts', None)
                scores = getattr(ocr_data, 'rec_scores', None)
                
                logger.debug(f"boxes type: {type(boxes)}, value: {boxes}")
                logger.debug(f"texts type: {type(texts)}, value: {texts}")
                logger.debug(f"scores type: {type(scores)}, value: {scores}")
                
                # Vérifier que ce ne sont pas des None
                if boxes is None or texts is None or scores is None:
                    logger.warning("One or more attributes is None")
                    return []
                
                logger.debug(f"boxes len: {len(boxes)}, texts len: {len(texts)}, scores len: {len(scores)}")
                
                # S'assurer que nous avons autant de textes que de boîtes et de scores
                if not (len(boxes) == len(texts) == len(scores)):
                    logger.warning(f"Mismatch in lengths - boxes: {len(boxes)}, texts: {len(texts)}, scores: {len(scores)}")
                    return []
                
                # Combiner les informations pour chaque ligne
                for i in range(len(texts)):
                    try:
                        box_points = boxes[i]
                        text_content = texts[i]
                        confidence = scores[i]
                        
                        if box_points is None:
                            continue
                        
                        # La boîte est déjà un numpy array, on le convertit en liste de tuples
                        bounding_box = [tuple(map(int, point)) for point in box_points]
                        
                        lines.append(OCRLine(
                            text=text_content,
                            confidence=float(confidence),
                            bounding_box=bounding_box
                        ))
                    except Exception as e:
                        logger.error(f"Failed to process line #{i}: {e}", exc_info=True)
                        continue
            except Exception as e:
                logger.error(f"Failed to process OCRResult object: {e}", exc_info=True)
                return []
        
        # Si result[0] est une liste (ancien format), pas un dict
        elif isinstance(ocr_data, list):
            # Format: [[[box], (text, confidence)], ...]
            for item in ocr_data:
                try:
                    if item is None or len(item) != 2:
                        continue
                    box_points, text_info = item
                    
                    # Vérifier que box_points n'est pas None et est itérable
                    if box_points is None:
                        continue
                    
                    # Extraire text et confidence du tuple
                    if isinstance(text_info, (list, tuple)) and len(text_info) == 2:
                        text_content, confidence = text_info
                    else:
                        logger.warning(f"Unexpected text_info format: {text_info}")
                        continue
                    
                    # Vérifier que box_points est une liste/array itérable
                    if not hasattr(box_points, '__iter__'):
                        logger.warning(f"box_points is not iterable: {box_points}")
                        continue
                    
                    # Convertir les coordonnées en liste de tuples
                    bounding_box = []
                    for point in box_points:
                        if point is not None and hasattr(point, '__iter__'):
                            bounding_box.append(tuple(map(int, point)))
                    
                    if len(bounding_box) == 0:
                        logger.warning("No valid bounding box points")
                        continue
                    
                    lines.append(OCRLine(
                        text=text_content,
                        confidence=float(confidence),
                        bounding_box=bounding_box
                    ))
                except Exception as e:
                    logger.error(f"Failed to process line: {e}, item: {item}", exc_info=True)
                    continue
        elif isinstance(ocr_data, dict):
            # Nouveau format (dict)
            boxes = ocr_data.get('dt_polys', ocr_data.get('rec_polys', []))
            texts = ocr_data.get('rec_texts', [])
            scores = ocr_data.get('rec_scores', [])

            # S'assurer que nous avons autant de textes que de boîtes et de scores
            if not (len(boxes) == len(texts) == len(scores)):
                logger.warning(f"Mismatch in lengths - boxes: {len(boxes)}, texts: {len(texts)}, scores: {len(scores)}")
                return []

            # Combiner les informations pour chaque ligne
            for i in range(len(texts)):
                try:
                    box_points = boxes[i]
                    text_content = texts[i]
                    confidence = scores[i]

                    # La conversion des coordonnées reste la même
                    # La boîte est déjà un numpy array, on le convertit en liste de tuples
                    bounding_box = [tuple(map(int, point)) for point in box_points]

                    lines.append(OCRLine(
                        text=text_content,
                        confidence=float(confidence),
                        bounding_box=bounding_box
                    ))
                except Exception as e:
                    logger.error(f"Failed to process line #{i}: {e}", exc_info=True)
                    continue
        else:
            logger.error(f"Unknown format for result[0]: {type(ocr_data)}")

        return lines