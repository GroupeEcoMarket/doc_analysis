# src/utils/ocr_engines.py

import os
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from paddleocr import PaddleOCR


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
        
        print(f"Initialisation de PaddleOCR avec les paramètres : {init_kwargs}")
        self.ocr_engine = PaddleOCR(**init_kwargs)

    def recognize(self, image: np.ndarray) -> List[OCRLine]:
        """
        Prend une image (numpy array) et retourne une liste de lignes de texte structurées.
        """
        result = self.ocr_engine.ocr(image)

        if not result or not result[0]:
            return []

        lines = []
        
        # --- CORRECTION ULTIME : Parser la structure du dictionnaire ---
        # result[0] est le dictionnaire contenant toutes les informations
        ocr_data = result[0]
        
        # Extraire les listes de données du dictionnaire
        # 'dt_polys' (detection polygons) ou 'rec_polys' sont les boîtes
        boxes = ocr_data.get('dt_polys', ocr_data.get('rec_polys', []))
        texts = ocr_data.get('rec_texts', [])
        scores = ocr_data.get('rec_scores', [])

        # S'assurer que nous avons autant de textes que de boîtes et de scores
        if not (len(boxes) == len(texts) == len(scores)):
            print(f"[DEBUG] WARNING: Mismatch in lengths - boxes: {len(boxes)}, texts: {len(texts)}, scores: {len(scores)}")
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
                print(f"[DEBUG] ERROR: Failed to process line #{i}. Error: {e}")
                continue

        return lines