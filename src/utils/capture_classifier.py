"""
Capture Method Classifier
Détermine si une image provient d'un scan ou d'une photo
"""

import cv2
import numpy as np
from typing import Dict, Optional, Any


class CaptureClassifier:
    """
    Classifie une image comme 'SCAN' ou 'PHOTO' en se basant sur le pourcentage
    de pixels presque blancs.
    """
    
    def __init__(self, 
                 white_level_threshold: int = 245,
                 white_percentage_threshold: float = 0.70,
                 enabled: bool = True):
        """
        Initialise le classificateur
        
        Args:
            white_level_threshold: Valeur de pixel (0-255) au-dessus de laquelle
                                  un pixel est considéré comme blanc
            white_percentage_threshold: Pourcentage (0.0-1.0) de pixels blancs
                                       nécessaire pour classer comme scan
            enabled: Active/désactive la classification (si False, toujours PHOTO)
        """
        self.white_level_threshold = white_level_threshold
        self.white_percentage_threshold = white_percentage_threshold
        self.enabled = enabled
    
    def classify(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Classifie une image comme 'SCAN' ou 'PHOTO'
        
        Args:
            image: Image en format OpenCV (BGR ou grayscale)
            
        Returns:
            dict: {
                'type': 'SCAN' ou 'PHOTO',
                'white_percentage': pourcentage de pixels blancs,
                'confidence': confiance de la classification (0.0-1.0),
                'enabled': si la classification était activée
            }
        """
        # Si désactivé, toujours retourner PHOTO (pipeline complet)
        if not self.enabled:
            return {
                'type': 'PHOTO',
                'white_percentage': 0.0,
                'confidence': 0.0,
                'enabled': False,
                'reason': 'Classification désactivée'
            }
        
        try:
            # 1. Convertir en niveaux de gris si nécessaire
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image
            
            # 2. Compter le nombre de pixels "presque blancs"
            num_white_pixels = np.sum(gray_image > self.white_level_threshold)
            
            # 3. Calculer le nombre total de pixels
            total_pixels = gray_image.size
            
            # 4. Calculer le pourcentage de blanc
            white_percentage = num_white_pixels / total_pixels
            
            # 5. Calculer la confiance (distance au seuil)
            # Plus on est loin du seuil, plus on est confiant
            distance_from_threshold = abs(white_percentage - self.white_percentage_threshold)
            confidence = min(1.0, distance_from_threshold * 2)  # Normaliser
            
            # 6. Prendre la décision
            if white_percentage > self.white_percentage_threshold:
                capture_type = "SCAN"
                reason = f"Blanc: {white_percentage:.1%} > seuil {self.white_percentage_threshold:.1%}"
            else:
                capture_type = "PHOTO"
                reason = f"Blanc: {white_percentage:.1%} <= seuil {self.white_percentage_threshold:.1%}"
            
            return {
                'type': capture_type,
                'white_percentage': float(white_percentage),
                'confidence': float(confidence),
                'enabled': True,
                'reason': reason,
                'white_level_threshold': self.white_level_threshold,
                'white_percentage_threshold': self.white_percentage_threshold
            }
            
        except Exception as e:
            # En cas d'erreur, retourner PHOTO (pipeline complet par sécurité)
            return {
                'type': 'PHOTO',
                'white_percentage': 0.0,
                'confidence': 0.0,
                'enabled': True,
                'reason': f'Erreur: {str(e)}',
                'error': str(e)
            }
    
    def classify_from_path(self, image_path: str) -> Dict[str, Any]:
        """
        Classifie une image à partir de son chemin
        
        Args:
            image_path: Chemin vers le fichier image
            
        Returns:
            dict: Résultat de la classification
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Impossible de charger l'image: {image_path}")
            
            return self.classify(image)
            
        except Exception as e:
            return {
                'type': 'PHOTO',
                'white_percentage': 0.0,
                'confidence': 0.0,
                'enabled': self.enabled,
                'reason': f'Erreur de chargement: {str(e)}',
                'error': str(e)
            }


def classify_capture_method(
    image_path: str,
    white_level_threshold: int = 245,
    white_percentage_threshold: float = 0.70
) -> str:
    """
    Fonction utilitaire simple pour classifier une image
    
    Args:
        image_path: Chemin vers l'image
        white_level_threshold: Seuil de niveau de blanc (0-255)
        white_percentage_threshold: Seuil de pourcentage de blanc (0.0-1.0)
        
    Returns:
        str: 'SCAN' ou 'PHOTO'
    """
    classifier = CaptureClassifier(
        white_level_threshold=white_level_threshold,
        white_percentage_threshold=white_percentage_threshold,
        enabled=True
    )
    result = classifier.classify_from_path(image_path)
    return result['type']


if __name__ == "__main__":
    # Test du classificateur
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        # Test avec paramètres par défaut
        classifier = CaptureClassifier()
        result = classifier.classify_from_path(image_path)
        
        print("=" * 60)
        print("Classification de Type de Capture")
        print("=" * 60)
        print(f"Image: {image_path}")
        print(f"Type détecté: {result['type']}")
        print(f"Pourcentage de blanc: {result['white_percentage']:.2%}")
        print(f"Confiance: {result['confidence']:.2%}")
        print(f"Raison: {result['reason']}")
        print("=" * 60)
        
        if result['type'] == 'SCAN':
            print("-> Pipeline SCAN: Rotation -> OCR (pas de crop)")
        else:
            print("-> Pipeline PHOTO: Crop/Perspective -> Rotation -> OCR")
    else:
        print("Usage: python -m src.utils.capture_classifier <chemin_image>")

