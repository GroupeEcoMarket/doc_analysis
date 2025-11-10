"""
Fonctions pour l'amélioration automatique de la qualité des images.
"""
import cv2
import numpy as np

def enhance_contrast_clahe(image: np.ndarray) -> np.ndarray:
    """
    Améliore le contraste d'une image en couleur ou en niveaux de gris en utilisant CLAHE.
    
    Args:
        image: Image en format OpenCV (BGR ou grayscale)
        
    Returns:
        Image améliorée en format BGR
    """
    if len(image.shape) < 3 or image.shape[2] == 1:
        gray = image.copy()
        if len(gray.shape) == 3:
             gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        return cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel_enhanced = clahe.apply(l_channel)

    lab_image_enhanced = cv2.merge((l_channel_enhanced, a_channel, b_channel))
    enhanced_image = cv2.cvtColor(lab_image_enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced_image

