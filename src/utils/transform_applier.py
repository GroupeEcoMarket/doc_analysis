"""
Transform Applier
Permet de réappliquer les transformations sauvegardées
"""

import cv2
import numpy as np
from typing import Optional, TYPE_CHECKING
from src.utils.transform_handler import TransformSequence, load_transforms

if TYPE_CHECKING:
    from src.utils.transform_handler import Transform


def apply_transform_sequence(image: np.ndarray, transform_sequence: TransformSequence) -> np.ndarray:
    """
    Applique une séquence de transformations à une image
    
    Args:
        image: Image d'entrée
        transform_sequence: Séquence de transformations à appliquer
        
    Returns:
        Image transformée
    """
    result = image.copy()
    
    for transform in transform_sequence.transforms:
        result = apply_single_transform(result, transform)
    
    return result


def apply_single_transform(image: np.ndarray, transform: 'Transform') -> np.ndarray:  # type: ignore
    """
    Applique une transformation unique à une image
    
    Args:
        image: Image d'entrée
        transform: Objet Transform à appliquer
        
    Returns:
        Image transformée
    """
    params = transform.params
    transform_type = transform.transform_type
    
    if transform_type == 'crop':
        # Appliquer la transformation de perspective (crop)
        if 'transform_matrix' in params and 'output_size' in params:
            M = np.array(params['transform_matrix'], dtype=np.float32)
            output_size = tuple(params['output_size'])
            result = cv2.warpPerspective(image, M, output_size)
            return result
    
    elif transform_type == 'deskew':
        # Appliquer la transformation affine (deskew)
        if 'transform_matrix' in params and 'output_size' in params:
            M = np.array(params['transform_matrix'], dtype=np.float32)
            output_size = tuple(params['output_size'])
            result = cv2.warpAffine(image, M, output_size,
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(255, 255, 255))
            return result
    
    elif transform_type == 'rotation':
        # Appliquer la rotation
        angle = params.get('angle', 0)
        rotation_type = params.get('rotation_type', 'standard')
        
        if rotation_type == 'standard':
            if angle == 90:
                return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                return cv2.rotate(image, cv2.ROTATE_180)
            elif angle == 270:
                return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            # Rotation arbitraire
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
    
    # Si la transformation n'est pas reconnue, retourner l'image originale
    return image


def apply_transforms_from_file(image: np.ndarray, output_path: str) -> Optional[np.ndarray]:
    """
    Charge les transformations depuis un fichier et les applique à une image
    
    Args:
        image: Image d'entrée
        output_path: Chemin du fichier de sortie (pour trouver le fichier .transform.json)
        
    Returns:
        Image transformée ou None si le fichier de transformation n'existe pas
    """
    transform_sequence = load_transforms(output_path)
    if transform_sequence is None:
        return None
    
    return apply_transform_sequence(image, transform_sequence)

