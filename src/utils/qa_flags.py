"""
QA Flags and Quality Metrics
Détecte les problèmes de qualité et génère des flags
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json
from src.utils.config_loader import get_config, GeometryConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QAFlags:
    """
    Flags de qualité pour un document traité
    """
    low_confidence_orientation: bool = False
    overcrop_risk: bool = False
    no_quad_detected: bool = False
    dewarp_applied: bool = False
    low_contrast_after_enhance: bool = False
    too_small_final: bool = False
    rotated: bool = False
    
    # Métadonnées supplémentaires
    orientation_confidence: float = 0.0
    orientation_angle: int = 0  # Angle détecté par ONNX (0, 90, 180, 270)
    deskew_angle: float = 0.0  # Angle appliqué pour le deskew
    deskew_confidence: float = 0.0  # Probabilité/confidence du deskew
    crop_margins: Dict[str, float] = None  # {'top': %, 'bottom': %, 'left': %, 'right': %}
    final_resolution: List[int] = None  # [width, height]
    processing_time: float = 0.0  # en secondes
    
    # Classification de capture
    capture_type: str = 'UNKNOWN'  # 'SCAN', 'PHOTO', ou 'UNKNOWN'
    capture_white_percentage: float = 0.0  # Pourcentage de blanc détecté
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        result = asdict(self)
        # Convertir les valeurs None
        if result['crop_margins'] is None:
            result['crop_margins'] = {}
        if result['final_resolution'] is None:
            result['final_resolution'] = []
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QAFlags':
        """Crée depuis un dictionnaire"""
        return cls(**data)


class QADetector:
    """
    Détecteur de problèmes de qualité
    """
    
    def __init__(self, geo_config: Optional[GeometryConfig] = None, config: Optional[Dict] = None):
        """
        Initialise le détecteur QA
        
        Args:
            geo_config: Configuration géométrique (source unique de vérité pour les seuils).
                       Si None, charge depuis config.yaml (pour compatibilité).
            config: Configuration optionnelle (dict) pour compatibilité avec l'ancien format.
                   Ignoré si geo_config est fourni.
        """
        # Priorité: geo_config (source unique de vérité) > config (dict) > get_config() (fallback)
        if geo_config is not None:
            # Utiliser GeometryConfig comme source unique de vérité
            self.orientation_confidence_threshold = geo_config.orientation_min_confidence
            self.overcrop_threshold = geo_config.crop_max_margin_ratio * 100  # Convertir en %
            self.min_resolution = [geo_config.quality_min_resolution_width, geo_config.quality_min_resolution_height]
            self.contrast_threshold = float(geo_config.quality_min_contrast)
        elif config is not None:
            # Compatibilité avec l'ancien format dict (pour les tests)
            self.orientation_confidence_threshold = config.get('orientation_confidence_threshold', 0.70)
            self.overcrop_threshold = config.get('overcrop_threshold', 8.0)
            self.min_resolution = config.get('min_resolution', [1200, 1600])
            self.contrast_threshold = config.get('contrast_threshold', 50.0)
        else:
            # Fallback: charger depuis config.yaml
            app_config = get_config()
            geo_config = app_config.geometry
            
            # Utiliser geometry.* comme source unique de vérité
            self.orientation_confidence_threshold = geo_config.orientation_min_confidence
            self.overcrop_threshold = geo_config.crop_max_margin_ratio * 100  # Convertir en %
            self.min_resolution = [geo_config.quality_min_resolution_width, geo_config.quality_min_resolution_height]
            self.contrast_threshold = float(geo_config.quality_min_contrast)
    
    def detect_flags(self, 
                     original_image: np.ndarray,
                     final_image: np.ndarray,
                     metadata: Dict[str, Any],
                     processing_time: float = 0.0) -> QAFlags:
        """
        Détecte tous les flags de qualité
        
        Args:
            original_image: Image originale
            final_image: Image finale après traitement
            metadata: Métadonnées du traitement
            processing_time: Temps de traitement en secondes
            
        Returns:
            QAFlags: Flags détectés
        """
        flags = QAFlags()
        flags.processing_time = processing_time
        
        # 1. low_confidence_orientation
        orientation_conf = metadata.get('orientation_confidence', 0.0)
        orientation_angle = metadata.get('angle', 0)
        flags.orientation_confidence = orientation_conf
        flags.orientation_angle = int(orientation_angle)
        if orientation_conf < self.orientation_confidence_threshold:
            flags.low_confidence_orientation = True
        
        # 2. overcrop_risk
        crop_metadata = metadata.get('crop_metadata', {})
        if crop_metadata.get('crop_applied', False):
            margins = self._calculate_crop_margins(original_image, final_image, crop_metadata)
            flags.crop_margins = margins
            if any(margin < self.overcrop_threshold for margin in margins.values()):
                flags.overcrop_risk = True
        
        # 3. no_quad_detected
        if crop_metadata.get('status') in ['no_detection', 'invalid_geometry', 'no_valid_detection']:
            flags.no_quad_detected = True
        
        # Note: Le statut 'rejected_overcrop' est géré dans intelligent_crop() qui rejette le crop
        # avant qu'il ne soit appliqué, donc overcrop_risk ne sera pas levé car crop_applied=False
        
        # 4. dewarp_applied (crop avec perspective = dewarp)
        if crop_metadata.get('crop_applied', False) and 'transform_matrix' in crop_metadata:
            flags.dewarp_applied = True

        # Informations sur le deskew
        deskew_metadata = metadata.get('deskew_metadata', {})
        flags.deskew_angle = float(deskew_metadata.get('angle', 0.0))
        flags.deskew_confidence = float(deskew_metadata.get('confidence', 0.0))
        
        # 5. low_contrast_after_enhance
        contrast = self._calculate_contrast(final_image)
        if contrast < self.contrast_threshold:
            flags.low_contrast_after_enhance = True
        
        # 6. too_small_final
        h, w = final_image.shape[:2]
        flags.final_resolution = [w, h]
        if w < self.min_resolution[0] or h < self.min_resolution[1]:
            flags.too_small_final = True
        
        # 7. rotated
        # Détecter la rotation de plusieurs façons :
        # 1. Via rotation_applied dans les métadonnées
        # 2. Via orientation_angle != 0 (une rotation a été détectée/appliquée)
        # 3. Via les transformations (transform_sequence)
        rotation_applied = metadata.get('rotation_applied', False)
        orientation_angle = float(metadata.get('angle', 0))
        
        # Vérifier aussi dans les transformations si disponibles
        transform_data = metadata.get('transforms', [])
        rotation_transform_angle = 0.0
        if isinstance(transform_data, list):
            for transform in transform_data:
                if transform.get('transform_type') == 'rotation':
                    try:
                        rotation_transform_angle = float(transform.get('params', {}).get('angle', 0.0))
                    except (TypeError, ValueError):
                        rotation_transform_angle = 0.0
                    break
        
        if rotation_applied and abs(orientation_angle) > 0.1:
            flags.rotated = True
        elif abs(rotation_transform_angle) > 0.1:
            flags.rotated = True
        else:
            flags.rotated = False
        
        # 8. Capture type (SCAN vs PHOTO)
        flags.capture_type = metadata.get('capture_type', 'UNKNOWN')
        flags.capture_white_percentage = float(metadata.get('capture_white_percentage', 0.0))
        
        return flags
    
    def _calculate_crop_margins(self, 
                                original: np.ndarray,
                                final: np.ndarray,
                                crop_metadata: Dict[str, Any]) -> Dict[str, float]:
        """
        Calcule les marges de crop en pourcentage
        
        Returns:
            Dict avec 'top', 'bottom', 'left', 'right' en pourcentage
        """
        orig_h, orig_w = original.shape[:2]
        final_h, final_w = final.shape[:2]
        
        # Si on a les points source, calculer précisément
        if 'source_points' in crop_metadata:
            source_points = np.array(crop_metadata['source_points'])
            # Trouver les bords du quadrilatère
            min_x = source_points[:, 0].min()
            max_x = source_points[:, 0].max()
            min_y = source_points[:, 1].min()
            max_y = source_points[:, 1].max()
            
            return {
                'top': (min_y / orig_h) * 100,
                'bottom': ((orig_h - max_y) / orig_h) * 100,
                'left': (min_x / orig_w) * 100,
                'right': ((orig_w - max_x) / orig_w) * 100
            }
        
        # Sinon, estimation basée sur les tailles
        width_ratio = final_w / orig_w
        height_ratio = final_h / orig_h
        
        # Estimation simple (suppose crop centré)
        margin = (1 - min(width_ratio, height_ratio)) * 50
        
        return {
            'top': margin,
            'bottom': margin,
            'left': margin,
            'right': margin
        }
    
    def _calculate_contrast(self, image: np.ndarray) -> float:
        """
        Calcule le contraste d'une image (écart-type des niveaux de gris)
        
        Returns:
            Contraste (écart-type)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        return float(np.std(gray))


def save_qa_flags(output_path: str, flags: QAFlags):
    """
    Sauvegarde les flags QA dans un fichier JSON
    
    Args:
        output_path: Chemin du fichier de sortie
        flags: Flags QA à sauvegarder
    """
    try:
        from pathlib import Path
        from src.utils.file_handler import ensure_dir
        
        qa_file = Path(output_path).with_suffix('.qa.json')
        
        # S'assurer que le répertoire existe
        ensure_dir(str(qa_file.parent))
        
        with open(qa_file, 'w', encoding='utf-8') as f:
            json.dump(flags.to_dict(), f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde des flags QA pour {output_path}", exc_info=True)


def load_qa_flags(output_path: str) -> Optional[QAFlags]:
    """
    Charge les flags QA depuis un fichier JSON
    
    Args:
        output_path: Chemin du fichier de sortie
        
    Returns:
        QAFlags ou None si le fichier n'existe pas
    """
    from pathlib import Path
    import os
    
    qa_file = Path(output_path).with_suffix('.qa.json')
    
    if not os.path.exists(qa_file):
        return None
    
    with open(qa_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return QAFlags.from_dict(data)

