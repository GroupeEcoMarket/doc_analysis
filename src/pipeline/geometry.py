"""
Geometry Normalization
Normalise la géométrie des documents (orientation, dimensions, etc.)
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import time
from dataclasses import dataclass
from onnxtr.io import DocumentFile as OnnxTRDocumentFile
from src.utils.file_handler import ensure_dir, get_files, get_output_path
from src.utils.transform_handler import (
    TransformSequence, Transform, save_transforms, get_transform_file_path
)
from src.utils.qa_flags import QADetector, QAFlags, save_qa_flags
from src.utils.config_loader import GeometryConfig, QAConfig, PerformanceConfig, OutputConfig
from src.utils.logger import get_logger
from src.utils.exceptions import GeometryError, ModelLoadingError, ImageProcessingError
from src.pipeline.models import GeometryOutput, CropMetadata, DeskewMetadata
from typing import Optional

logger = get_logger(__name__)


@dataclass
class OrientationResult:
    """
    Résultat standardisé de la détection d'orientation.
    Cette classe encapsule la sortie du modèle d'orientation pour isoler
    le reste du pipeline des changements internes de la librairie onnxtr.
    """
    angle: int  # Angle de rotation en degrés (0, 90, 180, 270)
    confidence: float  # Confiance entre 0.0 et 1.0
    
    def __post_init__(self):
        """Valide et normalise les valeurs après initialisation."""
        # Normaliser l'angle à une valeur entre 0 et 360
        self.angle = self.angle % 360
        # Convertir en angle standard (0, 90, 180, 270)
        if self.angle not in [0, 90, 180, 270]:
            # Arrondir à l'angle le plus proche
            self.angle = min([0, 90, 180, 270], key=lambda x: abs(x - self.angle))
        # S'assurer que la confiance est dans [0, 1]
        self.confidence = max(0.0, min(1.0, self.confidence))
    
    @property
    def needs_rotation(self) -> bool:
        """Indique si une rotation est nécessaire."""
        return self.angle != 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit le résultat en dictionnaire pour compatibilité."""
        return {
            'angle': self.angle,
            'needs_rotation': self.needs_rotation,
            'confidence': self.confidence,
            'status': 'success'
        }


class OrientationModelAdapter:
    """
    Adaptateur (Wrapper) autour du modèle d'orientation onnxtr.
    
    Le seul rôle de cet adaptateur est d'appeler le modèle et de toujours
    retourner un objet OrientationResult standardisé. Le reste du pipeline
    interagit uniquement avec cet objet standardisé, le rendant insensible
    aux changements internes de la librairie onnxtr.
    """
    
    def __init__(self, model: Any):
        """
        Initialise l'adaptateur avec le modèle onnxtr.
        
        Args:
            model: Le modèle d'orientation onnxtr (page_orientation_predictor)
        """
        self.model = model
    
    def predict(self, image_path: str) -> OrientationResult:
        """
        Prédit l'orientation d'un document et retourne un résultat standardisé.
        
        Args:
            image_path: Chemin vers l'image du document
            
        Returns:
            OrientationResult: Résultat standardisé de la détection
            
        Raises:
            GeometryError: Si la prédiction échoue ou si le format de sortie
                          du modèle n'est pas reconnu
        """
        try:
            # Charger l'image avec onnxtr
            doc = OnnxTRDocumentFile.from_images([image_path])
            
            # Prédire l'orientation
            orientation_result = self.model(doc)
            
            # Parser la sortie du modèle et extraire angle et confidence
            angle, confidence = self._parse_model_output(orientation_result)
            
            # Retourner un résultat standardisé
            return OrientationResult(angle=angle, confidence=confidence)
            
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction d'orientation: {e}", exc_info=True)
            raise GeometryError(f"Échec de la détection d'orientation: {e}") from e
    
    def _parse_model_output(self, orientation_result: Any) -> Tuple[int, float]:
        """
        Parse la sortie du modèle onnxtr et extrait l'angle et la confiance.
        
        Cette méthode centralise toute la logique de parsing fragile.
        Si la librairie change sa sortie, seule cette méthode doit être modifiée.
        
        Args:
            orientation_result: Sortie brute du modèle onnxtr
            
        Returns:
            tuple: (angle, confidence) où angle est en degrés et confidence entre 0 et 1
            
        Raises:
            GeometryError: Si le format de sortie n'est pas reconnu
        """
        angle = 0
        confidence = 1.0
        processed = False
        
        # Cas 1: Tuple/Liste de 3 éléments (class_indices, class_angles, confidences)
        if isinstance(orientation_result, (list, tuple)):
            if len(orientation_result) == 3 and all(isinstance(elem, (list, np.ndarray)) for elem in orientation_result):
                try:
                    class_indices, class_angles, confidences = orientation_result
                    if len(class_angles) > 0:
                        raw_angle = int(class_angles[0])
                        raw_confidence = float(confidences[0]) if len(confidences) > 0 else 1.0
                        angle = (raw_angle + 360) % 360
                        allowed_angles = [0, 90, 180, 270]
                        if angle not in allowed_angles:
                            angle = min(allowed_angles, key=lambda x: abs(x - angle))
                        confidence = max(0.0, min(1.0, raw_confidence))
                        processed = True
                except Exception as parse_error:
                    logger.debug("Erreur parsing OrientationPredictor (format tuple 3 éléments)", exc_info=True)
            
            # Cas 2: Liste avec un premier élément (array numpy ou liste)
            if not processed and len(orientation_result) > 0:
                first_result = orientation_result[0]
                angle, confidence, processed = self._parse_single_result(first_result)
        
        # Cas 3: Array numpy directement
        if not processed and isinstance(orientation_result, np.ndarray):
            angle, confidence, processed = self._parse_numpy_array(orientation_result)
        
        # Cas 4: Dictionnaire
        if not processed and isinstance(orientation_result, dict):
            angle = int(orientation_result.get('orientation', orientation_result.get('angle', 0)))
            confidence = float(orientation_result.get('confidence', orientation_result.get('prob', 1.0)))
            processed = True
        
        # Cas 5: Objet avec attributs
        if not processed:
            if hasattr(orientation_result, 'orientation'):
                angle = int(orientation_result.orientation)
                if hasattr(orientation_result, 'confidence'):
                    confidence = float(orientation_result.confidence)
                processed = True
            elif hasattr(orientation_result, 'angle'):
                angle = int(orientation_result.angle)
                if hasattr(orientation_result, 'confidence'):
                    confidence = float(orientation_result.confidence)
                processed = True
            elif hasattr(orientation_result, 'pages') and len(orientation_result.pages) > 0:
                # Objet Document avec pages
                first_page = orientation_result.pages[0]
                if hasattr(first_page, 'orientation'):
                    angle = int(first_page.orientation)
                elif isinstance(first_page, dict):
                    angle = int(first_page.get('orientation', first_page.get('angle', 0)))
                    confidence = float(first_page.get('confidence', first_page.get('prob', 1.0)))
                if hasattr(first_page, 'confidence'):
                    confidence = float(first_page.confidence)
                elif hasattr(first_page, 'prob'):
                    confidence = float(first_page.prob)
                processed = True
        
        # Cas 6: Nombre directement (angle)
        if not processed and isinstance(orientation_result, (int, float)):
            angle = int(orientation_result)
            confidence = 1.0
            processed = True
        
        if not processed:
            raise GeometryError(
                f"Format de sortie du modèle d'orientation non reconnu. "
                f"Type reçu: {type(orientation_result)}, "
                f"Valeur: {orientation_result}"
            )
        
        return angle, confidence
    
    def _parse_single_result(self, result: Any) -> Tuple[int, float, bool]:
        """
        Parse un résultat unique (premier élément d'une liste).
        
        Returns:
            tuple: (angle, confidence, processed) où processed indique si le parsing a réussi
        """
        # Array numpy
        if isinstance(result, np.ndarray):
            return self._parse_numpy_array(result)
        
        # Liste imbriquée
        if isinstance(result, (list, tuple)):
            try:
                arr = np.array(result)
                if isinstance(arr, np.ndarray):
                    return self._parse_numpy_array(arr)
            except Exception:
                pass
        
        # Dictionnaire
        if isinstance(result, dict):
            angle = int(result.get('orientation', result.get('angle', 0)))
            confidence = float(result.get('confidence', result.get('prob', 1.0)))
            return angle, confidence, True
        
        # Objet avec attributs
        if hasattr(result, 'orientation'):
            angle = int(result.orientation)
            confidence = float(result.confidence) if hasattr(result, 'confidence') else 1.0
            return angle, confidence, True
        
        # Nombre
        if isinstance(result, (int, float)):
            return int(result), 1.0, True
        
        return 0, 1.0, False
    
    def _parse_numpy_array(self, arr: np.ndarray) -> Tuple[int, float, bool]:
        """
        Parse un array numpy contenant des probabilités ou logits.
        
        Returns:
            tuple: (angle, confidence, processed) où processed indique si le parsing a réussi
        """
        if not isinstance(arr, np.ndarray):
            return 0, 1.0, False
        
        # Normaliser les dimensions
        if hasattr(arr, 'ndim') and arr.ndim == 2:
            arr = arr[0]  # Prendre le premier élément du batch
            if not isinstance(arr, np.ndarray):
                arr = np.array(arr)
        
        # Vérifier que c'est un array 1D avec 4 valeurs (probabilités pour 0°, 90°, 180°, 270°)
        if not isinstance(arr, np.ndarray) or not hasattr(arr, 'ndim') or arr.ndim != 1 or len(arr) != 4:
            return 0, 1.0, False
        
        # Vérifier si ce sont des logits (valeurs non normalisées) ou des probabilités
        if np.any(arr < 0) or np.any(arr > 1):
            # Appliquer softmax pour convertir les logits en probabilités
            exp_logits = np.exp(arr - np.max(arr))  # Pour stabilité numérique
            probs = exp_logits / np.sum(exp_logits)
        else:
            # Ce sont déjà des probabilités, normaliser pour être sûr
            probs = arr / np.sum(arr) if np.sum(arr) > 0 else arr
        
        # Trouver la classe avec la probabilité maximale
        predicted_class = int(np.argmax(probs))
        # Convertir l'index de classe en angle
        angle_map = {0: 0, 1: 90, 2: 180, 3: 270}
        angle = angle_map.get(predicted_class, 0)
        # La confiance est la probabilité maximale
        confidence = float(np.max(probs))
        
        return angle, confidence, True


def fit_quad_to_pts(points: np.ndarray) -> np.ndarray:
    """
    Ordonne les 4 points d'un quadrilatère dans l'ordre : top-left, top-right, bottom-right, bottom-left
    
    Args:
        points: Array de 4 points (shape: (4, 2))
        
    Returns:
        Array de points ordonnés (shape: (4, 2))
    """
    if len(points) != 4:
        raise ValueError("fit_quad_to_pts nécessite exactement 4 points")
    
    points = points.reshape(4, 2).astype(np.float32)
    
    # Méthode simple et robuste :
    # 1. Trouver le point avec la somme minimale (top-left)
    # 2. Trouver le point avec la somme maximale (bottom-right)
    # 3. Pour les deux autres, celui avec x plus grand est top-right
    
    sums = np.sum(points, axis=1)
    top_left_idx = np.argmin(sums)
    bottom_right_idx = np.argmax(sums)
    
    # Identifier les deux points restants
    remaining_indices = [i for i in range(4) if i != top_left_idx and i != bottom_right_idx]
    
    # Le point avec x plus grand parmi les deux restants est top-right
    if points[remaining_indices[0]][0] > points[remaining_indices[1]][0]:
        top_right_idx = remaining_indices[0]
        bottom_left_idx = remaining_indices[1]
    else:
        top_right_idx = remaining_indices[1]
        bottom_left_idx = remaining_indices[0]
    
    # Retourner dans l'ordre : top-left, top-right, bottom-right, bottom-left
    return np.array([
        points[top_left_idx],
        points[top_right_idx],
        points[bottom_right_idx],
        points[bottom_left_idx]
    ], dtype=np.float32)


class GeometryNormalizer:
    """
    Normalise la géométrie des documents (rotation, redimensionnement, crop intelligent, etc.)
    Utilise:
    - onnxtr-mobilenet-v3-small-page-orientation pour la détection d'orientation
    - doctr db_resnet50 pour la détection et le crop intelligent de pages
    """
    
    def __init__(
        self,
        geo_config: GeometryConfig,
        qa_config: QAConfig,
        perf_config: PerformanceConfig,
        output_config: OutputConfig,
        model_registry: "ModelRegistry"
    ):
        """
        Initialise le normaliseur géométrique.
        
        Args:
            geo_config: Configuration géométrique (injectée via DI, obligatoire)
            qa_config: Configuration QA (injectée via DI, obligatoire)
            perf_config: Configuration de performance (injectée via DI, obligatoire)
            output_config: Configuration de sortie (injectée via DI, obligatoire)
            model_registry: Registre de modèles (injecté via DI, obligatoire)
        """
        self.geo_config = geo_config
        self.qa_config = qa_config
        self.perf_config = perf_config
        self.output_config = output_config
        self.model_registry = model_registry
        
        # Charger les paramètres directement depuis la config
        self.crop_threshold = self.geo_config.crop_min_area_ratio
        self.crop_max_margin_ratio = self.geo_config.crop_max_margin_ratio
        self.enable_crop = self.geo_config.crop_enabled
        self.enable_deskew = self.geo_config.deskew_enabled
        self.deskew_max_angle = self.geo_config.deskew_max_angle
        self.deskew_min_angle = self.geo_config.deskew_min_angle
        self.deskew_min_confidence = self.geo_config.deskew_min_confidence
        self.deskew_hough_threshold = self.geo_config.deskew_hough_threshold
        
        # Orientation confidence threshold
        self.orientation_min_confidence = self.geo_config.orientation_min_confidence
        
        # Capture classifier (pour décider si on skip le crop)
        self.capture_classifier_skip_crop_if_scan = self.geo_config.capture_classifier_skip_crop_if_scan
    
    def _save_image(self, output_path: str, image: np.ndarray) -> None:
        """
        Sauvegarde une image dans le format configuré.
        
        Args:
            output_path: Chemin de sortie (sans extension)
            image: Image à sauvegarder (format OpenCV BGR)
        """
        # Déterminer l'extension selon le format configuré
        image_format = self.output_config.image_format.lower()
        if image_format == 'jpg' or image_format == 'jpeg':
            # Ajouter l'extension .jpg si nécessaire
            if not output_path.lower().endswith(('.jpg', '.jpeg')):
                output_path = str(Path(output_path).with_suffix('.jpg'))
            # Sauvegarder en JPEG avec la qualité configurée
            cv2.imwrite(
                output_path,
                image,
                [cv2.IMWRITE_JPEG_QUALITY, self.output_config.jpeg_quality]
            )
        else:
            # Par défaut, sauvegarder en PNG (lossless)
            if not output_path.lower().endswith('.png'):
                output_path = str(Path(output_path).with_suffix('.png'))
            cv2.imwrite(output_path, image)
    
    def _get_orientation_adapter(self) -> OrientationModelAdapter:
        """
        Récupère l'adaptateur du modèle d'orientation via le registre.
        
        Returns:
            OrientationModelAdapter: L'adaptateur qui encapsule le modèle onnxtr
        """
        return self.model_registry.get_orientation_adapter()
    
    def _get_detection_model(self) -> Any:
        """
        Récupère le modèle de détection via le registre.
        
        Returns:
            Modèle de détection doctr, ou None si le crop est désactivé
        """
        if not self.enable_crop:
            return None
        try:
            return self.model_registry.get_detection_model()
        except ModelLoadingError as e:
            logger.warning(f"Impossible de charger le modèle de détection. Le crop intelligent sera désactivé: {e}")
            self.enable_crop = False
            return None
    
    def detect_orientation(self, image_path: str) -> Dict[str, Any]:
        """
        Détecte l'orientation d'un document en utilisant onnxtr-mobilenet-v3-small-page-orientation.
        
        Cette méthode utilise l'adaptateur OrientationModelAdapter qui encapsule
        toute la logique fragile de parsing de la sortie du modèle. Le reste du
        pipeline interagit uniquement avec un objet OrientationResult standardisé.
        
        Args:
            image_path: Chemin vers l'image du document
            
        Returns:
            dict: Résultats de détection avec l'angle de rotation nécessaire
                - angle: Angle de rotation en degrés (0, 90, 180, 270)
                - needs_rotation: True si une rotation est nécessaire
                - confidence: Confiance entre 0.0 et 1.0
                - status: 'success' ou 'error'
        """
        try:
            # Récupérer l'adaptateur (qui gère le lazy loading du modèle)
            adapter = self._get_orientation_adapter()
            
            # Utiliser l'adaptateur pour obtenir un résultat standardisé
            result = adapter.predict(image_path)
            
            # Convertir le résultat en dictionnaire pour compatibilité
            return result.to_dict()
            
        except Exception as e:
            logger.error(f"Erreur lors de la détection d'orientation: {e}", exc_info=True)
            return {
                'angle': 0,
                'needs_rotation': False,
                'confidence': 0.0,
                'status': 'error',
                'error': str(e)
            }
    
    
    def intelligent_crop(self, image: np.ndarray, capture_type: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Analyse une image, détecte le document et le découpe UNIQUEMENT
        si le document n'occupe pas déjà toute l'image.
        
        Peut sauter le crop si l'image est détectée comme un SCAN.
        
        Args:
            image: Une image sous forme de tableau NumPy (format OpenCV BGR).
            capture_type: Type de capture ('SCAN' ou 'PHOTO'), optionnel
        
        Returns:
            tuple: (image_cropped, metadata)
                - image_cropped: L'image du document, découpée et redressée, ou l'image originale
                - metadata: Dictionnaire avec les informations sur le crop
        """
        metadata = {
            'crop_applied': False,
            'area_ratio': 1.0,
            'status': 'skipped'
        }
        
        if not self.enable_crop:
            metadata['reason'] = 'Crop désactivé dans la configuration'
            return image, metadata
        
        # Si c'est un SCAN et qu'on doit sauter le crop
        if capture_type == 'SCAN' and self.capture_classifier_skip_crop_if_scan:
            metadata['status'] = 'skipped_scan'
            metadata['reason'] = 'Image classée comme SCAN, crop ignoré'
            return image, metadata
        
        try:
            model = self._get_detection_model()
            if model is None:
                return image, metadata
            
            # Le predictor doctr accepte directement une liste d'images numpy (BGR)
            output = model([image])
            
            # Vérifier si quelque chose a été détecté
            # La structure peut varier : output peut être une liste, un dict, ou un objet
            detections = None
            if isinstance(output, dict):
                detections = output.get('preds', output.get('detections', output.get('pages', [])))
            elif isinstance(output, list):
                detections = output[0] if len(output) > 0 else []
            elif hasattr(output, 'preds'):
                detections = output.preds
            elif hasattr(output, 'pages'):
                detections = output.pages
            
            if not detections or (isinstance(detections, list) and len(detections) == 0):
                metadata['status'] = 'no_detection'
                return image, metadata
            
            # Si detections est une liste, prendre le premier élément
            if isinstance(detections, list):
                detection = detections[0]
            else:
                detection = detections
            
            # Extraire les coordonnées du polygone
            # Les modèles de détection doctr retournent généralement des géométries
            corners = None
            
            # Essayer différentes structures possibles
            if hasattr(detection, 'geometry'):
                corners = np.array(detection.geometry, dtype=np.float32)
            elif hasattr(detection, 'polygon'):
                corners = np.array(detection.polygon, dtype=np.float32)
            elif isinstance(detection, dict):
                if 'geometry' in detection or 'polygon' in detection:
                    corners = np.array(detection.get('geometry', detection.get('polygon', [])), dtype=np.float32)
                elif 'words' in detection:
                    words = detection['words']
                    if isinstance(words, np.ndarray) and words.size > 0:
                        # words format: [xmin, ymin, xmax, ymax, confidence] en coordonnées normalisées
                        xmin = float(np.min(words[:, 0]))
                        ymin = float(np.min(words[:, 1]))
                        xmax = float(np.max(words[:, 2]))
                        ymax = float(np.max(words[:, 3]))
                        corners = np.array([
                            [xmin, ymin],
                            [xmax, ymin],
                            [xmax, ymax],
                            [xmin, ymax]
                        ], dtype=np.float32)
            elif hasattr(detection, 'bbox'):
                # Si c'est une bounding box, convertir en polygone
                bbox = detection.bbox
                if isinstance(bbox, (list, np.ndarray)) and len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]
                    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            
            if corners is None or len(corners) < 4:
                metadata['status'] = 'invalid_geometry'
                return image, metadata
            
            # Analyser le résultat de la détection
            image_height, image_width = image.shape[:2]
            image_area = image_height * image_width
            
            # Convertir les coordonnées relatives (0-1) en coordonnées pixels
            # Les coordonnées peuvent être dans différents formats
            if corners.max() <= 1.0:
                # Coordonnées normalisées
                abs_corners = corners * np.array([image_width, image_height], dtype=np.float32)
            else:
                # Coordonnées déjà en pixels
                abs_corners = corners.astype(np.float32)
            
            # Calculer l'aire du polygone détecté
            # Convertir en format pour cv2.contourArea
            contour = abs_corners.reshape(-1, 1, 2).astype(np.int32)
            document_area = cv2.contourArea(contour)
            
            area_ratio = document_area / image_area
            metadata['area_ratio'] = float(area_ratio)
            
            # Décider s'il faut découper
            # Si le document occupe plus de crop_threshold (95% par défaut) de l'image,
            # on considère qu'il est déjà cadré
            if area_ratio > self.crop_threshold:
                metadata['status'] = 'already_cropped'
                return image, metadata
            
            # Ordonner les points du polygone
            try:
                target_points = fit_quad_to_pts(abs_corners)
            except Exception:
                # Si fit_quad_to_pts échoue, utiliser les points directement
                target_points = abs_corners.reshape(4, 2)
            
            # Calculer les marges du crop pour vérifier le risque d'overcrop
            # Trouver les bords du quadrilatère détecté
            min_x = float(target_points[:, 0].min())
            max_x = float(target_points[:, 0].max())
            min_y = float(target_points[:, 1].min())
            max_y = float(target_points[:, 1].max())
            
            # Calculer les marges en pourcentage de chaque bord
            margins = {
                'top': (min_y / image_height) * 100,
                'bottom': ((image_height - max_y) / image_height) * 100,
                'left': (min_x / image_width) * 100,
                'right': ((image_width - max_x) / image_width) * 100
            }
            
            # Vérifier si une marge est inférieure au seuil (overcrop risk)
            max_margin_threshold_percent = self.crop_max_margin_ratio * 100
            if any(margin < max_margin_threshold_percent for margin in margins.values()):
                # Rejeter le crop si une marge est trop petite
                smallest_margin = min(margins.values())
                logger.warning(
                    f"Crop rejeté: marge minimale {smallest_margin:.2f}% < seuil {max_margin_threshold_percent:.2f}%. "
                    f"Marges: top={margins['top']:.2f}%, bottom={margins['bottom']:.2f}%, "
                    f"left={margins['left']:.2f}%, right={margins['right']:.2f}%"
                )
                metadata['status'] = 'rejected_overcrop'
                metadata['reason'] = f'Marge minimale {smallest_margin:.2f}% < seuil {max_margin_threshold_percent:.2f}%'
                metadata['margins'] = margins
                return image, metadata
            
            # Découper et redresser
            metadata['crop_applied'] = True
            metadata['status'] = 'cropped'
            metadata['margins'] = margins
            
            # Calculer les dimensions de la sortie
            # Calculer la largeur et hauteur du rectangle de sortie
            width_top = np.linalg.norm(target_points[1] - target_points[0])
            width_bottom = np.linalg.norm(target_points[2] - target_points[3])
            height_left = np.linalg.norm(target_points[3] - target_points[0])
            height_right = np.linalg.norm(target_points[2] - target_points[1])
            
            target_width = int(max(width_top, width_bottom))
            target_height = int(max(height_left, height_right))
            
            # Points de destination pour la transformation de perspective
            dst_points = np.array([
                [0, 0],
                [target_width, 0],
                [target_width, target_height],
                [0, target_height]
            ], dtype=np.float32)
            
            # Calculer la matrice de transformation
            transform_matrix = cv2.getPerspectiveTransform(target_points, dst_points)
            
            # Stocker les paramètres de transformation dans les métadonnées
            metadata['transform_matrix'] = transform_matrix.tolist()
            metadata['source_points'] = target_points.tolist()
            metadata['destination_points'] = dst_points.tolist()
            metadata['output_size'] = [target_width, target_height]
            
            # Appliquer la transformation
            warped_image = cv2.warpPerspective(image, transform_matrix, (target_width, target_height))
            
            return warped_image, metadata
            
        except Exception as e:
            metadata['status'] = 'error'
            metadata['error'] = str(e)
            return image, metadata
    
    def rotate_image(self, image_path: str, angle: float) -> np.ndarray:
        """
        Fait tourner une image selon un angle
        
        Args:
            image_path: Chemin vers l'image
            angle: Angle de rotation en degrés (0, 90, 180, 270)
            
        Returns:
            np.ndarray: Image tournée
        """
        # Charger l'image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Impossible de charger l'image: {image_path}")
        
        # Convertir l'angle en entier pour les rotations standard
        if angle == 90:
            # Rotation 90° dans le sens horaire
            img_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            img_rotated = cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            # Rotation 270° = rotation 90° dans le sens anti-horaire
            img_rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            # Pour d'autres angles, utiliser une rotation avec interpolation
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img_rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
        
        return img_rotated
    
    def detect_skew_angle(self, image: np.ndarray) -> Tuple[float, float]:
        """
        Détecte l'angle d'inclinaison (skew) d'un document en utilisant la transformée de Hough
        
        Args:
            image: Image en niveaux de gris ou couleur (format OpenCV BGR)
            
        Returns:
            tuple:
                - angle (float): Inclinaison en degrés (positif = rotation dans le sens horaire)
                - confidence (float): Probabilité/confidence entre 0 et 1
        """
        # Convertir en niveaux de gris si nécessaire
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Appliquer un seuil pour obtenir une image binaire
        # Utiliser Otsu pour un seuillage automatique
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Détecter les contours
        # Utiliser Canny pour détecter les bords sur l'image binaire (binary) originale
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)
        
        # Appliquer la transformée de Hough pour détecter les lignes
        # Utiliser le seuil depuis la config
        lines = cv2.HoughLines(edges, 1, np.pi / 180, self.deskew_hough_threshold)
        
        if lines is None or len(lines) == 0:
            return 0.0, 0.0
        
        # Calculer les angles de toutes les lignes détectées
        angles = []
        for line in lines:
            rho, theta = line[0]
            # Convertir theta en angle en degrés
            angle = np.degrees(theta) - 90
            # Normaliser l'angle entre -45 et 45 degrés
            if angle < -45:
                angle += 90
            elif angle > 45:
                angle -= 90
            angles.append(angle)
        
        if len(angles) == 0:
            return 0.0, 0.0
        
        # Utiliser la médiane pour éviter les valeurs aberrantes
        median_angle = np.median(angles)
        
        # Limiter l'angle à la plage maximale autorisée
        if abs(median_angle) > self.deskew_max_angle:
            return 0.0, 0.0

        # Calculer une confiance basée sur la concentration des angles autour de la médiane
        deviations = np.abs(np.array(angles) - median_angle)
        tolerance = 1.0  # degrés
        inliers = np.sum(deviations <= tolerance)
        confidence = 0.0
        if len(angles) > 0:
            confidence = inliers / len(angles)
            # Pondérer légèrement par le nombre de lignes détectées (max 1.0)
            # confidence *= min(1.0, len(angles) / 20.0)
        
        confidence = float(np.clip(confidence, 0.0, 1.0))
        return float(median_angle), confidence
    
    def deskew_image(self, image: np.ndarray, angle: float, confidence: float = 0.0) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Corrige l'inclinaison d'une image
        
        Args:
            image: Image à corriger (format OpenCV BGR)
            angle: Angle d'inclinaison en degrés
            confidence: Probabilité/confidence associée à la détection d'angle
            
        Returns:
            tuple: (image_corrected, metadata)
                - image_corrected: Image corrigée
                - metadata: Dictionnaire avec les informations sur le deskew
        """
        metadata = {
            'deskew_applied': False,
            'angle': float(angle),
            'status': 'skipped',
            'confidence': float(confidence)
        }
        
        # Vérifier si le deskew est activé
        if not self.enable_deskew:
            metadata['status'] = 'disabled'
            return image, metadata
        
        # Vérifier si l'angle est assez grand pour justifier une correction
        if abs(angle) < self.deskew_min_angle:
            metadata['status'] = 'angle_too_small'
            return image, metadata
        
        # Vérifier si la confiance est suffisante
        if confidence < self.deskew_min_confidence:
            metadata['status'] = 'low_confidence'
            metadata['reason'] = f'Confidence {confidence:.2f} < threshold {self.deskew_min_confidence:.2f}'
            return image, metadata
        
        try:
            # Calculer les nouvelles dimensions pour éviter de couper l'image
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            
            # Calculer la matrice de rotation
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Calculer les nouvelles dimensions pour contenir toute l'image
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            
            # Ajuster la matrice de rotation pour le nouveau centre
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]
            
            # Stocker les paramètres de transformation dans les métadonnées
            metadata['transform_matrix'] = M.tolist()
            metadata['center'] = list(center)
            metadata['output_size'] = [new_w, new_h]
            metadata['original_size'] = [w, h]
            
            # Appliquer la rotation
            corrected = cv2.warpAffine(image, M, (new_w, new_h), 
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=(255, 255, 255))
            
            metadata['deskew_applied'] = True
            metadata['status'] = 'success'
            
            return corrected, metadata
            
        except Exception as e:
            metadata['status'] = 'error'
            metadata['error'] = str(e)
            return image, metadata
    
    def process(self, img: np.ndarray, output_path: str, capture_type: str, original_input_path: str, capture_info: Optional[Dict[str, Any]] = None) -> GeometryOutput:
        """
        Traite un document pour normaliser sa géométrie
        
        Pipeline selon spécifications:
        1. Détection de Page (doctr detection.db_resnet50) -> Coordonnées des 4 coins
        2. Découpage & Redressement (cv2.warpPerspective) -> Image rectangulaire mais peut-être penchée
        3. Calcul de l'Angle Fin (deskew detect_skew) -> Angle en degrés (ex: 1.7)
        4. Correction de l'Angle Fin (cv2.warpAffine) -> Image parfaitement droite
        5. Classification 0/90/180/270 (onnxtr) -> Classe d'orientation
        6. Rotation Finale (cv2.rotate) -> Document final dans le bon sens
        
        Args:
            img: Image en format OpenCV (BGR) - déjà chargée et prétraitée
            output_path: Chemin de sortie pour le document normalisé
            capture_type: Type de capture ('SCAN' ou 'PHOTO') - déterminé par l'étape preprocessing
            original_input_path: Chemin original du fichier source (pour les métadonnées)
            capture_info: Informations complètes sur la classification de capture (optionnel)
            
        Returns:
            dict: Résultats du traitement
        """
        try:
            # Démarrer le chronomètre
            start_time = time.time()
            
            # S'assurer que le répertoire de sortie existe
            ensure_dir(os.path.dirname(output_path))
            
            # Utiliser le chemin original pour les transformations
            transform_input_path = original_input_path
            
            # Déterminer le chemin de l'image originale
            # Si output_path contient _transformed, on peut déduire _original
            output_original_path = None
            if '_transformed' in output_path:
                output_original_path = output_path.replace('_transformed', '_original')
            elif hasattr(self, '_current_output_original_path'):
                output_original_path = self._current_output_original_path
            
            # Initialiser la séquence de transformations
            transform_sequence = TransformSequence(transform_input_path, output_path, output_original_path)
            
            # Sauvegarder une copie de l'image originale pour QA
            original_image = img.copy()
            
            # Sauvegarder l'image originale si configuré
            if self.output_config.save_original and hasattr(self, '_current_output_original_path') and self._current_output_original_path:
                try:
                    # S'assurer que le répertoire de destination existe
                    ensure_dir(os.path.dirname(self._current_output_original_path))
                    self._save_image(self._current_output_original_path, original_image)
                except Exception as e:
                    logger.warning(f"Impossible de sauvegarder la copie de l'image source à {self._current_output_original_path}", exc_info=True)

            # Initialiser le détecteur QA avec GeometryConfig comme source unique de vérité
            qa_detector = QADetector(geo_config=self.geo_config)
            
            # Enregistrer la classification de capture si disponible
            if capture_info:
                transform_sequence.add_transform(Transform(
                    transform_type='capture_classification',
                    params={
                        'capture_type': capture_type,
                        'white_percentage': capture_info.get('white_percentage', 0.0),
                        'confidence': capture_info.get('confidence', 0.0),
                        'enabled': capture_info.get('enabled', False),
                        'reason': capture_info.get('reason', ''),
                        'white_level_threshold': capture_info.get('white_level_threshold'),
                        'white_percentage_threshold': capture_info.get('white_percentage_threshold')
                    },
                    order=0  # Première transformation (après preprocessing)
                ))
                        
            # Étape 1: Détection de Page (doctr detection.db_resnet50)
            # Étape 2: Découpage & Redressement (cv2.warpPerspective)
            crop_metadata = {}
            if self.enable_crop:
                img, crop_metadata = self.intelligent_crop(img, capture_type=capture_type)
                # Enregistrer la transformation de crop (même si rejetée) pour l'historique
                if crop_metadata.get('status') in ['cropped', 'rejected_overcrop']:
                    crop_params = {
                        'area_ratio': crop_metadata.get('area_ratio', 1.0),
                        'status': crop_metadata.get('status', ''),
                        'crop_applied': crop_metadata.get('crop_applied', False),
                    }
                    # Ajouter la raison si le crop a été rejeté
                    if crop_metadata.get('status') == 'rejected_overcrop':
                        crop_params['reason'] = crop_metadata.get('reason', '')
                        crop_params['margins'] = crop_metadata.get('margins', {})
                    # Ajouter les paramètres de transformation si disponibles
                    if 'transform_matrix' in crop_metadata:
                        crop_params['transform_matrix'] = crop_metadata['transform_matrix']
                    if 'source_points' in crop_metadata:
                        crop_params['source_points'] = crop_metadata['source_points']
                    if 'destination_points' in crop_metadata:
                        crop_params['destination_points'] = crop_metadata['destination_points']
                    if 'output_size' in crop_metadata:
                        crop_params['output_size'] = crop_metadata['output_size']
                    
                    transform_sequence.add_transform(Transform(
                        transform_type='crop',
                        params=crop_params,
                        order=1
                    ))
            
            # Étape 3: Calcul de l'Angle Fin (deskew detect_skew)
            # Étape 4: Correction de l'Angle Fin (cv2.warpAffine)
            skew_angle, skew_confidence = self.detect_skew_angle(img)
            deskew_metadata = {
                'deskew_applied': False,
                'angle': float(skew_angle),
                'confidence': float(skew_confidence),
                'status': 'skipped'
            }
            if self.enable_deskew and abs(skew_angle) >= 0.1:
                img, deskew_metadata = self.deskew_image(img, skew_angle, skew_confidence)
                if deskew_metadata.get('deskew_applied', False):
                    # Enregistrer la transformation de deskew avec tous les paramètres
                    deskew_params = {
                        'angle': deskew_metadata.get('angle', 0.0),
                        'confidence': deskew_metadata.get('confidence', 0.0),
                        'status': deskew_metadata.get('status', '')
                    }
                    # Ajouter les paramètres de transformation si disponibles
                    if 'transform_matrix' in deskew_metadata:
                        deskew_params['transform_matrix'] = deskew_metadata['transform_matrix']
                    if 'center' in deskew_metadata:
                        deskew_params['center'] = deskew_metadata['center']
                    if 'output_size' in deskew_metadata:
                        deskew_params['output_size'] = deskew_metadata['output_size']
                    if 'original_size' in deskew_metadata:
                        deskew_params['original_size'] = deskew_metadata['original_size']
                    
                    transform_sequence.add_transform(Transform(
                        transform_type='deskew',
                        params=deskew_params,
                        order=2
                    ))
            
            # Sauvegarder temporairement l'image après crop et deskew pour la détection d'orientation
            # (le modèle d'orientation peut nécessiter un fichier)
            temp_path = None
            import tempfile
            temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
            os.close(temp_fd)
            cv2.imwrite(temp_path, img)
            orientation_input = temp_path
            
            # Étape 5: Classification 0/90/180/270 (onnxtr)
            orientation_result = self.detect_orientation(orientation_input)
            
            # Nettoyer le fichier temporaire
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            
            if orientation_result['status'] == 'error':
                error_msg = orientation_result.get('error', 'Erreur inconnue lors de la détection d\'orientation')
                raise GeometryError(f"Échec de la détection d'orientation: {error_msg}")
            
            # Étape 6: Rotation Finale (cv2.rotate)
            rotation_applied = False
            rotation_angle = 0
            orientation_confidence = orientation_result.get('confidence', 1.0)
            
            # Vérifier la confiance avant d'appliquer la rotation
            if orientation_result['needs_rotation']:
                if orientation_confidence < self.orientation_min_confidence:
                    logger.warning(
                        f"Rotation rejetée: confiance {orientation_confidence:.2f} < seuil {self.orientation_min_confidence:.2f}. "
                        f"Angle détecté: {orientation_result.get('angle', 0)}°"
                    )
                    # Ne pas appliquer la rotation si la confiance est insuffisante
                    rotation_applied = False
                    rotation_angle = 0
                    # Enregistrer que la rotation a été rejetée pour l'historique
                    transform_sequence.add_transform(Transform(
                        transform_type='rotation',
                        params={
                            'angle': orientation_result.get('angle', 0),
                            'confidence': orientation_confidence,
                            'status': 'rejected',
                            'reason': f'Confidence {orientation_confidence:.2f} < threshold {self.orientation_min_confidence:.2f}'
                        },
                        order=3
                    ))
                else:
                    angle = orientation_result['angle']
                    rotation_angle = angle
                    # Appliquer la rotation directement sur l'image en mémoire
                    if angle == 90:
                        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    elif angle == 180:
                        img = cv2.rotate(img, cv2.ROTATE_180)
                    elif angle == 270:
                        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    else:
                        # Rotation arbitraire
                        h, w = img.shape[:2]
                        center = (w // 2, h // 2)
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
                    rotation_applied = True
                    
                    # Enregistrer la transformation de rotation
                    transform_sequence.add_transform(Transform(
                        transform_type='rotation',
                        params={
                            'angle': angle,
                            'confidence': orientation_confidence,
                            'status': 'applied',
                            'rotation_type': 'standard' if angle in [90, 180, 270] else 'arbitrary'
                        },
                        order=3
                    ))
            
            # Sauvegarder l'image finale si configuré
            if self.output_config.save_transformed:
                self._save_image(output_path, img)
            
            # Calculer le temps de traitement
            processing_time = time.time() - start_time
            
            # Détecter les flags QA
            # Convertir transform_sequence en liste de dicts pour la détection QA
            transforms_list = []
            if hasattr(transform_sequence, 'transforms'):
                for transform in transform_sequence.transforms:
                    transforms_list.append({
                        'transform_type': transform.transform_type,
                        'params': transform.params,
                        'order': transform.order
                    })
            
            metadata_for_qa = {
                'orientation_confidence': orientation_result.get('confidence', 1.0),
                'angle': orientation_result.get('angle', 0),  # Angle détecté par ONNX
                'crop_metadata': crop_metadata,
                'deskew_metadata': deskew_metadata,
                'rotation_applied': rotation_applied,
                'transforms': transforms_list,  # Ajouter les transformations pour détection QA
                'capture_type': capture_type,  # Type de capture (SCAN/PHOTO)
                'capture_white_percentage': capture_info.get('white_percentage', 0.0) if capture_info else 0.0
            }
            
            qa_flags = qa_detector.detect_flags(
                original_image=original_image,
                final_image=img,
                metadata=metadata_for_qa,
                processing_time=processing_time
            )
            
            # Sauvegarder les flags QA si configuré
            if self.output_config.save_qa_flags:
                try:
                    save_qa_flags(output_path, qa_flags)
                except Exception as e:
                    logger.error("Erreur lors de la sauvegarde des flags QA", exc_info=True)
            
            # Sauvegarder la séquence de transformations si configuré
            if self.output_config.save_transforms:
                try:
                    save_transforms(output_path, transform_sequence)
                except Exception as e:
                    logger.error("Erreur lors de la sauvegarde des transformations", exc_info=True)
            
            # Créer les modèles de métadonnées
            crop_metadata_obj = CropMetadata(**crop_metadata)
            deskew_metadata_obj = DeskewMetadata(**deskew_metadata)
            
            # Créer le modèle de sortie
            return GeometryOutput(
                status='success',
                input_path=original_input_path,
                output_path=output_path,  # Pour compatibilité
                output_transformed_path=output_path,
                output_original_path=output_original_path,
                transform_file=get_transform_file_path(output_path),
                qa_file=str(Path(output_path).with_suffix('.qa.json')),
                crop_applied=crop_metadata.get('crop_applied', False),
                crop_metadata=crop_metadata_obj,
                deskew_applied=deskew_metadata.get('deskew_applied', False),
                deskew_angle=deskew_metadata.get('angle', 0.0),
                deskew_metadata=deskew_metadata_obj,
                orientation_detected=True,
                angle=orientation_result.get('angle', 0),
                rotation_applied=rotation_applied,
                transforms=transform_sequence.to_dict(),
                qa_flags=qa_flags.to_dict(),
                processing_time=processing_time
            )
        except GeometryError:
            # Réexécuter les GeometryError telles quelles
            raise
        except Exception as e:
            # Convertir les autres exceptions en GeometryError
            raise GeometryError(f"Erreur lors du traitement géométrique: {str(e)}") from e
    
    def process_batch(self, input_dir: str, output_dir: str) -> List[GeometryOutput]:
        """
        Traite un lot de documents depuis le répertoire preprocessing avec parallélisation et batching
        
        Args:
            input_dir: Répertoire contenant les images prétraitées (sortie de preprocessing)
            output_dir: Répertoire de sortie
            
        Returns:
            list: Liste des résultats pour chaque document
        """
        import json
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        ensure_dir(output_dir)
        
        # Obtenir la liste des fichiers images à traiter
        image_extensions = ['.png', '.jpg', '.jpeg']
        files = get_files(input_dir, extensions=image_extensions)
        
        # Traiter par batches avec parallélisation
        batch_size = self.perf_config.batch_size
        max_workers = self.perf_config.max_workers
        results = []
        
        # Diviser les fichiers en batches
        for batch_start in range(0, len(files), batch_size):
            batch_end = min(batch_start + batch_size, len(files))
            batch_files = files[batch_start:batch_end]
            
            logger.info(f"Traitement du batch {batch_start // batch_size + 1} ({len(batch_files)} fichiers)")
            
            # Traiter le batch en parallèle
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._process_single_file, file_path, output_dir): file_path
                    for file_path in batch_files
                }
                
                for future in as_completed(futures):
                    file_path = futures[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        logger.error(f"Erreur lors du traitement de {file_path}", exc_info=True)
                        # Créer un GeometryOutput avec status='error'
                        error_output = GeometryOutput(
                            status='error',
                            input_path=file_path,
                            output_path='',
                            output_transformed_path='',
                            transform_file='',
                            qa_file='',
                            crop_applied=False,
                            crop_metadata=CropMetadata(
                                crop_applied=False,
                                area_ratio=0.0,
                                status='error'
                            ),
                            deskew_applied=False,
                            deskew_angle=0.0,
                            deskew_metadata=DeskewMetadata(
                                deskew_applied=False,
                                angle=0.0,
                                confidence=0.0,
                                status='error'
                            ),
                            orientation_detected=False,
                            angle=0,
                            rotation_applied=False,
                            transforms={},
                            qa_flags={},
                            processing_time=0.0,
                            error=str(e)
                        )
                        results.append(error_output)
        
        return results
    
    def _process_single_file(self, file_path: str, output_dir: str) -> Optional[GeometryOutput]:
        """
        Traite un seul fichier (méthode helper pour le traitement parallèle).
        
        Args:
            file_path: Chemin vers le fichier à traiter
            output_dir: Répertoire de sortie
            
        Returns:
            GeometryOutput ou None en cas d'erreur
        """
        import json
        
        try:
            # Charger l'image prétraitée
            img = cv2.imread(file_path)
            if img is None:
                raise ImageProcessingError(f"Impossible de charger l'image: {file_path}")
            
            # Essayer de charger les métadonnées de preprocessing (si disponibles)
            # Format attendu: fichier.json avec le même nom que l'image
            metadata_path = Path(file_path).with_suffix('.json')
            capture_type = 'PHOTO'  # Par défaut
            capture_info = None
            original_input_path = file_path  # Par défaut, utiliser le chemin de l'image prétraitée
            
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        capture_type = metadata.get('capture_type', 'PHOTO')
                        capture_info = metadata.get('capture_info', {})
                        original_input_path = metadata.get('input_path', file_path)
                except Exception as e:
                    logger.warning(f"Impossible de charger les métadonnées de {metadata_path}", exc_info=True)
            
            # Générer les chemins de sortie
            base_output = get_output_path(file_path, output_dir)
            path_obj = Path(base_output)
            
            # Chemin pour l'image transformée
            output_transformed_path = str(path_obj.with_stem(f"{path_obj.stem}_transformed"))
            
            # Stocker le chemin original pour l'utiliser dans process()
            output_original_path = str(path_obj.with_stem(f"{path_obj.stem}_original"))
            self._current_output_original_path = output_original_path
            
            # Traiter l'image
            result = self.process(
                img=img,
                output_path=output_transformed_path,
                capture_type=capture_type,
                original_input_path=original_input_path,
                capture_info=capture_info
            )
            
            # Mettre à jour les chemins dans le résultat
            result.output_original_path = output_original_path
            result.output_transformed_path = output_transformed_path
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de {file_path}", exc_info=True)
            raise

