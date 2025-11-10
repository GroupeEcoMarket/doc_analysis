"""
Geometry Normalization
Normalise la géométrie des documents (orientation, dimensions, etc.)
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
from onnxtr.models import from_hub
from onnxtr.models.classification.zoo import page_orientation_predictor
from onnxtr.io import DocumentFile as OnnxTRDocumentFile
from doctr.models import detection
from src.utils.file_handler import ensure_dir, get_files, get_output_path
from src.utils.transform_handler import (
    TransformSequence, Transform, save_transforms, get_transform_file_path
)
from src.utils.qa_flags import QADetector, QAFlags, save_qa_flags
from src.utils.config_loader import get_config
import time


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
    
    def __init__(self, config=None):
        """
        Initialise le normaliseur géométrique
        
        Args:
            config: Configuration optionnelle (dict ou None pour charger depuis config.yaml)
        """
        # Charger la configuration depuis config.yaml si non fournie
        if config is None:
            self.app_config = get_config()
            self.geo_config = self.app_config.geometry
            self.qa_config = self.app_config.qa
        else:
            # Compatibilité avec l'ancien format dict
            self.app_config = None
            self.geo_config = None
            self.qa_config = None
            self.legacy_config = config
        
        # Modèles (lazy loading)
        self.orientation_model = None
        self.detection_model = None
        
        # Charger les paramètres depuis la config
        if self.geo_config:
            # Nouvelle config (config.yaml)
            self.crop_threshold = self.geo_config.crop_min_area_ratio
            self.enable_crop = self.geo_config.crop_enabled
            self.enable_deskew = self.geo_config.deskew_enabled
            self.deskew_max_angle = self.geo_config.deskew_max_angle
            self.deskew_min_angle = self.geo_config.deskew_min_angle
            self.deskew_min_confidence = self.geo_config.deskew_min_confidence
            self.deskew_hough_threshold = self.geo_config.deskew_hough_threshold
            
            # Capture classifier (pour décider si on skip le crop)
            self.capture_classifier_skip_crop_if_scan = self.geo_config.capture_classifier_skip_crop_if_scan
        else:
            # Ancienne config (dict)
            self.crop_threshold = self.legacy_config.get('crop_threshold', 0.85)
            self.enable_crop = self.legacy_config.get('enable_crop', True)
            self.enable_deskew = self.legacy_config.get('enable_deskew', True)
            self.deskew_max_angle = self.legacy_config.get('deskew_max_angle', 15.0)
            self.deskew_min_angle = self.legacy_config.get('deskew_min_angle', 0.5)
            self.deskew_min_confidence = self.legacy_config.get('deskew_min_confidence', 0.20)
            self.deskew_hough_threshold = self.legacy_config.get('deskew_hough_threshold', 100)
            
            # Capture classifier (désactivé par défaut en mode legacy)
            self.capture_classifier_skip_crop_if_scan = self.legacy_config.get('capture_classifier_skip_crop_if_scan', True)
        
        self._load_orientation_model()
        self._load_detection_model()
    
    def _load_orientation_model(self):
        """
        Charge le modèle de détection d'orientation depuis Hugging Face
        """
        try:
            raw_model = from_hub('Felix92/onnxtr-mobilenet-v3-small-page-orientation')
            self.orientation_model = page_orientation_predictor(arch=raw_model)
        except Exception as e:
            print(f"Warning: Impossible de charger le modèle d'orientation: {e}")
            print("Le modèle sera chargé à la première utilisation")
    
    def _load_detection_model(self):
        """
        Charge le modèle de détection de pages doctr
        """
        if self.enable_crop:
            try:
                # Utiliser le predictor doctr qui gère le préprocessing
                self.detection_model = detection.detection_predictor(arch="db_resnet50", pretrained=True)
            except Exception as e:
                print(f"Warning: Impossible de charger le modèle de détection doctr: {e}")
                print("Le crop intelligent sera désactivé")
                self.enable_crop = False
    
    def _get_orientation_model(self):
        """
        Récupère le modèle d'orientation (lazy loading)
        """
        if self.orientation_model is None:
            try:
                raw_model = from_hub('Felix92/onnxtr-mobilenet-v3-small-page-orientation')
                self.orientation_model = page_orientation_predictor(arch=raw_model)
            except Exception as e:
                raise RuntimeError(f"Impossible de charger le modèle d'orientation: {e}")
        return self.orientation_model
    
    def _get_detection_model(self):
        """
        Récupère le modèle de détection (lazy loading)
        """
        if self.detection_model is None and self.enable_crop:
            try:
                self.detection_model = detection.detection_predictor(arch="db_resnet50", pretrained=True)
            except Exception as e:
                raise RuntimeError(f"Impossible de charger le modèle de détection: {e}")
        return self.detection_model
    
    def detect_orientation(self, image_path: str) -> Dict:
        """
        Détecte l'orientation d'un document en utilisant onnxtr-mobilenet-v3-small-page-orientation
        
        Args:
            image_path: Chemin vers l'image du document
            
        Returns:
            dict: Résultats de détection avec l'angle de rotation nécessaire
                - angle: Angle de rotation en degrés (0, 90, 180, 270)
                - needs_rotation: True si une rotation est nécessaire
                - status: 'success' ou 'error'
        """
        try:
            # Charger le modèle
            model = self._get_orientation_model()
            
            # Charger l'image avec onnxtr
            doc = OnnxTRDocumentFile.from_images([image_path])
            
            # Prédire l'orientation
            orientation_result = model(doc)
            
            # Le modèle peut retourner différentes structures :
            # - Un array numpy avec des probabilités/logits pour chaque classe
            # - Un dictionnaire avec 'orientation' et 'confidence'
            # - Un objet avec des attributs
            # - Une liste de résultats
            
            angle = 0
            confidence = 1.0
            
            processed_orientation = False

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
                            processed_orientation = True
                            class_idx_display = class_indices[0] if len(class_indices) > 0 else "N/A"
                            print(f"[DEBUG ONNX] OrientationPredictor -> class_idx={class_idx_display}, angle={angle}, confiance={confidence}")
                    except Exception as parse_error:
                        print(f"[DEBUG ONNX] Erreur parsing OrientationPredictor: {parse_error}")

            if not processed_orientation:
                # IMPORTANT: Traiter les listes EN PREMIER car elles peuvent contenir des arrays
                # Si c'est une liste (doit être vérifié avant np.ndarray car une liste peut être convertie en array)
                if isinstance(orientation_result, (list, tuple)) and len(orientation_result) > 0:
                    first_result = orientation_result[0]
                    print(f"[DEBUG ONNX] Premier élément de la liste: type={type(first_result)}, valeur={first_result}")
                    
                    # Si c'est un array numpy dans la liste
                    if isinstance(first_result, np.ndarray):
                        # Normaliser les dimensions
                        if hasattr(first_result, 'ndim') and first_result.ndim == 2:
                            first_result = first_result[0]
                            # Vérifier que first_result est toujours un array après l'indexation
                            if not isinstance(first_result, np.ndarray):
                                # Si ce n'est pas un array, essayer de le convertir
                                first_result = np.array(first_result)
                        
                        if isinstance(first_result, np.ndarray) and hasattr(first_result, 'ndim') and first_result.ndim == 1 and len(first_result) == 4:
                            # Vérifier si ce sont des logits ou des probabilités
                            if np.any(first_result < 0) or np.any(first_result > 1):
                                # Appliquer softmax
                                exp_logits = np.exp(first_result - np.max(first_result))
                                probs = exp_logits / np.sum(exp_logits)
                            else:
                                probs = first_result / np.sum(first_result) if np.sum(first_result) > 0 else first_result
                            
                            predicted_class = int(np.argmax(probs))
                            angle_map = {0: 0, 1: 90, 2: 180, 3: 270}
                            angle = angle_map.get(predicted_class, 0)
                            confidence = float(np.max(probs))
                    # Si c'est une liste imbriquée (liste de listes)
                    elif isinstance(first_result, (list, tuple)):
                        # Convertir en array numpy
                        try:
                            arr = np.array(first_result)
                            # Vérifier que c'est bien un array numpy et pas une liste
                            if not isinstance(arr, np.ndarray):
                                print(f"[DEBUG ONNX] Erreur: np.array() n'a pas créé un array numpy, type={type(arr)}")
                                arr = None
                            else:
                                # Vérifier que arr a l'attribut ndim avant d'y accéder
                                if hasattr(arr, 'ndim'):
                                    print(f"[DEBUG ONNX] Liste convertie en array: ndim={arr.ndim}, len={len(arr) if arr.ndim > 0 else 'N/A'}, valeurs={arr}")
                                else:
                                    print(f"[DEBUG ONNX] Liste convertie en array (pas d'attribut ndim): type={type(arr)}, valeurs={arr}")
                                    arr = None
                        
                            if arr is not None and isinstance(arr, np.ndarray) and arr.ndim == 1 and len(arr) == 4:
                                # Vérifier si ce sont des logits ou des probabilités
                                if np.any(arr < 0) or np.any(arr > 1):
                                    # Appliquer softmax
                                    exp_logits = np.exp(arr - np.max(arr))
                                    probs = exp_logits / np.sum(exp_logits)
                                else:
                                    probs = arr / np.sum(arr) if np.sum(arr) > 0 else arr
                                
                                predicted_class = int(np.argmax(probs))
                                angle_map = {0: 0, 1: 90, 2: 180, 3: 270}
                                angle = angle_map.get(predicted_class, 0)
                                confidence = float(np.max(probs))
                            elif arr is not None and isinstance(arr, np.ndarray) and arr.ndim == 2 and len(arr) > 0:
                                # Vérifier que arr[0] est accessible et a la bonne longueur
                                try:
                                    first_elem = arr[0]
                                    if isinstance(first_elem, np.ndarray) and len(first_elem) == 4:
                                        # Prendre le premier élément du batch
                                        arr = first_elem
                                    elif isinstance(first_elem, (list, tuple)) and len(first_elem) == 4:
                                        # Convertir la liste en array
                                        arr = np.array(first_elem)
                                    else:
                                        arr = None
                                except:
                                    arr = None
                                
                                if arr is not None and isinstance(arr, np.ndarray):
                                    if np.any(arr < 0) or np.any(arr > 1):
                                        exp_logits = np.exp(arr - np.max(arr))
                                        probs = exp_logits / np.sum(exp_logits)
                                    else:
                                        probs = arr / np.sum(arr) if np.sum(arr) > 0 else arr
                                    
                                    predicted_class = int(np.argmax(probs))
                                    angle_map = {0: 0, 1: 90, 2: 180, 3: 270}
                                    angle = angle_map.get(predicted_class, 0)
                                    confidence = float(np.max(probs))
                        except Exception as e:
                            print(f"[DEBUG ONNX] Erreur conversion liste en array: {e}")
                    # Si c'est un dictionnaire
                    elif isinstance(first_result, dict):
                        angle = int(first_result.get('orientation', first_result.get('angle', 0)))
                        confidence = float(first_result.get('confidence', first_result.get('prob', 1.0)))
                    # Si c'est un objet
                    elif hasattr(first_result, 'orientation'):
                        angle = int(first_result.orientation)
                        if hasattr(first_result, 'confidence'):
                            confidence = float(first_result.confidence)
                    # Si c'est directement un nombre
                    elif isinstance(first_result, (int, float)):
                        angle = int(first_result)
                        confidence = 1.0
                # Si c'est un array numpy (probabilités ou logits)
                elif isinstance(orientation_result, np.ndarray):
                    # Normaliser les dimensions
                    if hasattr(orientation_result, 'ndim') and orientation_result.ndim == 2:
                        orientation_result = orientation_result[0]  # Prendre le premier élément du batch
                        # Vérifier que orientation_result est toujours un array après l'indexation
                        if not isinstance(orientation_result, np.ndarray):
                            # Si ce n'est pas un array, essayer de le convertir
                            orientation_result = np.array(orientation_result)
                    
                    # Si c'est un array 1D avec 4 valeurs (probabilités pour 0°, 90°, 180°, 270°)
                    if isinstance(orientation_result, np.ndarray) and hasattr(orientation_result, 'ndim') and orientation_result.ndim == 1 and len(orientation_result) == 4:
                        # Vérifier si ce sont des logits (valeurs non normalisées) ou des probabilités
                        # Si les valeurs sont négatives ou > 1, ce sont probablement des logits
                        if np.any(orientation_result < 0) or np.any(orientation_result > 1):
                            # Appliquer softmax pour convertir les logits en probabilités
                            exp_logits = np.exp(orientation_result - np.max(orientation_result))  # Pour stabilité numérique
                            probs = exp_logits / np.sum(exp_logits)
                        else:
                            # Ce sont déjà des probabilités, normaliser pour être sûr
                            probs = orientation_result / np.sum(orientation_result) if np.sum(orientation_result) > 0 else orientation_result
                        
                        # Trouver la classe avec la probabilité maximale
                        predicted_class = int(np.argmax(probs))
                        # Convertir l'index de classe en angle
                        angle_map = {0: 0, 1: 90, 2: 180, 3: 270}
                        angle = angle_map.get(predicted_class, 0)
                        # La confiance est la probabilité maximale
                        confidence = float(np.max(probs))
                # Si c'est un dictionnaire
                elif isinstance(orientation_result, dict):
                    angle = int(orientation_result.get('orientation', orientation_result.get('angle', 0)))
                    confidence = float(orientation_result.get('confidence', orientation_result.get('prob', 1.0)))
                # Si c'est un objet avec attributs
                elif hasattr(orientation_result, 'orientation'):
                    angle = int(orientation_result.orientation)
                    if hasattr(orientation_result, 'confidence'):
                        confidence = float(orientation_result.confidence)
                elif hasattr(orientation_result, 'angle'):
                    angle = int(orientation_result.angle)
                    if hasattr(orientation_result, 'confidence'):
                        confidence = float(orientation_result.confidence)
                # Si c'est un objet Document (onnxtr peut retourner un objet avec des pages)
                elif hasattr(orientation_result, 'pages') and len(orientation_result.pages) > 0:
                    # Prendre la première page
                    first_page = orientation_result.pages[0]
                    if hasattr(first_page, 'orientation'):
                        angle = int(first_page.orientation)
                    elif isinstance(first_page, dict):
                        angle = int(first_page.get('orientation', first_page.get('angle', 0)))
                        confidence = float(first_page.get('confidence', first_page.get('prob', 1.0)))
                    # Chercher dans les attributs de la page
                    if hasattr(first_page, 'confidence'):
                        confidence = float(first_page.confidence)
                    elif hasattr(first_page, 'prob'):
                        confidence = float(first_page.prob)
                # Si c'est directement un nombre (angle)
                elif isinstance(orientation_result, (int, float)):
                    angle = int(orientation_result)
                    confidence = 1.0  # Pas de confiance disponible
                # Si aucun format reconnu, essayer d'accéder directement aux attributs
                else:
                    # Dernière tentative : chercher des attributs communs
                    if hasattr(orientation_result, 'orientation'):
                        angle = int(orientation_result.orientation)
                    if hasattr(orientation_result, 'confidence'):
                        confidence = float(orientation_result.confidence)
                    elif hasattr(orientation_result, 'prob'):
                        confidence = float(orientation_result.prob)
                    elif hasattr(orientation_result, 'probability'):
                        confidence = float(orientation_result.probability)
            
            # Normaliser l'angle à une valeur entre 0 et 360
            angle = angle % 360
            # Convertir en angle standard (0, 90, 180, 270)
            if angle not in [0, 90, 180, 270]:
                # Arrondir à l'angle le plus proche
                angle = min([0, 90, 180, 270], key=lambda x: abs(x - angle))
            
            # S'assurer que la confiance est dans [0, 1]
            confidence = max(0.0, min(1.0, confidence))
            
            return {
                'angle': angle,
                'needs_rotation': angle != 0,
                'confidence': confidence,
                'status': 'success'
            }
        except Exception as e:
            return {
                'angle': 0,
                'needs_rotation': False,
                'status': 'error',
                'error': str(e)
            }
    
    
    def intelligent_crop(self, image: np.ndarray, capture_type: Optional[str] = None) -> Tuple[np.ndarray, Dict]:
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
            
            # Découper et redresser
            metadata['crop_applied'] = True
            metadata['status'] = 'cropped'
            
            # Ordonner les points du polygone
            try:
                target_points = fit_quad_to_pts(abs_corners)
            except Exception:
                # Si fit_quad_to_pts échoue, utiliser les points directement
                target_points = abs_corners.reshape(4, 2)
            
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
        # Utiliser Canny pour détecter les bords
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
            confidence *= min(1.0, len(angles) / 20.0)
        
        confidence = float(np.clip(confidence, 0.0, 1.0))
        return float(median_angle), confidence
    
    def deskew_image(self, image: np.ndarray, angle: float, confidence: float = 0.0) -> Tuple[np.ndarray, Dict]:
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
    
    def process(self, img: np.ndarray, output_path: str, capture_type: str, original_input_path: str, capture_info: Optional[Dict] = None) -> Dict:
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
            
            if hasattr(self, '_current_output_original_path') and self._current_output_original_path:
                try:
                    # S'assurer que le répertoire de destination existe
                    ensure_dir(os.path.dirname(self._current_output_original_path))
                    cv2.imwrite(self._current_output_original_path, original_image)
                except Exception as e:
                    print(f"⚠️  Attention : Impossible de sauvegarder la copie de l'image source à {self._current_output_original_path}: {e}")

            # Initialiser le détecteur QA
            qa_detector = QADetector()
            
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
            
            print(f"[CAPTURE] Type reçu: {capture_type}")
            
            # Étape 1: Détection de Page (doctr detection.db_resnet50)
            # Étape 2: Découpage & Redressement (cv2.warpPerspective)
            crop_metadata = {}
            if self.enable_crop:
                img, crop_metadata = self.intelligent_crop(img, capture_type=capture_type)
                if crop_metadata.get('crop_applied', False):
                    # Enregistrer la transformation de crop avec tous les paramètres
                    crop_params = {
                        'area_ratio': crop_metadata.get('area_ratio', 1.0),
                        'status': crop_metadata.get('status', ''),
                    }
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
            print(f"[DEBUG] Avant détection orientation, fichier: {orientation_input}")
            orientation_result = self.detect_orientation(orientation_input)
            print(f"[DEBUG] Après détection orientation, résultat: {orientation_result}")
            
            # Nettoyer le fichier temporaire
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            
            if orientation_result['status'] == 'error':
                # En cas d'erreur d'orientation, sauvegarder l'image actuelle
                cv2.imwrite(output_path, img)
                
                # Calculer le temps de traitement même en cas d'erreur
                processing_time = time.time() - start_time
                
                # Détecter les flags QA même en cas d'erreur
                metadata_for_qa = {
                    'orientation_confidence': 0.0,  # Erreur = pas de confiance
                    'crop_metadata': crop_metadata,
                    'deskew_metadata': deskew_metadata,
                    'rotation_applied': False,
                    'angle': 0
                }
                
                qa_flags = qa_detector.detect_flags(
                    original_image=original_image,
                    final_image=img,
                    metadata=metadata_for_qa,
                    processing_time=processing_time
                )
                
                # Sauvegarder les flags QA même en cas d'erreur
                save_qa_flags(output_path, qa_flags)
                
                # Sauvegarder les transformations partielles
                save_transforms(output_path, transform_sequence)
                
                return {
                    'input_path': original_input_path,  # Utiliser le chemin original
                    'output_path': output_path,  # Chemin de l'image transformée (pour compatibilité)
                    'output_transformed_path': output_path,  # Chemin de l'image transformée
                    'transform_file': get_transform_file_path(output_path),
                    'qa_file': str(Path(output_path).with_suffix('.qa.json')),
                    'crop_applied': crop_metadata.get('crop_applied', False),
                    'crop_metadata': crop_metadata,
                    'deskew_applied': deskew_metadata.get('deskew_applied', False),
                    'deskew_metadata': deskew_metadata,
                    'orientation_detected': False,
                    'rotation_applied': False,
                    'qa_flags': qa_flags.to_dict(),
                    'transforms': transform_sequence.to_dict(),
                    'processing_time': processing_time,
                    'error': orientation_result.get('error'),
                    'status': 'error'
                }
            
            # Étape 6: Rotation Finale (cv2.rotate)
            rotation_applied = False
            rotation_angle = 0
            if orientation_result['needs_rotation']:
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
                        'rotation_type': 'standard' if angle in [90, 180, 270] else 'arbitrary'
                    },
                    order=3
                ))
            
            # Sauvegarder l'image finale
            cv2.imwrite(output_path, img)
            
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
            
            # Sauvegarder les flags QA
            try:
                save_qa_flags(output_path, qa_flags)
            except Exception as e:
                import traceback
                print(f"⚠️  Erreur lors de la sauvegarde des flags QA: {e}")
                print(f"   Traceback: {traceback.format_exc()}")
            
            # Sauvegarder la séquence de transformations
            try:
                save_transforms(output_path, transform_sequence)
            except Exception as e:
                import traceback
                print(f"⚠️  Erreur lors de la sauvegarde des transformations: {e}")
                print(f"   Traceback: {traceback.format_exc()}")
            
            return {
                'input_path': original_input_path,  # Utiliser le chemin original
                'output_path': output_path,  # Chemin de l'image transformée (pour compatibilité)
                'output_transformed_path': output_path,  # Chemin de l'image transformée
                'transform_file': get_transform_file_path(output_path),
                'qa_file': str(Path(output_path).with_suffix('.qa.json')),
                'crop_applied': crop_metadata.get('crop_applied', False),
                'crop_metadata': crop_metadata,
                'deskew_applied': deskew_metadata.get('deskew_applied', False),
                'deskew_angle': deskew_metadata.get('angle', 0.0),
                'deskew_metadata': deskew_metadata,
                'orientation_detected': True,
                'angle': orientation_result.get('angle', 0),
                'rotation_applied': rotation_applied,
                'transforms': transform_sequence.to_dict(),
                'qa_flags': qa_flags.to_dict(),
                'processing_time': processing_time,
                'status': 'success'
            }
        except Exception as e:
            return {
                'input_path': original_input_path,
                'output_path': output_path,  # Chemin de l'image transformée (pour compatibilité)
                'output_transformed_path': output_path,  # Chemin de l'image transformée
                'status': 'error',
                'error': str(e)
            }
    
    def process_batch(self, input_dir: str, output_dir: str) -> List[Dict]:
        """
        Traite un lot de documents depuis le répertoire preprocessing
        
        Args:
            input_dir: Répertoire contenant les images prétraitées (sortie de preprocessing)
            output_dir: Répertoire de sortie
            
        Returns:
            list: Liste des résultats pour chaque document
        """
        import json
        
        ensure_dir(output_dir)
        
        # Obtenir la liste des fichiers images à traiter
        image_extensions = ['.png', '.jpg', '.jpeg']
        files = get_files(input_dir, extensions=image_extensions)
        
        results = []
        for idx, file_path in enumerate(files):
            
            try:
                # Charger l'image prétraitée
                img = cv2.imread(file_path)
                if img is None:
                    raise ValueError(f"Impossible de charger l'image: {file_path}")
                
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
                        print(f"⚠️  Impossible de charger les métadonnées de {metadata_path}: {e}")
                
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
                
                # Ajouter les chemins dans le résultat
                result['output_original_path'] = output_original_path
                result['output_transformed_path'] = output_transformed_path
                results.append(result)
                
                # Vérifier que les fichiers ont été créés
                qa_file = Path(output_transformed_path).with_suffix('.qa.json')
                transform_file = Path(output_transformed_path).with_suffix('.transform.json')
                if not qa_file.exists():
                    print(f"⚠️  Fichier QA non créé pour {file_path}: {qa_file}")
                if not transform_file.exists():
                    print(f"⚠️  Fichier transformation non créé pour {file_path}: {transform_file}")
                    
            except Exception as e:
                print(f"❌ Erreur lors du traitement de {file_path}: {e}")
                # Générer les chemins même en cas d'erreur
                base_output = get_output_path(file_path, output_dir)
                path_obj = Path(base_output)
                output_original_path = str(path_obj.with_stem(f"{path_obj.stem}_original"))
                output_transformed_path = str(path_obj.with_stem(f"{path_obj.stem}_transformed"))
                
                results.append({
                    'input_path': file_path,
                    'output_path': output_transformed_path,  # Pour compatibilité
                    'output_original_path': output_original_path,
                    'output_transformed_path': output_transformed_path,
                    'status': 'error',
                    'error': str(e)
                })
        
        return results

