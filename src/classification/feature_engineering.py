"""
Feature Engineering pour la Classification de Documents

Ce module transforme les données OCR (JSON) en vecteurs numériques
multi-modaux (sémantique + positionnel) pour la classification.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

from src.pipeline.models import OCRLine, FeaturesOutput


class FeatureEngineer:
    """
    Transforme les données OCR en vecteurs numériques pour la classification.
    
    Cette classe gère :
    - Le filtrage des lignes OCR selon la confiance
    - La création d'embeddings sémantiques (via sentence-transformers)
    - La création d'embeddings positionnels (coordonnées normalisées)
    - L'agrégation des vecteurs en un vecteur unique par document
    """
    
    def __init__(
        self,
        semantic_model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        min_confidence: float = 0.70,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None
    ):
        """
        Initialise le FeatureEngineer.
        
        Args:
            semantic_model_name: Nom du modèle sentence-transformers à utiliser.
                               Par défaut, un modèle multilingue léger.
            min_confidence: Seuil de confiance minimum pour filtrer les lignes OCR.
            image_width: Largeur de l'image (pour normaliser les coordonnées).
                        Si None, sera calculé depuis les bounding boxes.
            image_height: Hauteur de l'image (pour normaliser les coordonnées).
                         Si None, sera calculé depuis les bounding boxes.
        """
        self.semantic_model_name = semantic_model_name
        self.semantic_model = SentenceTransformer(semantic_model_name)
        self.min_confidence = min_confidence
        self.image_width = image_width
        self.image_height = image_height
    
    def extract_document_embedding(
        self,
        ocr_data: Any,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None
    ) -> np.ndarray:
        """
        Extrait un vecteur d'embedding unique pour un document entier.
        
        Args:
            ocr_data: Données OCR. Peut être :
                     - FeaturesOutput (objet Pydantic)
                     - Dict avec clé 'ocr_lines'
                     - List[OCRLine] ou List[Dict]
            image_width: Largeur de l'image (override self.image_width si fourni).
            image_height: Hauteur de l'image (override self.image_height si fourni).
        
        Returns:
            np.ndarray: Vecteur d'embedding unique représentant le document.
        """
        # Normaliser l'entrée en liste d'OCRLine
        ocr_lines = self._normalize_ocr_input(ocr_data)
        
        # Filtrer les lignes selon la confiance
        filtered_lines = filter_ocr_lines(ocr_lines, self.min_confidence)
        
        if not filtered_lines:
            # Si aucune ligne valide, retourner un vecteur zéro
            # La taille dépend du modèle sémantique + 4 pour les coordonnées
            semantic_dim = self.semantic_model.get_sentence_embedding_dimension()
            return np.zeros(semantic_dim + 4)
        
        # Calculer les dimensions de l'image si non fournies
        width, height = self._compute_image_dimensions(
            filtered_lines, image_width, image_height
        )
        
        # Créer les embeddings multi-modaux pour chaque ligne
        line_embeddings = []
        line_confidences = []
        
        for line in filtered_lines:
            embedding = create_multimodal_embedding(
                line,
                self.semantic_model,
                width,
                height
            )
            line_embeddings.append(embedding)
            line_confidences.append(line.confidence)
        
        # Agréger les embeddings en un seul vecteur
        document_embedding = aggregate_line_embeddings(
            line_embeddings,
            line_confidences
        )
        
        return document_embedding
    
    def _normalize_ocr_input(self, ocr_data: Any) -> List[OCRLine]:
        """
        Normalise différentes formes d'entrée OCR en List[OCRLine].
        
        Args:
            ocr_data: Données OCR sous différentes formes.
        
        Returns:
            List[OCRLine]: Liste normalisée d'objets OCRLine.
        """
        if isinstance(ocr_data, FeaturesOutput):
            return ocr_data.ocr_lines
        elif isinstance(ocr_data, dict):
            if 'ocr_lines' in ocr_data:
                lines = ocr_data['ocr_lines']
            elif 'lines' in ocr_data:
                lines = ocr_data['lines']
            else:
                lines = ocr_data
        elif isinstance(ocr_data, list):
            lines = ocr_data
        else:
            raise ValueError(
                f"Format d'entrée OCR non supporté: {type(ocr_data)}. "
                "Attendu: FeaturesOutput, Dict avec 'ocr_lines', ou List[OCRLine/Dict]"
            )
        
        # Convertir les dicts en OCRLine si nécessaire
        normalized = []
        for line in lines:
            if isinstance(line, OCRLine):
                normalized.append(line)
            elif isinstance(line, dict):
                normalized.append(OCRLine(**line))
            else:
                raise ValueError(f"Format de ligne OCR non supporté: {type(line)}")
        
        return normalized
    
    def _compute_image_dimensions(
        self,
        ocr_lines: List[OCRLine],
        provided_width: Optional[int],
        provided_height: Optional[int]
    ) -> Tuple[int, int]:
        """
        Calcule les dimensions de l'image depuis les bounding boxes si non fournies.
        
        Args:
            ocr_lines: Liste des lignes OCR.
            provided_width: Largeur fournie (si disponible).
            provided_height: Hauteur fournie (si disponible).
        
        Returns:
            Tuple[int, int]: (width, height)
        """
        if provided_width is not None and provided_height is not None:
            return (provided_width, provided_height)
        
        if self.image_width is not None and self.image_height is not None:
            return (self.image_width, self.image_height)
        
        # Calculer depuis les bounding boxes
        max_x = 0
        max_y = 0
        
        for line in ocr_lines:
            bbox = line.bounding_box
            
            # Gérer différents formats de bounding_box
            if isinstance(bbox, list):
                if len(bbox) == 4:
                    # Format [x1, y1, x2, y2]
                    x1, y1, x2, y2 = bbox
                    max_x = max(max_x, x1, x2)
                    max_y = max(max_y, y1, y2)
                elif len(bbox) >= 4:
                    # Format avec 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    for point in bbox[:4]:
                        if isinstance(point, (list, tuple)) and len(point) >= 2:
                            max_x = max(max_x, point[0])
                            max_y = max(max_y, point[1])
        
        # Ajouter une marge de sécurité (10%)
        width = int(max_x * 1.1) if max_x > 0 else 2000
        height = int(max_y * 1.1) if max_y > 0 else 2000
        
        return (width, height)


def filter_ocr_lines(
    ocr_lines: List[OCRLine],
    min_confidence: float = 0.70
) -> List[OCRLine]:
    """
    Filtre les lignes OCR selon leur confiance.
    
    Args:
        ocr_lines: Liste des lignes OCR à filtrer.
        min_confidence: Seuil de confiance minimum.
    
    Returns:
        List[OCRLine]: Lignes filtrées avec confiance >= min_confidence.
    """
    return [
        line for line in ocr_lines
        if line.confidence >= min_confidence
    ]


def create_multimodal_embedding(
    ocr_line: OCRLine,
    semantic_model: SentenceTransformer,
    image_width: int,
    image_height: int
) -> np.ndarray:
    """
    Crée un embedding multi-modal pour une ligne OCR.
    
    Concatène :
    - Embedding sémantique (via sentence-transformers)
    - Embedding positionnel (4 coordonnées normalisées)
    
    Args:
        ocr_line: Ligne OCR à transformer.
        semantic_model: Modèle sentence-transformers pour l'embedding sémantique.
        image_width: Largeur de l'image (pour normalisation).
        image_height: Hauteur de l'image (pour normalisation).
    
    Returns:
        np.ndarray: Vecteur concaténé [semantic_embedding, positional_embedding]
    """
    # 1. Embedding sémantique
    semantic_embedding = semantic_model.encode(
        ocr_line.text,
        convert_to_numpy=True,
        normalize_embeddings=True, # Normaliser pour améliorer la qualité
        show_progress_bar=False
    )
    
    # 2. Embedding positionnel (coordonnées normalisées)
    positional_embedding = extract_positional_features(
        ocr_line.bounding_box,
        image_width,
        image_height
    )
    
    # 3. Concaténation
    multimodal_embedding = np.concatenate([
        semantic_embedding,
        positional_embedding
    ])
    
    return multimodal_embedding


def extract_positional_features(
    bounding_box: List[Any],
    image_width: int,
    image_height: int
) -> np.ndarray:
    """
    Extrait et normalise les features positionnelles depuis un bounding box.
    
    Args:
        bounding_box: Boîte englobante. Formats supportés :
                     - [x1, y1, x2, y2] (4 floats)
                     - [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] (4 points)
        image_width: Largeur de l'image.
        image_height: Hauteur de l'image.
    
    Returns:
        np.ndarray: Vecteur de 4 features normalisées [x1_norm, y1_norm, x2_norm, y2_norm]
    """
    # Normaliser le format du bounding box
    if len(bounding_box) == 4:
        if isinstance(bounding_box[0], (int, float)):
            # Format [x1, y1, x2, y2]
            x1, y1, x2, y2 = bounding_box
        elif isinstance(bounding_box[0], (list, tuple)):
            # Format avec 4 points, extraire le rectangle englobant
            points = bounding_box
            x_coords = [p[0] if isinstance(p, (list, tuple)) else p for p in points]
            y_coords = [p[1] if isinstance(p, (list, tuple)) and len(p) > 1 else 0 for p in points]
            x1, x2 = min(x_coords), max(x_coords)
            y1, y2 = min(y_coords), max(y_coords)
        else:
            raise ValueError(f"Format de bounding_box non reconnu: {type(bounding_box[0])}")
    else:
        raise ValueError(f"Bounding box doit avoir 4 éléments, reçu: {len(bounding_box)}")
    
    # Normaliser les coordonnées entre 0 et 1
    x1_norm = x1 / image_width if image_width > 0 else 0.0
    y1_norm = y1 / image_height if image_height > 0 else 0.0
    x2_norm = x2 / image_width if image_width > 0 else 0.0
    y2_norm = y2 / image_height if image_height > 0 else 0.0
    
    return np.array([x1_norm, y1_norm, x2_norm, y2_norm], dtype=np.float32)


def aggregate_line_embeddings(
    line_embeddings: List[np.ndarray],
    line_confidences: Optional[List[float]] = None,
    aggregation_method: str = 'weighted_mean'
) -> np.ndarray:
    """
    Agrége les embeddings de toutes les lignes en un seul vecteur pour le document.
    
    Args:
        line_embeddings: Liste des embeddings de chaque ligne.
        line_confidences: Liste des confiances (pour pondération). Si None, utilise moyenne simple.
        aggregation_method: Méthode d'agrégation :
                           - 'weighted_mean': Moyenne pondérée par confiance
                           - 'mean': Moyenne simple
                           - 'max': Maximum par dimension
                           - 'sum': Somme
    
    Returns:
        np.ndarray: Vecteur d'embedding unique pour le document.
    """
    if not line_embeddings:
        raise ValueError("La liste d'embeddings ne peut pas être vide")
    
    embeddings_array = np.array(line_embeddings)
    
    if aggregation_method == 'weighted_mean':
        if line_confidences is None:
            # Fallback sur moyenne simple si pas de confiances
            return np.mean(embeddings_array, axis=0)
        
        # Normaliser les confiances pour qu'elles somment à 1
        weights = np.array(line_confidences, dtype=np.float32)
        weights = weights / (weights.sum() + 1e-8)  # Éviter division par zéro
        
        # Moyenne pondérée
        aggregated = np.average(embeddings_array, axis=0, weights=weights)
    
    elif aggregation_method == 'mean':
        aggregated = np.mean(embeddings_array, axis=0)
    
    elif aggregation_method == 'max':
        aggregated = np.max(embeddings_array, axis=0)
    
    elif aggregation_method == 'sum':
        aggregated = np.sum(embeddings_array, axis=0)
    
    else:
        raise ValueError(
            f"Méthode d'agrégation inconnue: {aggregation_method}. "
            "Choix: 'weighted_mean', 'mean', 'max', 'sum'"
        )
    
    return aggregated.astype(np.float32)


# Fonction de convenance pour usage direct
def extract_document_embedding(
    ocr_data: Any,
    semantic_model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    min_confidence: float = 0.70,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None
) -> np.ndarray:
    """
    Fonction de convenance pour extraire un embedding de document.
    
    Args:
        ocr_data: Données OCR (FeaturesOutput, Dict, ou List[OCRLine]).
        semantic_model_name: Nom du modèle sentence-transformers.
        min_confidence: Seuil de confiance minimum.
        image_width: Largeur de l'image (optionnel).
        image_height: Hauteur de l'image (optionnel).
    
    Returns:
        np.ndarray: Vecteur d'embedding unique pour le document.
    """
    engineer = FeatureEngineer(
        semantic_model_name=semantic_model_name,
        min_confidence=min_confidence,
        image_width=image_width,
        image_height=image_height
    )
    return engineer.extract_document_embedding(ocr_data)

