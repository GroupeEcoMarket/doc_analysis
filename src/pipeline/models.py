"""
Modèles de données pour le pipeline de traitement de documents.

Ces modèles Pydantic définissent les "contrats" explicites entre les étapes du pipeline,
rendant les interfaces claires et réduisant les erreurs liées aux dictionnaires.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum


class CaptureType(str, Enum):
    """Type de capture d'un document."""
    SCAN = "SCAN"
    PHOTO = "PHOTO"


class CaptureInfo(BaseModel):
    """
    Informations sur la classification du type de capture (SCAN vs PHOTO).
    """
    type: CaptureType = Field(..., description="Type de capture détecté")
    white_percentage: float = Field(..., ge=0.0, le=1.0, description="Pourcentage de pixels blancs (0.0-1.0)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confiance de la classification (0.0-1.0)")
    enabled: bool = Field(..., description="Si la classification était activée")
    reason: str = Field(..., description="Raison de la classification")
    white_level_threshold: Optional[int] = Field(None, ge=0, le=255, description="Seuil de niveau de blanc utilisé")
    white_percentage_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Seuil de pourcentage de blanc utilisé")
    error: Optional[str] = Field(None, description="Message d'erreur si la classification a échoué")
    
    model_config = {
        "use_enum_values": True
    }


class PreprocessingOutput(BaseModel):
    """
    Résultat de l'étape de prétraitement.
    
    Ce modèle définit le contrat de sortie de l'étape preprocessing,
    qui est ensuite utilisé comme entrée pour l'étape geometry.
    """
    status: str = Field(..., description="Statut du traitement: 'success' ou 'error'")
    input_path: str = Field(..., description="Chemin vers le fichier d'entrée original")
    processed_path: str = Field(..., description="Chemin vers l'image prétraitée sauvegardée")
    capture_type: CaptureType = Field(..., description="Type de capture détecté")
    capture_info: CaptureInfo = Field(..., description="Informations complètes sur la classification")
    processing_time: float = Field(..., ge=0.0, description="Temps de traitement en secondes")
    error: Optional[str] = Field(None, description="Message d'erreur si le traitement a échoué")
    
    model_config = {
        "use_enum_values": True
    }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreprocessingOutput":
        """
        Crée une instance depuis un dictionnaire (pour compatibilité avec l'ancien code).
        
        Args:
            data: Dictionnaire avec les données de preprocessing
            
        Returns:
            PreprocessingOutput: Instance du modèle
        """
        # Convertir capture_info en CaptureInfo si c'est un dict
        if isinstance(data.get('capture_info'), dict):
            data['capture_info'] = CaptureInfo(**data['capture_info'])
        
        # Convertir capture_type en CaptureType si c'est une string
        if isinstance(data.get('capture_type'), str):
            data['capture_type'] = CaptureType(data['capture_type'])
        
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit le modèle en dictionnaire (pour compatibilité avec l'ancien code).
        
        Returns:
            dict: Dictionnaire avec les données
        """
        return self.model_dump()


class CropMetadata(BaseModel):
    """Métadonnées de l'opération de crop."""
    crop_applied: bool = Field(..., description="Si le crop a été appliqué")
    area_ratio: float = Field(..., ge=0.0, le=1.0, description="Ratio de l'aire du document par rapport à l'image")
    status: str = Field(..., description="Statut du crop: 'cropped', 'skipped', 'already_cropped', etc.")
    reason: Optional[str] = Field(None, description="Raison si le crop a été ignoré")
    transform_matrix: Optional[List[List[float]]] = Field(None, description="Matrice de transformation appliquée")
    source_points: Optional[List[List[float]]] = Field(None, description="Points sources du polygone")
    destination_points: Optional[List[List[float]]] = Field(None, description="Points de destination")
    output_size: Optional[List[int]] = Field(None, description="Taille de l'image de sortie [width, height]")
    error: Optional[str] = Field(None, description="Message d'erreur si le crop a échoué")


class DeskewMetadata(BaseModel):
    """Métadonnées de l'opération de deskew (correction d'inclinaison)."""
    deskew_applied: bool = Field(..., description="Si le deskew a été appliqué")
    angle: float = Field(..., description="Angle d'inclinaison détecté en degrés")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confiance de la détection d'angle")
    status: str = Field(..., description="Statut: 'success', 'skipped', 'angle_too_small', 'low_confidence', etc.")
    reason: Optional[str] = Field(None, description="Raison si le deskew a été ignoré")
    transform_matrix: Optional[List[List[float]]] = Field(None, description="Matrice de transformation appliquée")
    center: Optional[List[float]] = Field(None, description="Centre de rotation [x, y]")
    output_size: Optional[List[int]] = Field(None, description="Taille de l'image de sortie [width, height]")
    original_size: Optional[List[int]] = Field(None, description="Taille de l'image originale [width, height]")
    error: Optional[str] = Field(None, description="Message d'erreur si le deskew a échoué")


class GeometryOutput(BaseModel):
    """
    Résultat de l'étape de normalisation géométrique.
    
    Ce modèle définit le contrat de sortie de l'étape geometry,
    qui contient toutes les informations sur les transformations appliquées.
    """
    status: str = Field(..., description="Statut du traitement: 'success' ou 'error'")
    input_path: str = Field(..., description="Chemin vers le fichier d'entrée original")
    output_path: str = Field(..., description="Chemin vers l'image transformée (pour compatibilité)")
    output_transformed_path: str = Field(..., description="Chemin vers l'image transformée")
    output_original_path: Optional[str] = Field(None, description="Chemin vers l'image originale sauvegardée")
    transform_file: str = Field(..., description="Chemin vers le fichier JSON des transformations")
    qa_file: str = Field(..., description="Chemin vers le fichier JSON des flags QA")
    
    # Métadonnées des transformations
    crop_applied: bool = Field(..., description="Si le crop intelligent a été appliqué")
    crop_metadata: CropMetadata = Field(..., description="Métadonnées du crop")
    deskew_applied: bool = Field(..., description="Si la correction d'inclinaison a été appliquée")
    deskew_angle: float = Field(..., description="Angle d'inclinaison détecté en degrés")
    deskew_metadata: DeskewMetadata = Field(..., description="Métadonnées du deskew")
    orientation_detected: bool = Field(..., description="Si l'orientation a été détectée")
    angle: int = Field(..., description="Angle d'orientation détecté (0, 90, 180, 270)")
    rotation_applied: bool = Field(..., description="Si la rotation a été appliquée")
    
    # Données supplémentaires
    transforms: Dict[str, Any] = Field(..., description="Séquence complète des transformations appliquées")
    qa_flags: Dict[str, Any] = Field(..., description="Flags de qualité (QA)")
    processing_time: float = Field(..., ge=0.0, description="Temps de traitement en secondes")
    error: Optional[str] = Field(None, description="Message d'erreur si le traitement a échoué")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeometryOutput":
        """
        Crée une instance depuis un dictionnaire (pour compatibilité avec l'ancien code).
        
        Args:
            data: Dictionnaire avec les données de geometry
            
        Returns:
            GeometryOutput: Instance du modèle
        """
        # Convertir crop_metadata en CropMetadata si c'est un dict
        if isinstance(data.get('crop_metadata'), dict):
            data['crop_metadata'] = CropMetadata(**data['crop_metadata'])
        
        # Convertir deskew_metadata en DeskewMetadata si c'est un dict
        if isinstance(data.get('deskew_metadata'), dict):
            data['deskew_metadata'] = DeskewMetadata(**data['deskew_metadata'])
        
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit le modèle en dictionnaire (pour compatibilité avec l'ancien code).
        
        Returns:
            dict: Dictionnaire avec les données
        """
        return self.model_dump()


class ColometryOutput(BaseModel):
    """
    Résultat de l'étape de normalisation colométrique.
    
    Ce modèle définit le contrat de sortie de l'étape colometry,
    qui normalise la structure des colonnes d'un document.
    """
    status: str = Field(..., description="Statut du traitement: 'success' ou 'error'")
    input_path: str = Field(..., description="Chemin vers le fichier d'entrée original")
    output_path: str = Field(..., description="Chemin vers le document normalisé")
    processing_time: float = Field(..., ge=0.0, description="Temps de traitement en secondes")
    error: Optional[str] = Field(None, description="Message d'erreur si le traitement a échoué")
    
    # Métadonnées de colométrie (à définir selon l'implémentation)
    columns_detected: Optional[int] = Field(None, description="Nombre de colonnes détectées")
    column_structure: Optional[Dict[str, Any]] = Field(None, description="Structure des colonnes détectées")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ColometryOutput":
        """
        Crée une instance depuis un dictionnaire (pour compatibilité avec l'ancien code).
        
        Args:
            data: Dictionnaire avec les données de colometry
            
        Returns:
            ColometryOutput: Instance du modèle
        """
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit le modèle en dictionnaire (pour compatibilité avec l'ancien code).
        
        Returns:
            dict: Dictionnaire avec les données
        """
        return self.model_dump()


class OCRLine(BaseModel):
    """Représente une ligne de texte détectée par OCR."""
    text: str = Field(..., description="Texte de la ligne")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confiance de la reconnaissance (0.0-1.0)")
    bounding_box: List[float] = Field(..., min_length=4, max_length=4, description="Boîte englobante [x1, y1, x2, y2]")


class CheckboxDetection(BaseModel):
    """Représente une checkbox détectée."""
    position: List[float] = Field(..., min_length=4, max_length=4, description="Position [x1, y1, x2, y2]")
    checked: bool = Field(..., description="Si la checkbox est cochée")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confiance de la détection (0.0-1.0)")


class FeaturesOutput(BaseModel):
    """
    Résultat de l'étape d'extraction de features.
    
    Ce modèle définit le contrat de sortie de l'étape features,
    qui extrait les features d'un document (OCR, checkboxes, etc.).
    """
    status: str = Field(..., description="Statut du traitement: 'success' ou 'error'")
    input_path: str = Field(..., description="Chemin vers le fichier d'entrée original")
    output_path: str = Field(..., description="Chemin vers le fichier de sortie des features")
    processing_time: float = Field(..., ge=0.0, description="Temps de traitement en secondes")
    error: Optional[str] = Field(None, description="Message d'erreur si le traitement a échoué")
    
    # Features extraites
    ocr_lines: List[OCRLine] = Field(default_factory=list, description="Lignes de texte détectées par OCR")
    checkboxes: List[CheckboxDetection] = Field(default_factory=list, description="Checkboxes détectées")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeaturesOutput":
        """
        Crée une instance depuis un dictionnaire (pour compatibilité avec l'ancien code).
        
        Args:
            data: Dictionnaire avec les données de features
            
        Returns:
            FeaturesOutput: Instance du modèle
        """
        # Convertir ocr_lines en OCRLine si ce sont des dicts
        if 'ocr_lines' in data and data['ocr_lines']:
            if isinstance(data['ocr_lines'][0], dict):
                data['ocr_lines'] = [OCRLine(**line) for line in data['ocr_lines']]
        
        # Convertir checkboxes en CheckboxDetection si ce sont des dicts
        if 'checkboxes' in data and data['checkboxes']:
            if isinstance(data['checkboxes'][0], dict):
                data['checkboxes'] = [CheckboxDetection(**box) for box in data['checkboxes']]
        
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit le modèle en dictionnaire (pour compatibilité avec l'ancien code).
        
        Returns:
            dict: Dictionnaire avec les données
        """
        return self.model_dump()
