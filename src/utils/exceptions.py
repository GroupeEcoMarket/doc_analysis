"""
Exceptions personnalisées pour le pipeline de traitement de documents
"""


class PipelineError(Exception):
    """
    Erreur de base pour le pipeline de traitement de documents.
    Toutes les exceptions du pipeline héritent de cette classe.
    """
    pass


class PreprocessingError(PipelineError):
    """
    Erreur lors de l'étape de preprocessing.
    Se produit lors de l'amélioration du contraste, de la classification de capture, etc.
    """
    pass


class GeometryError(PipelineError):
    """
    Erreur lors de la normalisation géométrique.
    Se produit lors du crop, du deskew, de la détection d'orientation, etc.
    """
    pass


class ColometryError(PipelineError):
    """
    Erreur lors de la normalisation colométrique.
    """
    pass


class FeatureExtractionError(PipelineError):
    """
    Erreur lors de l'extraction de features (OCR, checkboxes, etc.).
    """
    pass


class ModelLoadingError(PipelineError):
    """
    Erreur lors du chargement d'un modèle ML.
    """
    pass


class ImageProcessingError(PipelineError):
    """
    Erreur lors du traitement d'une image (chargement, sauvegarde, etc.).
    """
    pass

