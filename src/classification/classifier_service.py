"""
Service de Classification de Documents

Ce module contient la logique principale de classification de type de document.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import joblib
from pathlib import Path

from src.classification.feature_engineering import FeatureEngineer
from src.pipeline.models import FeaturesOutput
from src.utils.config_loader import get_config, Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentClassifier:
    """
    Service principal de classification de documents.
    
    Utilise les embeddings multi-modaux (sémantique + positionnel)
    pour classifier le type de document via un modèle ML entraîné.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        semantic_model_name: Optional[str] = None,
        min_confidence: Optional[float] = None,
        app_config: Optional[Config] = None
    ):
        """
        Initialise le classifieur de documents.
        
        Args:
            model_path: Chemin vers le modèle ML sauvegardé (joblib).
                      Si None, sera chargé depuis config.yaml.
            semantic_model_name: Nom du modèle sentence-transformers à utiliser.
                               Si None, sera chargé depuis config.yaml.
            min_confidence: Seuil de confiance minimum pour filtrer les lignes OCR.
                          Si None, sera chargé depuis config.yaml.
            app_config: Configuration de l'application (injectée via DI).
                       Si None, charge la config depuis config.yaml (pour compatibilité).
        """
        # Charger la configuration
        if app_config is not None:
            config_dict = app_config.get('classification', {})
        else:
            config_obj = get_config()
            config_dict = config_obj.get('classification', {})
        
        # Utiliser les paramètres fournis ou ceux de la config
        self.model_path = model_path or config_dict.get('model_path', 'models/document_classifier.joblib')
        # Support pour 'embedding_model' (nouveau) et 'semantic_model_name' (ancien, pour compatibilité)
        self.semantic_model_name = semantic_model_name or config_dict.get(
            'embedding_model',
            config_dict.get('semantic_model_name', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        )
        self.min_confidence = min_confidence or config_dict.get('min_confidence', 0.70)
        self.classification_confidence_threshold = config_dict.get('classification_confidence_threshold', 0.60)
        
        # Initialiser le feature engineer
        self.feature_engineer = FeatureEngineer(
            semantic_model_name=self.semantic_model_name,
            min_confidence=self.min_confidence
        )
        
        # Charger le modèle ML
        self.model = None
        self.class_names: Optional[List[str]] = None
        self._load_model()
    
    def _load_model(self) -> None:
        """
        Charge le modèle ML depuis le fichier joblib.
        
        Raises:
            FileNotFoundError: Si le fichier du modèle n'existe pas.
            ValueError: Si le modèle ne peut pas être chargé.
        """
        # Résoudre le chemin (relatif ou absolu)
        model_path = Path(self.model_path)
        if not model_path.is_absolute():
            # Si chemin relatif, résoudre depuis la racine du projet
            project_root = Path(__file__).parent.parent.parent
            model_path = project_root / model_path
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modèle de classification introuvable: {model_path}\n"
                f"Veuillez entraîner et sauvegarder un modèle à cet emplacement."
            )
        
        try:
            # Charger le modèle avec joblib
            # Le modèle peut être :
            # - Un modèle seul (sklearn, lightgbm, etc.)
            # - Un dict avec 'model' et 'class_names' (recommandé)
            loaded_data = joblib.load(model_path)
            
            if isinstance(loaded_data, dict):
                self.model = loaded_data.get('model')
                self.class_names = loaded_data.get('class_names')
                if self.model is None:
                    raise ValueError(
                        f"Le fichier {model_path} contient un dict mais pas de clé 'model'"
                    )
            else:
                # Modèle seul, sans noms de classes
                self.model = loaded_data
                self.class_names = None
                logger.warning(
                    f"Le modèle {model_path} ne contient pas de noms de classes. "
                    "Les prédictions retourneront des indices numériques."
                )
            
            logger.info(f"Modèle de classification chargé depuis: {model_path}")
            if self.class_names:
                logger.info(f"Classes disponibles: {', '.join(self.class_names)}")
        
        except Exception as e:
            raise ValueError(
                f"Impossible de charger le modèle depuis {model_path}: {e}"
            ) from e
    
    def predict(self, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prédit le type de document depuis les résultats OCR.
        
        Args:
            ocr_result: Dictionnaire contenant les résultats OCR.
                       Peut être :
                       - FeaturesOutput (converti en dict)
                       - Dict avec clé 'ocr_lines'
                       - Dict avec clé 'features' contenant 'ocr_lines'
        
        Returns:
            Dict contenant :
            - document_type: Type de document détecté (str)
            - confidence: Confiance de la classification (float entre 0.0 et 1.0)
        """
        if self.model is None:
            raise ValueError(
                "Le modèle n'a pas été chargé. Vérifiez que le chemin du modèle est correct."
            )
        
        # Normaliser l'entrée OCR
        ocr_data = self._normalize_ocr_input(ocr_result)
        
        # Extraire l'embedding du document
        document_embedding = self.feature_engineer.extract_document_embedding(ocr_data)
        
        # Prédire avec le modèle
        # Reshape pour sklearn (1 sample, n features)
        embedding_2d = document_embedding.reshape(1, -1)
        
        # Détecter le type de modèle pour gérer les prédictions correctement
        is_lightgbm = hasattr(self.model, 'predict') and not hasattr(self.model, 'predict_proba')
        
        # Probabilités et prédiction
        confidence = 0.0
        probabilities = None
        
        if hasattr(self.model, 'predict_proba'):
            # Modèles sklearn standard
            probabilities = self.model.predict_proba(embedding_2d)[0]
            prediction = self.model.predict(embedding_2d)[0]
            confidence = float(np.max(probabilities))
        elif is_lightgbm:
            # LightGBM: predict retourne les probabilités pour chaque classe
            # Format: array de shape (n_samples, n_classes)
            probabilities = self.model.predict(embedding_2d)
            if probabilities.ndim == 2:
                probabilities = probabilities[0]
            # Prédiction = classe avec probabilité maximale
            prediction = int(np.argmax(probabilities))
            confidence = float(np.max(probabilities))
        else:
            # Modèle sans probabilités
            prediction = self.model.predict(embedding_2d)[0]
            confidence = 1.0
        
        # Convertir la prédiction en nom de classe si disponible
        if self.class_names is not None:
            prediction_idx = int(prediction)
            if 0 <= prediction_idx < len(self.class_names):
                document_type = self.class_names[prediction_idx]
            else:
                logger.warning(
                    f"Index de prédiction {prediction_idx} hors limites. "
                    f"Classes disponibles: {len(self.class_names)}"
                )
                document_type = str(prediction)
        else:
            document_type = str(prediction)
        
        # Vérifier le seuil de confiance
        if confidence < self.classification_confidence_threshold:
            logger.warning(
                f"Confiance de classification ({confidence:.3f}) en dessous du seuil "
                f"({self.classification_confidence_threshold}). Prédiction: {document_type}"
            )
            # Retourner None pour document_type si confiance trop faible
            # ou garder la prédiction mais marquer comme incertaine
            document_type = None
        
        return {
            'document_type': document_type,
            'confidence': confidence
        }
    
    def _normalize_ocr_input(self, ocr_result: Dict[str, Any]) -> Any:
        """
        Normalise différentes formes d'entrée OCR.
        
        Args:
            ocr_result: Résultat OCR sous différentes formes.
        
        Returns:
            Données OCR normalisées (FeaturesOutput, Dict, ou List[OCRLine]).
        """
        # Si c'est déjà un FeaturesOutput
        if isinstance(ocr_result, FeaturesOutput):
            return ocr_result
        
        # Si c'est un dict
        if isinstance(ocr_result, dict):
            # Chercher 'ocr_lines' directement
            if 'ocr_lines' in ocr_result:
                return ocr_result
            
            # Chercher dans 'features'
            if 'features' in ocr_result:
                features = ocr_result['features']
                if isinstance(features, dict) and 'ocr_lines' in features:
                    return features
            
            # Si le dict contient directement les lignes (compatibilité)
            if 'lines' in ocr_result:
                return {'ocr_lines': ocr_result['lines']}
        
        # Retourner tel quel (sera géré par FeatureEngineer)
        return ocr_result

