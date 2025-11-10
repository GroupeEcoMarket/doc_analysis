"""
Étape de Prétraitement du Pipeline
- Amélioration du contraste
- Classification du type de capture (SCAN/PHOTO)
"""
import os
import json
import cv2
from pathlib import Path
from typing import Dict, List
import time

from src.utils.file_handler import ensure_dir, get_files, get_output_path
from src.utils.pdf_handler import is_pdf, pdf_to_images
from src.utils.capture_classifier import CaptureClassifier
from src.utils.config_loader import get_config, PDFConfig, GeometryConfig
from src.utils.image_enhancer import enhance_contrast_clahe
from src.utils.logger import get_logger
from src.utils.exceptions import PreprocessingError, ImageProcessingError
from src.pipeline.models import PreprocessingOutput, CaptureInfo, CaptureType
from typing import Optional

logger = get_logger(__name__)


class PreprocessingNormalizer:
    """
    Applique les prétraitements à une image avant les étapes géométriques.
    """

    def __init__(
        self, 
        capture_classifier: Optional[CaptureClassifier] = None,
        pdf_config: Optional[PDFConfig] = None
    ):
        """
        Initialise le normaliseur de prétraitement.
        
        Args:
            capture_classifier: Classificateur de capture (injecté via DI).
                              Si None, crée un classificateur avec la config par défaut
                              (pour compatibilité avec l'ancien code).
            pdf_config: Configuration PDF (injectée via DI).
                       Si None, charge depuis config.yaml (pour compatibilité).
        """
        if capture_classifier is None:
            # Fallback pour compatibilité : charger la config si non fournie
            app_config = get_config()
            geo_config = app_config.geometry
            self.capture_classifier = CaptureClassifier(
                white_level_threshold=geo_config.capture_classifier_white_level_threshold,
                white_percentage_threshold=geo_config.capture_classifier_white_percentage_threshold,
                enabled=geo_config.capture_classifier_enabled
            )
        else:
            self.capture_classifier = capture_classifier
        
        # Charger la configuration PDF si non fournie
        if pdf_config is None:
            app_config = get_config()
            self.pdf_config = app_config.pdf
        else:
            self.pdf_config = pdf_config

    def process(self, input_path: str, output_path: str) -> PreprocessingOutput:
        """
        Charge une image, améliore son contraste et classifie son type.
        Sauvegarde l'image prétraitée et retourne les métadonnées.
        
        Args:
            input_path: Chemin vers le document d'entrée (PDF ou image)
            output_path: Chemin de sortie pour l'image prétraitée
            
        Returns:
            PreprocessingOutput: Métadonnées structurées du prétraitement
        """
        start_time = time.time()
        
        if is_pdf(input_path):
            # Utiliser pdf_config.dpi et pdf_config.min_dpi pour la conversion
            images = pdf_to_images(input_path, dpi=self.pdf_config.dpi, min_dpi=self.pdf_config.min_dpi)
            if not images:
                raise PreprocessingError(f"Aucune page trouvée dans le PDF: {input_path}")
            img = images[0]
        else:
            img = cv2.imread(input_path)
            if img is None:
                raise ImageProcessingError(f"Impossible de charger l'image: {input_path}")

        # 1. Amélioration du contraste
        img_enhanced = enhance_contrast_clahe(img)

        # 2. Classification du type de capture
        capture_info_dict = self.capture_classifier.classify(img_enhanced)
        capture_info = CaptureInfo(**capture_info_dict)
        
        # 3. Sauvegarde de l'image prétraitée
        ensure_dir(os.path.dirname(output_path))
        cv2.imwrite(output_path, img_enhanced)
        
        processing_time = time.time() - start_time

        # 4. Créer le modèle de sortie
        output = PreprocessingOutput(
            status='success',
            input_path=input_path,
            processed_path=output_path,
            capture_type=capture_info.type,
            capture_info=capture_info,
            processing_time=processing_time
        )
        
        # 5. Sauvegarder les métadonnées dans un fichier JSON pour l'étape suivante
        metadata_path = Path(output_path).with_suffix('.json')
        try:
            with open(metadata_path, 'w') as f:
                json.dump(output.to_dict(), f, indent=2, default=str)
        except Exception as e:
            logger.warning("Impossible de sauvegarder les métadonnées", exc_info=True)

        # 6. Retourne les métadonnées structurées pour l'étape suivante
        return output

    def process_batch(self, input_dir: str, output_dir: str) -> List[PreprocessingOutput]:
        """
        Traite un lot de documents.
        Pour les PDFs multi-pages, traite chaque page séparément.
        
        Args:
            input_dir: Répertoire contenant les documents d'entrée
            output_dir: Répertoire de sortie
            
        Returns:
            list: Liste des résultats pour chaque document/page
        """
        ensure_dir(output_dir)
        files = get_files(input_dir, extensions=['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.pdf'])
        results = []
        
        for file_path in files:
            if is_pdf(file_path):
                # Traiter chaque page du PDF séparément
                # Utiliser pdf_config.dpi et pdf_config.min_dpi pour la conversion
                pdf_images = pdf_to_images(file_path, dpi=self.pdf_config.dpi, min_dpi=self.pdf_config.min_dpi)
                if not pdf_images:
                    logger.warning(f"Aucune page trouvée dans le PDF: {file_path}")
                    continue
                
                # Générer le nom de base pour les fichiers de sortie
                base_name = Path(file_path).stem
                base_output = get_output_path(file_path, output_dir)
                base_output_path = Path(base_output).parent / base_name
                
                for page_num, pdf_img in enumerate(pdf_images):
                    # Générer le nom de fichier pour cette page
                    # Format: document_original_page_1.png, document_original_page_2.png, etc.
                    page_output_path = base_output_path.parent / f"{base_name}_page{page_num + 1}.png"
                    
                    try:
                        # 1. Amélioration du contraste
                        img_enhanced = enhance_contrast_clahe(pdf_img)
                        
                        # 2. Classification du type de capture
                        capture_info = self.capture_classifier.classify(img_enhanced)
                        
                        # 3. Sauvegarde de l'image prétraitée
                        ensure_dir(page_output_path.parent)
                        cv2.imwrite(str(page_output_path), img_enhanced)
                        
                        # 4. Créer le modèle de sortie
                        capture_info_obj = CaptureInfo(**capture_info)
                        output = PreprocessingOutput(
                            status='success',
                            input_path=file_path,
                            processed_path=str(page_output_path),
                            capture_type=capture_info_obj.type,
                            capture_info=capture_info_obj,
                            processing_time=0.0  # Temps de traitement par page (optionnel)
                        )
                        
                        # 5. Sauvegarder les métadonnées dans un fichier JSON
                        metadata_path = page_output_path.with_suffix('.json')
                        try:
                            metadata_dict = output.to_dict()
                            metadata_dict['page_num'] = page_num + 1
                            metadata_dict['total_pages'] = len(pdf_images)
                            with open(metadata_path, 'w') as f:
                                json.dump(metadata_dict, f, indent=2, default=str)
                        except Exception as e:
                            logger.warning(f"Impossible de sauvegarder les métadonnées pour la page {page_num + 1}", exc_info=True)
                        
                        results.append(output)
                        logger.info(f"Page {page_num + 1}/{len(pdf_images)} traitée: {page_output_path.name}")
                        
                    except Exception as e:
                        logger.error(f"Erreur lors du traitement de la page {page_num + 1} de {file_path}", exc_info=True)
                        # Créer un PreprocessingOutput avec status='error'
                        error_output = PreprocessingOutput(
                            status='error',
                            input_path=file_path,
                            processed_path='',  # Pas de fichier traité en cas d'erreur
                            capture_type=CaptureType.PHOTO,  # Valeur par défaut
                            capture_info=CaptureInfo(
                                type=CaptureType.PHOTO,
                                white_percentage=0.0,
                                confidence=0.0,
                                enabled=False,
                                reason='Error during processing'
                            ),
                            processing_time=0.0,
                            error=str(e)
                        )
                        results.append(error_output)
            else:
                # Traiter les images normales (une seule image)
                output_path = get_output_path(file_path, output_dir).replace(Path(file_path).suffix, '.png')
                try:
                    result = self.process(file_path, output_path)
                    results.append(result)
                    logger.info(f"Image traitée: {Path(output_path).name}")
                except Exception as e:
                    logger.error(f"Erreur lors du traitement de {file_path}", exc_info=True)
                    # Créer un PreprocessingOutput avec status='error'
                    error_output = PreprocessingOutput(
                        status='error',
                        input_path=file_path,
                        processed_path='',  # Pas de fichier traité en cas d'erreur
                        capture_type=CaptureType.PHOTO,  # Valeur par défaut
                        capture_info=CaptureInfo(
                            type=CaptureType.PHOTO,
                            white_percentage=0.0,
                            confidence=0.0,
                            enabled=False,
                            reason='Error during processing'
                        ),
                        processing_time=0.0,
                        error=str(e)
                    )
                    results.append(error_output)
        
        return results

