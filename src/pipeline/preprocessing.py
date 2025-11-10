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
from src.utils.config_loader import get_config
from src.utils.image_enhancer import enhance_contrast_clahe


class PreprocessingNormalizer:
    """
    Applique les prétraitements à une image avant les étapes géométriques.
    """

    def __init__(self, config=None):
        """
        Initialise le normaliseur de prétraitement
        
        Args:
            config: Configuration optionnelle (dict ou None pour charger depuis config.yaml)
        """
        app_config = get_config()
        geo_config = app_config.geometry
        self.capture_classifier = CaptureClassifier(
            white_level_threshold=geo_config.capture_classifier_white_level_threshold,
            white_percentage_threshold=geo_config.capture_classifier_white_percentage_threshold,
            enabled=geo_config.capture_classifier_enabled
        )

    def process(self, input_path: str, output_path: str) -> Dict:
        """
        Charge une image, améliore son contraste et classifie son type.
        Sauvegarde l'image prétraitée et retourne les métadonnées.
        
        Args:
            input_path: Chemin vers le document d'entrée (PDF ou image)
            output_path: Chemin de sortie pour l'image prétraitée
            
        Returns:
            dict: Métadonnées du prétraitement avec capture_type, capture_info, etc.
        """
        start_time = time.time()
        
        if is_pdf(input_path):
            images = pdf_to_images(input_path, dpi=300)
            if not images:
                raise ValueError(f"Aucune page trouvée dans le PDF: {input_path}")
            img = images[0]
        else:
            img = cv2.imread(input_path)
            if img is None:
                raise ValueError(f"Impossible de charger l'image: {input_path}")

        # 1. Amélioration du contraste
        img_enhanced = enhance_contrast_clahe(img)

        # 2. Classification du type de capture
        capture_info = self.capture_classifier.classify(img_enhanced)
        
        # 3. Sauvegarde de l'image prétraitée
        ensure_dir(os.path.dirname(output_path))
        cv2.imwrite(output_path, img_enhanced)
        
        processing_time = time.time() - start_time

        # 4. Sauvegarder les métadonnées dans un fichier JSON pour l'étape suivante
        metadata = {
            'status': 'success',
            'input_path': input_path,
            'processed_path': output_path,
            'capture_type': capture_info['type'],
            'capture_info': capture_info,
            'processing_time': processing_time
        }
        
        # Sauvegarder les métadonnées dans un fichier JSON
        metadata_path = Path(output_path).with_suffix('.json')
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"⚠️  Impossible de sauvegarder les métadonnées: {e}")

        # 5. Retourne les métadonnées pour l'étape suivante
        return metadata

    def process_batch(self, input_dir: str, output_dir: str) -> List[Dict]:
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
                pdf_images = pdf_to_images(file_path, dpi=300)
                if not pdf_images:
                    print(f"⚠️  Aucune page trouvée dans le PDF: {file_path}")
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
                        
                        # 4. Sauvegarder les métadonnées dans un fichier JSON
                        metadata = {
                            'status': 'success',
                            'input_path': file_path,
                            'processed_path': str(page_output_path),
                            'page_num': page_num + 1,
                            'total_pages': len(pdf_images),
                            'capture_type': capture_info['type'],
                            'capture_info': capture_info,
                            'processing_time': 0.0  # Temps de traitement par page (optionnel)
                        }
                        
                        metadata_path = page_output_path.with_suffix('.json')
                        try:
                            with open(metadata_path, 'w') as f:
                                json.dump(metadata, f, indent=2)
                        except Exception as e:
                            print(f"⚠️  Impossible de sauvegarder les métadonnées pour la page {page_num + 1}: {e}")
                        
                        results.append(metadata)
                        print(f"✓ Page {page_num + 1}/{len(pdf_images)} traitée: {page_output_path.name}")
                        
                    except Exception as e:
                        print(f"❌ Erreur lors du traitement de la page {page_num + 1} de {file_path}: {e}")
                        results.append({
                            'status': 'error',
                            'input_path': file_path,
                            'page_num': page_num + 1,
                            'total_pages': len(pdf_images),
                            'error': str(e)
                        })
            else:
                # Traiter les images normales (une seule image)
                output_path = get_output_path(file_path, output_dir).replace(Path(file_path).suffix, '.png')
                try:
                    result = self.process(file_path, output_path)
                    results.append(result)
                    print(f"✓ Image traitée: {Path(output_path).name}")
                except Exception as e:
                    print(f"❌ Erreur lors du traitement de {file_path}: {e}")
                    results.append({
                        'status': 'error',
                        'input_path': file_path,
                        'error': str(e)
                    })
        
        return results

