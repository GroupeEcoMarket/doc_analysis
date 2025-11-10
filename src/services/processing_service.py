"""
Processing Service
Business logic layer for document processing pipeline.
This service is independent of FastAPI and can be reused by other interfaces (CLI, workers, etc.).
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np

from src.pipeline.preprocessing import PreprocessingNormalizer
from src.pipeline.geometry import GeometryNormalizer
from src.pipeline.models import PreprocessingOutput, GeometryOutput, CaptureInfo
from src.utils.capture_classifier import CaptureClassifier
from src.utils.config_loader import GeometryConfig, QAConfig
from src.utils.logger import get_logger
from src.utils.exceptions import (
    GeometryError,
    PreprocessingError,
    ModelLoadingError,
    ImageProcessingError,
    PipelineError
)
from typing import Optional

logger = get_logger(__name__)


class ProcessingService:
    """
    Service layer for document processing.
    
    This service encapsulates all business logic for processing documents,
    making it independent of the API framework (FastAPI). It can be reused
    by CLI tools, background workers (Celery, Dramatiq), or other interfaces.
    """
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.pdf', '.tiff', '.bmp'}
    
    def __init__(
        self,
        preprocessing_normalizer: Optional[PreprocessingNormalizer] = None,
        geometry_normalizer: Optional[GeometryNormalizer] = None
    ):
        """
        Initialize the processing service.
        
        Args:
            preprocessing_normalizer: Normaliseur de prétraitement (injecté via DI).
                                    Si None, crée une instance par défaut.
            geometry_normalizer: Normaliseur géométrique (injecté via DI).
                               Si None, crée une instance par défaut.
        """
        self.preprocessing_normalizer = preprocessing_normalizer
        self.geometry_normalizer = geometry_normalizer
    
    def validate_file(self, filename: Optional[str], file_size: int = 0) -> Tuple[str, None] | Tuple[None, str]:
        """
        Validate uploaded file.
        
        Args:
            filename: Name of the uploaded file
            file_size: Size of the file in bytes
            
        Returns:
            tuple: (file_extension, error_message)
                - If valid: (extension, None)
                - If invalid: (None, error_message)
        """
        if not filename:
            return None, "Le nom du fichier est requis"
        
        file_extension = Path(filename).suffix.lower()
        
        if not file_extension:
            return None, "Impossible de déterminer l'extension du fichier"
        
        if file_extension not in self.ALLOWED_EXTENSIONS:
            return (
                None,
                f"Format de fichier non supporté. Extensions autorisées: {', '.join(self.ALLOWED_EXTENSIONS)}"
            )
        
        if file_size == 0:
            return None, "Le fichier uploadé est vide"
        
        return file_extension, None
    
    def create_temp_directories(self, prefix: str = "processing_") -> Dict[str, str]:
        """
        Create temporary directories for processing.
        
        Args:
            prefix: Prefix for temporary directory names
            
        Returns:
            dict: Dictionary with keys 'input', 'output', 'preprocessing'
        """
        return {
            'input': tempfile.mkdtemp(prefix=f"{prefix}input_"),
            'output': tempfile.mkdtemp(prefix=f"{prefix}output_"),
            'preprocessing': tempfile.mkdtemp(prefix=f"{prefix}preprocessing_")
        }
    
    def save_uploaded_file(
        self,
        file_content: bytes,
        filename: str,
        temp_input_dir: str
    ) -> str:
        """
        Save uploaded file to temporary directory.
        
        Args:
            file_content: Content of the uploaded file
            filename: Original filename
            temp_input_dir: Temporary input directory
            
        Returns:
            str: Path to the saved file
            
        Raises:
            ImageProcessingError: If file cannot be saved or is empty
        """
        file_extension = Path(filename).suffix.lower()
        temp_file_path = os.path.join(temp_input_dir, filename or f"upload{file_extension}")
        
        # Save file
        with open(temp_file_path, "wb") as buffer:
            buffer.write(file_content)
        
        # Verify file was saved and is not empty
        if not os.path.exists(temp_file_path) or os.path.getsize(temp_file_path) == 0:
            raise ImageProcessingError("Le fichier uploadé est vide ou n'a pas pu être sauvegardé")
        
        return temp_file_path
    
    def process_geometry(
        self,
        file_path: str,
        filename: Optional[str] = None,
        temp_dirs: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Process a document through preprocessing and geometry normalization.
        
        This is the main business logic method that orchestrates:
        1. Preprocessing (enhancement, contrast, capture type classification)
        2. Geometry normalization (crop, deskew, rotation)
        
        Args:
            file_path: Path to the input file
            filename: Original filename (optional, for output naming)
            temp_dirs: Dictionary with temp directories (optional, will create if not provided)
                Keys: 'input', 'output', 'preprocessing'
            
        Returns:
            dict: Processing results with the following structure:
                - status: 'success' or 'error'
                - input_filename: Original filename
                - output_files: Dictionary with paths to output files
                - metadata: Processing metadata
                - qa_flags: Quality assurance flags
                - temp_dirs: Temporary directories used
                - processing_time: Total processing time
                
        Raises:
            PreprocessingError: If preprocessing fails
            GeometryError: If geometry normalization fails
            ModelLoadingError: If models cannot be loaded
            ImageProcessingError: If image processing fails
        """
        # Create temp directories if not provided
        if temp_dirs is None:
            temp_dirs = self.create_temp_directories(prefix="geometry_")
        
        try:
            # 1. Apply preprocessing
            logger.info(f"Prétraitement du fichier: {filename or file_path}")
            preprocessor = self.preprocessing_normalizer or PreprocessingNormalizer()
            preprocessing_output: PreprocessingOutput = preprocessor.process(
                file_path,
                os.path.join(temp_dirs['preprocessing'], "preprocessed.png")
            )
            
            if preprocessing_output.status != 'success':
                raise PreprocessingError(
                    preprocessing_output.error or 'Erreur inconnue lors du prétraitement'
                )
            
            # 2. Load preprocessed image
            preprocessed_image_path = preprocessing_output.processed_path
            img = cv2.imread(preprocessed_image_path)
            if img is None:
                raise ImageProcessingError("Impossible de charger l'image prétraitée")
            
            # 3. Apply geometry normalization
            logger.info("Application de la normalisation géométrique")
            geometry_normalizer = self.geometry_normalizer or GeometryNormalizer()
            
            # Prepare output paths
            output_filename = Path(filename or Path(file_path).stem or "output").stem
            output_path = os.path.join(temp_dirs['output'], f"{output_filename}_transformed.png")
            
            # Get capture information from preprocessing (convert to dict for compatibility)
            capture_info_dict = preprocessing_output.capture_info.model_dump() if isinstance(preprocessing_output.capture_info, CaptureInfo) else preprocessing_output.capture_info
            
            # Process image (returns GeometryOutput directly)
            result: GeometryOutput = geometry_normalizer.process(
                img=img,
                output_path=output_path,
                capture_type=preprocessing_output.capture_type,
                original_input_path=file_path,
                capture_info=capture_info_dict
            )
            
            # 4. Prepare response (convert to dict for API compatibility)
            response_data = {
                "status": result.status,
                "input_filename": filename or Path(file_path).name,
                "output_files": {
                    "transformed": result.output_transformed_path,
                    "original": result.output_original_path,
                    "transform_file": result.transform_file,
                    "qa_file": result.qa_file
                },
                "metadata": {
                    "crop_applied": result.crop_applied,
                    "deskew_applied": result.deskew_applied,
                    "rotation_applied": result.rotation_applied,
                    "orientation_angle": result.angle,
                    "capture_type": preprocessing_output.capture_type,
                    "processing_time": result.processing_time
                },
                "qa_flags": result.qa_flags,
                "temp_dirs": temp_dirs,
                "processing_time": result.processing_time
            }
            
            return response_data
            
        except (PreprocessingError, GeometryError, ModelLoadingError, ImageProcessingError) as e:
            logger.error(f"Erreur lors du traitement: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Erreur inattendue lors du traitement: {e}", exc_info=True)
            raise PipelineError(f"Erreur interne lors du traitement: {str(e)}") from e
    
    def process_full_analysis(
        self,
        file_path: str,
        filename: Optional[str] = None,
        temp_dirs: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Process a document through the complete analysis pipeline.
        
        Currently implements:
        - Preprocessing
        - Geometry normalization
        
        Future stages (not yet implemented):
        - Colometry normalization
        - Feature extraction
        
        Args:
            file_path: Path to the input file
            filename: Original filename (optional)
            temp_dirs: Dictionary with temp directories (optional)
            
        Returns:
            dict: Complete processing results with pipeline_stages information
        """
        # Process geometry (which includes preprocessing)
        result = self.process_geometry(file_path, filename, temp_dirs)
        
        # Add pipeline stages information
        result["pipeline_stages"] = {
            "preprocessing": "completed",
            "geometry": "completed",
            "colometry": "not_implemented",
            "features": "not_implemented"
        }
        
        return result
    
    def cleanup_temp_directories(self, temp_dirs: Dict[str, str]) -> None:
        """
        Clean up temporary directories.
        
        Args:
            temp_dirs: Dictionary with paths to temporary directories
        """
        for dir_name, dir_path in temp_dirs.items():
            if dir_path and os.path.exists(dir_path):
                try:
                    shutil.rmtree(dir_path, ignore_errors=True)
                    logger.debug(f"Nettoyage du répertoire temporaire {dir_name}: {dir_path}")
                except Exception as e:
                    logger.warning(f"Impossible de nettoyer le répertoire temporaire {dir_path}: {e}")

