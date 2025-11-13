"""
Integration tests for preprocessing.py
"""

import pytest
import tempfile
import os
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import patch
from src.pipeline.preprocessing import PreprocessingNormalizer
from src.utils.exceptions import PreprocessingError, ImageProcessingError
from src.utils.capture_classifier import CaptureClassifier


def create_preprocessing_normalizer(app_config):
    """
    Helper pour créer un PreprocessingNormalizer avec la vraie configuration.
    Évite la répétition dans les tests d'intégration.
    """
    capture_classifier = CaptureClassifier(
        white_level_threshold=app_config.geometry.capture_classifier_white_level_threshold,
        white_percentage_threshold=app_config.geometry.capture_classifier_white_percentage_threshold,
        enabled=app_config.geometry.capture_classifier_enabled
    )
    return PreprocessingNormalizer(
        capture_classifier=capture_classifier,
        pdf_config=app_config.pdf
    )


class TestPreprocessingNormalizer:
    """Integration tests for PreprocessingNormalizer"""
    
    @patch('src.pipeline.preprocessing.time.time', side_effect=[100.0, 100.5])
    def test_process_image_success(self, mock_time, app_config):
        """Test processing a single image successfully"""
        # Utilise la VRAIE configuration pour les tests d'intégration
        normalizer = create_preprocessing_normalizer(app_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test image
            input_path = os.path.join(tmpdir, "test_image.png")
            test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
            cv2.imwrite(input_path, test_image)
            
            output_path = os.path.join(tmpdir, "output", "processed.png")
            
            # Process the image
            result = normalizer.process(input_path, output_path)
            
            # Check results (PreprocessingOutput is now a Pydantic model)
            assert result.status == 'success'
            assert result.input_path == input_path
            assert result.processed_path == output_path
            assert result.capture_type is not None
            assert result.capture_info is not None
            # Vérifier que processing_time est égal à la valeur contrôlée (0.5 secondes)
            assert result.processing_time == pytest.approx(0.5, rel=1e-6)
            
            # Check that output file was created
            assert os.path.exists(output_path)
            
            # Check that metadata file was created
            metadata_path = Path(output_path).with_suffix('.json')
            assert os.path.exists(metadata_path)
    
    def test_process_image_nonexistent(self, app_config):
        """Test processing a non-existent image"""
        normalizer = create_preprocessing_normalizer(app_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "nonexistent.png")
            output_path = os.path.join(tmpdir, "output.png")
            
            with pytest.raises(ImageProcessingError):
                normalizer.process(input_path, output_path)
    
    def test_process_batch_images(self, app_config):
        """Test processing a batch of images"""
        normalizer = create_preprocessing_normalizer(app_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "input")
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(input_dir)
            
            # Create multiple test images
            for i in range(3):
                image_path = os.path.join(input_dir, f"test_{i}.png")
                test_image = np.ones((100, 100, 3), dtype=np.uint8) * (100 + i * 50)
                cv2.imwrite(image_path, test_image)
            
            # Process batch
            results = normalizer.process_batch(input_dir, output_dir)
            
            # Check results (results is now List[PreprocessingOutput])
            assert len(results) == 3
            assert all(r.status == 'success' for r in results)
            
            # Check that output files were created
            for i in range(3):
                output_path = os.path.join(output_dir, f"test_{i}.png")
                assert os.path.exists(output_path)
    
    def test_process_batch_empty_directory(self, app_config):
        """Test processing an empty directory"""
        normalizer = create_preprocessing_normalizer(app_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "input")
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(input_dir)
            
            results = normalizer.process_batch(input_dir, output_dir)
            
            assert results == []
    
    def test_process_pdf(self, app_config):
        """Test processing a PDF file from fixtures"""
        normalizer = create_preprocessing_normalizer(app_config)
        
        # Use PDF from fixtures directory
        fixtures_dir = Path(__file__).parent.parent / "fixtures"
        pdf_path = fixtures_dir / "test.pdf"
        
        if not pdf_path.exists():
            pytest.skip("Le fichier PDF de test 'tests/fixtures/test.pdf' est manquant.")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "processed.png")
            
            # Process the PDF
            result = normalizer.process(str(pdf_path), output_path)
            
            # Check results (PreprocessingOutput is now a Pydantic model)
            assert result.status == 'success'
            assert result.input_path == str(pdf_path)
            assert result.processed_path == output_path
            assert result.capture_type is not None
            assert result.capture_info is not None
            assert result.processing_time > 0
            
            # Check that output file was created
            assert os.path.exists(output_path)
            
            # Check that metadata file was created
            metadata_path = Path(output_path).with_suffix('.json')
            assert os.path.exists(metadata_path)
    
    def test_process_creates_metadata_file(self, app_config):
        """Test that process() creates a metadata JSON file"""
        normalizer = create_preprocessing_normalizer(app_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "test.png")
            test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
            cv2.imwrite(input_path, test_image)
            
            output_path = os.path.join(tmpdir, "output.png")
            normalizer.process(input_path, output_path)
            
            # Check metadata file
            metadata_path = Path(output_path).with_suffix('.json')
            assert os.path.exists(metadata_path)
            
            # Check metadata content
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            assert metadata['status'] == 'success'
            assert 'capture_type' in metadata
            assert 'processing_time' in metadata
    
    def test_process_batch_with_mixed_files(self, app_config):
        """Test process_batch() with mixed image types"""
        normalizer = create_preprocessing_normalizer(app_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "input")
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(input_dir)
            
            # Create different image types
            for ext in ['.png', '.jpg', '.jpeg']:
                image_path = os.path.join(input_dir, f"test{ext}")
                test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
                cv2.imwrite(image_path, test_image)
            
            results = normalizer.process_batch(input_dir, output_dir)
            
            assert len(results) == 3
            assert all(r.status == 'success' for r in results)
    
    def test_process_batch_handles_errors_gracefully(self, app_config):
        """Test that process_batch() handles errors gracefully"""
        normalizer = create_preprocessing_normalizer(app_config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "input")
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(input_dir)
            
            # Create a valid image
            valid_path = os.path.join(input_dir, "valid.png")
            test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
            cv2.imwrite(valid_path, test_image)
            
            # Create an invalid file (text file with .png extension)
            invalid_path = os.path.join(input_dir, "invalid.png")
            with open(invalid_path, 'w') as f:
                f.write("not an image")
            
            results = normalizer.process_batch(input_dir, output_dir)
            
            # Should have results for both, with error for invalid file
            assert len(results) == 2
            valid_results = [r for r in results if r.status == 'success']
            error_results = [r for r in results if r.status == 'error']
            assert len(valid_results) >= 1
            assert len(error_results) >= 1

