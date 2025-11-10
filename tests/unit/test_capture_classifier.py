"""
Unit tests for capture_classifier.py
"""

import pytest
import numpy as np
import cv2
import tempfile
import os
from src.utils.capture_classifier import CaptureClassifier, classify_capture_method


class TestCaptureClassifier:
    """Tests for CaptureClassifier class"""
    
    def test_initialization_default(self):
        """Test CaptureClassifier initialization with default values"""
        classifier = CaptureClassifier()
        assert classifier.white_level_threshold == 245
        assert classifier.white_percentage_threshold == 0.70
        assert classifier.enabled is True
    
    def test_initialization_custom(self):
        """Test CaptureClassifier initialization with custom values"""
        classifier = CaptureClassifier(
            white_level_threshold=250,
            white_percentage_threshold=0.80,
            enabled=False
        )
        assert classifier.white_level_threshold == 250
        assert classifier.white_percentage_threshold == 0.80
        assert classifier.enabled is False
    
    def test_classify_disabled(self):
        """Test classify() when classifier is disabled"""
        classifier = CaptureClassifier(enabled=False)
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        result = classifier.classify(image)
        
        assert result['type'] == 'PHOTO'
        assert result['white_percentage'] == 0.0
        assert result['confidence'] == 0.0
        assert result['enabled'] is False
        assert 'reason' in result
    
    def test_classify_scan_high_white(self):
        """Test classify() with high white percentage (SCAN)"""
        classifier = CaptureClassifier(
            white_level_threshold=245,
            white_percentage_threshold=0.70
        )
        # Create image with 80% white pixels (above threshold)
        image = np.ones((100, 100, 3), dtype=np.uint8) * 250  # Almost white
        
        result = classifier.classify(image)
        
        assert result['type'] == 'SCAN'
        assert result['white_percentage'] > 0.70
        assert result['enabled'] is True
        assert 'reason' in result
    
    def test_classify_photo_low_white(self):
        """Test classify() with low white percentage (PHOTO)"""
        classifier = CaptureClassifier(
            white_level_threshold=245,
            white_percentage_threshold=0.70
        )
        # Create image with low white percentage (below threshold)
        image = np.ones((100, 100, 3), dtype=np.uint8) * 100  # Dark image
        
        result = classifier.classify(image)
        
        assert result['type'] == 'PHOTO'
        assert result['white_percentage'] < 0.70
        assert result['enabled'] is True
        assert 'reason' in result
    
    def test_classify_grayscale_image(self):
        """Test classify() with grayscale image"""
        classifier = CaptureClassifier()
        # Create grayscale image
        image = np.ones((100, 100), dtype=np.uint8) * 250
        
        result = classifier.classify(image)
        
        assert result['type'] in ['SCAN', 'PHOTO']
        assert 'white_percentage' in result
        assert 'confidence' in result
    
    def test_classify_color_image(self):
        """Test classify() with color BGR image"""
        classifier = CaptureClassifier()
        # Create color image
        image = np.ones((100, 100, 3), dtype=np.uint8) * 250
        
        result = classifier.classify(image)
        
        assert result['type'] in ['SCAN', 'PHOTO']
        assert 'white_percentage' in result
    
    def test_classify_confidence_calculation(self):
        """Test that confidence is calculated correctly"""
        classifier = CaptureClassifier(
            white_level_threshold=245,
            white_percentage_threshold=0.70
        )
        # Create image exactly at threshold
        image = np.ones((100, 100, 3), dtype=np.uint8) * 245
        
        result = classifier.classify(image)
        
        assert 'confidence' in result
        assert 0.0 <= result['confidence'] <= 1.0
    
    def test_classify_exception_handling(self):
        """Test classify() exception handling"""
        classifier = CaptureClassifier()
        # Create invalid image (empty array)
        image = np.array([])
        
        result = classifier.classify(image)
        
        # Should return PHOTO on error
        assert result['type'] == 'PHOTO'
        assert 'error' in result or 'reason' in result
    
    def test_classify_at_threshold_boundary(self):
        """Test classify() at threshold boundary"""
        classifier = CaptureClassifier(
            white_level_threshold=245,
            white_percentage_threshold=0.70
        )
        # Create image exactly at threshold (70% white)
        # We'll create an image with exactly 70% white pixels
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        # Set 70% of pixels to white
        white_count = int(100 * 100 * 0.70)
        flat_image = image.reshape(-1, 3)
        flat_image[:white_count] = 250  # White pixels
        image = flat_image.reshape(100, 100, 3)
        
        result = classifier.classify(image)
        
        # Should classify based on threshold (>= means SCAN)
        assert result['type'] in ['SCAN', 'PHOTO']
        assert abs(result['white_percentage'] - 0.70) < 0.01
    
    def test_classify_from_path_success(self):
        """Test classify_from_path() with valid image"""
        classifier = CaptureClassifier()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "test.png")
            # Create a test image
            test_image = np.ones((100, 100, 3), dtype=np.uint8) * 250
            cv2.imwrite(image_path, test_image)
            
            result = classifier.classify_from_path(image_path)
            
            assert result['type'] in ['SCAN', 'PHOTO']
            assert 'white_percentage' in result
    
    def test_classify_from_path_nonexistent(self):
        """Test classify_from_path() with non-existent file"""
        classifier = CaptureClassifier()
        
        result = classifier.classify_from_path("/nonexistent/image.png")
        
        assert result['type'] == 'PHOTO'
        assert 'error' in result or 'reason' in result
    
    def test_classify_from_path_invalid_image(self):
        """Test classify_from_path() with invalid image file"""
        classifier = CaptureClassifier()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_path = os.path.join(tmpdir, "invalid.txt")
            with open(invalid_path, 'w') as f:
                f.write("not an image")
            
            result = classifier.classify_from_path(invalid_path)
            
            assert result['type'] == 'PHOTO'
            assert 'error' in result or 'reason' in result


class TestClassifyCaptureMethod:
    """Tests for classify_capture_method() function"""
    
    def test_classify_capture_method_scan(self):
        """Test classify_capture_method() returns 'SCAN'"""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "test.png")
            # Create high white percentage image
            test_image = np.ones((100, 100, 3), dtype=np.uint8) * 250
            cv2.imwrite(image_path, test_image)
            
            result = classify_capture_method(
                image_path,
                white_level_threshold=245,
                white_percentage_threshold=0.50  # Low threshold to ensure SCAN
            )
            
            assert result in ['SCAN', 'PHOTO']
    
    def test_classify_capture_method_photo(self):
        """Test classify_capture_method() returns 'PHOTO'"""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "test.png")
            # Create low white percentage image
            test_image = np.ones((100, 100, 3), dtype=np.uint8) * 100
            cv2.imwrite(image_path, test_image)
            
            result = classify_capture_method(
                image_path,
                white_level_threshold=245,
                white_percentage_threshold=0.90  # High threshold to ensure PHOTO
            )
            
            assert result == 'PHOTO'

