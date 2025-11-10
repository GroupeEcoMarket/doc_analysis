"""
Unit tests for qa_flags.py
"""

import pytest
import tempfile
import os
import json
import numpy as np
from src.utils.qa_flags import QAFlags, QADetector, save_qa_flags, load_qa_flags


class TestQAFlags:
    """Tests for QAFlags dataclass"""
    
    def test_qa_flags_defaults(self):
        """Test QAFlags with default values"""
        flags = QAFlags()
        assert flags.low_confidence_orientation is False
        assert flags.overcrop_risk is False
        assert flags.orientation_confidence == 0.0
        assert flags.crop_margins is None
        assert flags.final_resolution is None
    
    def test_qa_flags_custom_values(self):
        """Test QAFlags with custom values"""
        flags = QAFlags(
            low_confidence_orientation=True,
            overcrop_risk=True,
            orientation_confidence=0.5,
            orientation_angle=90,
            deskew_angle=2.5,
            crop_margins={"top": 5.0, "bottom": 5.0},
            final_resolution=[1200, 1600],
            capture_type="SCAN"
        )
        assert flags.low_confidence_orientation is True
        assert flags.orientation_confidence == 0.5
        assert flags.orientation_angle == 90
        assert flags.crop_margins == {"top": 5.0, "bottom": 5.0}
        assert flags.final_resolution == [1200, 1600]
        assert flags.capture_type == "SCAN"
    
    def test_qa_flags_to_dict(self):
        """Test QAFlags.to_dict()"""
        flags = QAFlags(
            low_confidence_orientation=True,
            orientation_confidence=0.7,
            crop_margins={"top": 5.0},
            final_resolution=[1000, 1500]
        )
        result = flags.to_dict()
        assert result["low_confidence_orientation"] is True
        assert result["orientation_confidence"] == 0.7
        assert result["crop_margins"] == {"top": 5.0}
        assert result["final_resolution"] == [1000, 1500]
    
    def test_qa_flags_to_dict_with_none(self):
        """Test QAFlags.to_dict() with None values"""
        flags = QAFlags()
        result = flags.to_dict()
        assert result["crop_margins"] == {}
        assert result["final_resolution"] == []
    
    def test_qa_flags_from_dict(self):
        """Test QAFlags.from_dict()"""
        data = {
            "low_confidence_orientation": True,
            "overcrop_risk": False,
            "orientation_confidence": 0.8,
            "orientation_angle": 180,
            "crop_margins": {"top": 3.0},
            "final_resolution": [800, 1200],
            "capture_type": "PHOTO"
        }
        flags = QAFlags.from_dict(data)
        assert flags.low_confidence_orientation is True
        assert flags.overcrop_risk is False
        assert flags.orientation_confidence == 0.8
        assert flags.orientation_angle == 180
        assert flags.crop_margins == {"top": 3.0}
        assert flags.final_resolution == [800, 1200]
        assert flags.capture_type == "PHOTO"


class TestQADetector:
    """Tests for QADetector class"""
    
    def test_qadetector_initialization_default(self):
        """Test QADetector initialization with default config"""
        detector = QADetector()
        # Should load from config.yaml
        assert hasattr(detector, "orientation_confidence_threshold")
        assert hasattr(detector, "overcrop_threshold")
        assert hasattr(detector, "min_resolution")
        assert hasattr(detector, "contrast_threshold")
    
    def test_qadetector_initialization_custom_config(self):
        """Test QADetector initialization with custom config dict"""
        config = {
            "orientation_confidence_threshold": 0.75,
            "overcrop_threshold": 10.0,
            "min_resolution": [1000, 1500],
            "contrast_threshold": 60.0
        }
        detector = QADetector(config=config)
        assert detector.orientation_confidence_threshold == 0.75
        assert detector.overcrop_threshold == 10.0
        assert detector.min_resolution == [1000, 1500]
        assert detector.contrast_threshold == 60.0
    
    def test_detect_flags_low_confidence_orientation(self):
        """Test detect_flags() with low orientation confidence"""
        detector = QADetector(config={
            "orientation_confidence_threshold": 0.70,
            "overcrop_threshold": 8.0,
            "min_resolution": [1200, 1600],
            "contrast_threshold": 50.0
        })
        
        original = np.ones((2000, 3000, 3), dtype=np.uint8) * 255
        final = np.ones((1900, 2900, 3), dtype=np.uint8) * 255
        
        metadata = {
            "orientation_confidence": 0.5,  # Below threshold
            "angle": 90,  # Note: metadata uses 'angle', not 'orientation_angle'
            "deskew_applied": False,
            "crop_applied": True,
            "rotation_applied": True,
            "capture_type": "SCAN"
        }
        
        flags = detector.detect_flags(original, final, metadata, processing_time=1.0)
        assert flags.low_confidence_orientation is True
        assert flags.orientation_confidence == 0.5
        assert flags.orientation_angle == 90
    
    def test_detect_flags_overcrop_risk(self):
        """Test detect_flags() with overcrop risk"""
        detector = QADetector(config={
            "orientation_confidence_threshold": 0.70,
            "overcrop_threshold": 8.0,  # 8% margin threshold
            "min_resolution": [1200, 1600],
            "contrast_threshold": 50.0
        })
        
        # Original: 2000x3000, Final: 1800x2800 (10% crop on each side = risk)
        original = np.ones((2000, 3000, 3), dtype=np.uint8) * 255
        final = np.ones((1800, 2800, 3), dtype=np.uint8) * 255
        
        metadata = {
            "orientation_confidence": 0.9,
            "angle": 0,
            "deskew_applied": False,
            "crop_applied": True,
            "rotation_applied": False,
            "crop_metadata": {
                "crop_applied": True,  # Required for overcrop detection
                "crop_x": 100,
                "crop_y": 100,
                "crop_width": 2800,
                "crop_height": 1800,
                "original_width": 3000,
                "original_height": 2000,
                "source_points": [[100, 100], [2900, 100], [2900, 1900], [100, 1900]]  # Required for margin calculation
            },
            "capture_type": "SCAN"
        }
        
        flags = detector.detect_flags(original, final, metadata, processing_time=1.0)
        # Should detect overcrop risk (margins < 8% threshold)
        # With source_points, margins are: top=5%, bottom=5%, left=3.33%, right=3.33%
        # left and right are < 8%, so overcrop_risk should be True
        assert flags.overcrop_risk is True
    
    def test_detect_flags_too_small_final(self):
        """Test detect_flags() with too small final image"""
        detector = QADetector(config={
            "orientation_confidence_threshold": 0.70,
            "overcrop_threshold": 8.0,
            "min_resolution": [1200, 1600],  # Minimum required
            "contrast_threshold": 50.0
        })
        
        original = np.ones((2000, 3000, 3), dtype=np.uint8) * 255
        final = np.ones((800, 1000, 3), dtype=np.uint8) * 255  # Too small
        
        metadata = {
            "orientation_confidence": 0.9,
            "orientation_angle": 0,
            "deskew_applied": False,
            "crop_applied": True,
            "rotation_applied": False,
            "capture_type": "SCAN"
        }
        
        flags = detector.detect_flags(original, final, metadata, processing_time=1.0)
        assert flags.too_small_final is True
        assert flags.final_resolution == [1000, 800]  # [width, height]
    
    def test_detect_flags_rotated(self):
        """Test detect_flags() with rotation applied"""
        detector = QADetector(config={
            "orientation_confidence_threshold": 0.70,
            "overcrop_threshold": 8.0,
            "min_resolution": [1200, 1600],
            "contrast_threshold": 50.0
        })
        
        original = np.ones((2000, 3000, 3), dtype=np.uint8) * 255
        final = np.ones((3000, 2000, 3), dtype=np.uint8) * 255  # Rotated
        
        metadata = {
            "orientation_confidence": 0.9,
            "angle": 90,  # Note: metadata uses 'angle', not 'orientation_angle'
            "deskew_applied": False,
            "crop_applied": False,
            "rotation_applied": True,  # Required for rotated flag
            "capture_type": "SCAN"
        }
        
        flags = detector.detect_flags(original, final, metadata, processing_time=1.0)
        assert flags.rotated is True
        assert flags.orientation_angle == 90


class TestQAFlagsFunctions:
    """Tests for module-level functions"""
    
    def test_save_and_load_qa_flags(self):
        """Test save_qa_flags() and load_qa_flags()"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output", "image.png")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            flags = QAFlags(
                low_confidence_orientation=True,
                orientation_confidence=0.6,
                orientation_angle=90,
                crop_margins={"top": 5.0, "bottom": 5.0},
                final_resolution=[1200, 1600],
                capture_type="SCAN"
            )
            
            # Save
            save_qa_flags(output_path, flags)
            
            # Load
            loaded_flags = load_qa_flags(output_path)
            assert loaded_flags is not None
            assert loaded_flags.low_confidence_orientation is True
            assert loaded_flags.orientation_confidence == 0.6
            assert loaded_flags.orientation_angle == 90
            assert loaded_flags.crop_margins == {"top": 5.0, "bottom": 5.0}
            assert loaded_flags.final_resolution == [1200, 1600]
            assert loaded_flags.capture_type == "SCAN"
    
    def test_load_qa_flags_nonexistent(self):
        """Test load_qa_flags() with non-existent file"""
        result = load_qa_flags("/nonexistent/image.png")
        assert result is None

