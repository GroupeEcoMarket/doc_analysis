"""
Unit tests for transform_applier.py
"""

import pytest
import numpy as np
import cv2
import tempfile
import os
from src.utils.transform_applier import (
    apply_single_transform,
    apply_transform_sequence,
    apply_transforms_from_file
)
from src.utils.transform_handler import Transform, TransformSequence


class TestApplySingleTransform:
    """Tests for apply_single_transform()"""
    
    def test_apply_crop_transform(self):
        """Test applying a crop (perspective) transform"""
        # Create a test image
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        # Create a crop transform with perspective matrix
        transform_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        transform = Transform(
            transform_type='crop',
            params={
                'transform_matrix': transform_matrix.tolist(),
                'output_size': [80, 80]
            }
        )
        
        result = apply_single_transform(image, transform)
        
        assert result.shape == (80, 80, 3)
        assert result.dtype == image.dtype
    
    def test_apply_deskew_transform(self):
        """Test applying a deskew (affine) transform"""
        # Create a test image
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        # Create a deskew transform with affine matrix
        transform_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        transform = Transform(
            transform_type='deskew',
            params={
                'transform_matrix': transform_matrix.tolist(),
                'output_size': [100, 100]
            }
        )
        
        result = apply_single_transform(image, transform)
        
        assert result.shape == (100, 100, 3)
        assert result.dtype == image.dtype
    
    def test_apply_rotation_90(self):
        """Test applying a 90-degree rotation"""
        image = np.ones((100, 200, 3), dtype=np.uint8) * 128
        
        transform = Transform(
            transform_type='rotation',
            params={'angle': 90, 'rotation_type': 'standard'}
        )
        
        result = apply_single_transform(image, transform)
        
        # 90-degree rotation swaps width and height
        assert result.shape == (200, 100, 3)
    
    def test_apply_rotation_180(self):
        """Test applying a 180-degree rotation"""
        image = np.ones((100, 200, 3), dtype=np.uint8) * 128
        
        transform = Transform(
            transform_type='rotation',
            params={'angle': 180, 'rotation_type': 'standard'}
        )
        
        result = apply_single_transform(image, transform)
        
        # 180-degree rotation keeps same dimensions
        assert result.shape == (100, 200, 3)
    
    def test_apply_rotation_270(self):
        """Test applying a 270-degree rotation"""
        image = np.ones((100, 200, 3), dtype=np.uint8) * 128
        
        transform = Transform(
            transform_type='rotation',
            params={'angle': 270, 'rotation_type': 'standard'}
        )
        
        result = apply_single_transform(image, transform)
        
        # 270-degree rotation swaps width and height
        assert result.shape == (200, 100, 3)
    
    def test_apply_rotation_arbitrary(self):
        """Test applying an arbitrary rotation"""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        transform = Transform(
            transform_type='rotation',
            params={'angle': 45, 'rotation_type': 'arbitrary'}
        )
        
        result = apply_single_transform(image, transform)
        
        # Arbitrary rotation keeps same dimensions
        assert result.shape == (100, 100, 3)
    
    def test_apply_unknown_transform(self):
        """Test applying an unknown transform type (should return original)"""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        transform = Transform(
            transform_type='unknown_transform',
            params={}
        )
        
        result = apply_single_transform(image, transform)
        
        # Should return original image unchanged
        assert np.array_equal(result, image)


class TestApplyTransformSequence:
    """Tests for apply_transform_sequence()"""
    
    def test_apply_multiple_transforms(self):
        """Test applying a sequence of transforms"""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        # Create a sequence with multiple transforms
        sequence = TransformSequence(
            input_path="/input/image.png",
            output_path="/output/image.png"
        )
        
        # Add rotation
        sequence.add_transform(Transform(
            transform_type='rotation',
            params={'angle': 90, 'rotation_type': 'standard'},
            order=0
        ))
        
        # Add another rotation (should be applied in order)
        sequence.add_transform(Transform(
            transform_type='rotation',
            params={'angle': 90, 'rotation_type': 'standard'},
            order=1
        ))
        
        result = apply_transform_sequence(image, sequence)
        
        # Two 90-degree rotations = 180 degrees, so dimensions should be back to original
        assert result.shape == (100, 100, 3)
    
    def test_apply_empty_sequence(self):
        """Test applying an empty transform sequence"""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        sequence = TransformSequence(
            input_path="/input/image.png",
            output_path="/output/image.png"
        )
        
        result = apply_transform_sequence(image, sequence)
        
        # Should return original image
        assert np.array_equal(result, image)


class TestApplyTransformsFromFile:
    """Tests for apply_transforms_from_file()"""
    
    def test_apply_transforms_from_file(self):
        """Test loading and applying transforms from a file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output", "image.png")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Create a transform sequence and save it
            sequence = TransformSequence(
                input_path="/input/image.png",
                output_path=output_path
            )
            sequence.add_transform(Transform(
                transform_type='rotation',
                params={'angle': 90, 'rotation_type': 'standard'},
                order=0
            ))
            
            # Save the transform file
            from src.utils.transform_handler import save_transforms
            save_transforms(output_path, sequence)
            
            # Create test image
            image = np.ones((100, 200, 3), dtype=np.uint8) * 128
            
            # Apply transforms from file
            result = apply_transforms_from_file(image, output_path)
            
            assert result is not None
            assert result.shape == (200, 100, 3)  # Rotated 90 degrees
    
    def test_apply_transforms_from_file_nonexistent(self):
        """Test applying transforms when file doesn't exist"""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        result = apply_transforms_from_file(image, "/nonexistent/image.png")
        
        assert result is None

