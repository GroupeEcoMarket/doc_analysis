"""
Unit tests for transform_handler.py
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from src.utils.transform_handler import (
    Transform,
    TransformSequence,
    get_transform_file_path,
    save_transforms,
    load_transforms
)


class TestTransform:
    """Tests for Transform class"""
    
    def test_transform_initialization(self):
        """Test Transform initialization"""
        transform = Transform(
            transform_type="rotation",
            params={"angle": 90},
            order=1
        )
        assert transform.transform_type == "rotation"
        assert transform.params == {"angle": 90}
        assert transform.order == 1
    
    def test_transform_to_dict(self):
        """Test Transform.to_dict()"""
        transform = Transform(
            transform_type="crop",
            params={"x": 10, "y": 20, "width": 100, "height": 200},
            order=0
        )
        result = transform.to_dict()
        assert result["transform_type"] == "crop"
        assert result["params"] == {"x": 10, "y": 20, "width": 100, "height": 200}
        assert result["order"] == 0
    
    def test_transform_from_dict(self):
        """Test Transform.from_dict()"""
        data = {
            "transform_type": "deskew",
            "params": {"angle": 2.5},
            "order": 2
        }
        transform = Transform.from_dict(data)
        assert transform.transform_type == "deskew"
        assert transform.params == {"angle": 2.5}
        assert transform.order == 2
    
    def test_transform_from_dict_without_order(self):
        """Test Transform.from_dict() without order (should default to 0)"""
        data = {
            "transform_type": "rotation",
            "params": {"angle": 180}
        }
        transform = Transform.from_dict(data)
        assert transform.order == 0


class TestTransformSequence:
    """Tests for TransformSequence class"""
    
    def test_transform_sequence_initialization(self):
        """Test TransformSequence initialization"""
        seq = TransformSequence(
            input_path="/input/image.png",
            output_path="/output/image.png",
            output_original_path="/output/image_original.png"
        )
        assert seq.input_path == "/input/image.png"
        assert seq.output_path == "/output/image.png"
        assert seq.output_original_path == "/output/image_original.png"
        assert len(seq.transforms) == 0
    
    def test_add_transform(self):
        """Test adding transforms to sequence"""
        seq = TransformSequence(
            input_path="/input/image.png",
            output_path="/output/image.png"
        )
        
        transform1 = Transform("crop", {"x": 0, "y": 0}, order=0)
        transform2 = Transform("rotation", {"angle": 90}, order=1)
        transform3 = Transform("deskew", {"angle": 2.5}, order=2)
        
        # Add in wrong order
        seq.add_transform(transform3)
        seq.add_transform(transform1)
        seq.add_transform(transform2)
        
        # Should be sorted by order
        assert len(seq.transforms) == 3
        assert seq.transforms[0].transform_type == "crop"
        assert seq.transforms[1].transform_type == "rotation"
        assert seq.transforms[2].transform_type == "deskew"
    
    def test_to_dict(self):
        """Test TransformSequence.to_dict()"""
        seq = TransformSequence(
            input_path="/input/image.png",
            output_path="/output/image.png",
            output_original_path="/output/image_original.png"
        )
        seq.add_transform(Transform("crop", {"x": 10}, order=0))
        seq.add_transform(Transform("rotation", {"angle": 90}, order=1))
        
        result = seq.to_dict()
        assert result["input_path"] == "/input/image.png"
        assert result["output_path"] == "/output/image.png"
        assert result["output_original_path"] == "/output/image_original.png"
        assert len(result["transforms"]) == 2
        assert result["transforms"][0]["transform_type"] == "crop"
        assert result["transforms"][1]["transform_type"] == "rotation"
    
    def test_to_dict_without_original_path(self):
        """Test TransformSequence.to_dict() without output_original_path"""
        seq = TransformSequence(
            input_path="/input/image.png",
            output_path="/output/image.png"
        )
        result = seq.to_dict()
        assert "output_original_path" not in result
    
    def test_from_dict(self):
        """Test TransformSequence.from_dict()"""
        data = {
            "input_path": "/input/image.png",
            "output_path": "/output/image.png",
            "output_original_path": "/output/image_original.png",
            "transforms": [
                {"transform_type": "crop", "params": {"x": 10}, "order": 0},
                {"transform_type": "rotation", "params": {"angle": 90}, "order": 1}
            ]
        }
        seq = TransformSequence.from_dict(data)
        assert seq.input_path == "/input/image.png"
        assert seq.output_path == "/output/image.png"
        assert seq.output_original_path == "/output/image_original.png"
        assert len(seq.transforms) == 2
    
    def test_from_dict_legacy_format(self):
        """Test TransformSequence.from_dict() with legacy output_transformed_path"""
        data = {
            "input_path": "/input/image.png",
            "output_transformed_path": "/output/image.png",
            "transforms": []
        }
        seq = TransformSequence.from_dict(data)
        assert seq.output_path == "/output/image.png"
    
    def test_save_and_load(self):
        """Test saving and loading TransformSequence"""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.transform.json")
            
            seq = TransformSequence(
                input_path="/input/image.png",
                output_path="/output/image.png"
            )
            seq.add_transform(Transform("crop", {"x": 10}, order=0))
            seq.add_transform(Transform("rotation", {"angle": 90}, order=1))
            
            # Save
            seq.save(file_path)
            assert os.path.exists(file_path)
            
            # Load
            loaded_seq = TransformSequence.load(file_path)
            assert loaded_seq.input_path == seq.input_path
            assert loaded_seq.output_path == seq.output_path
            assert len(loaded_seq.transforms) == 2
            assert loaded_seq.transforms[0].transform_type == "crop"
            assert loaded_seq.transforms[1].transform_type == "rotation"


class TestTransformFunctions:
    """Tests for module-level functions"""
    
    def test_get_transform_file_path(self):
        """Test get_transform_file_path()"""
        result = get_transform_file_path("/output/image.png")
        # Normalize path for cross-platform compatibility
        expected = os.path.normpath("/output/image.transform.json")
        assert os.path.normpath(result) == expected
        
        result = get_transform_file_path("image.jpg")
        assert result == "image.transform.json"
    
    def test_save_and_load_transforms(self):
        """Test save_transforms() and load_transforms()"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output", "image.png")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            seq = TransformSequence(
                input_path="/input/image.png",
                output_path=output_path
            )
            seq.add_transform(Transform("crop", {"x": 10}, order=0))
            
            # Save
            save_transforms(output_path, seq)
            
            # Load
            loaded_seq = load_transforms(output_path)
            assert loaded_seq is not None
            assert loaded_seq.input_path == seq.input_path
            assert len(loaded_seq.transforms) == 1
    
    def test_load_transforms_nonexistent(self):
        """Test load_transforms() with non-existent file"""
        result = load_transforms("/nonexistent/image.png")
        assert result is None

