"""
Unit tests for file_handler.py
"""

import pytest
import tempfile
import os
from pathlib import Path
from src.utils.file_handler import ensure_dir, get_files, get_output_path


class TestEnsureDir:
    """Tests for ensure_dir()"""
    
    def test_ensure_dir_creates_directory(self):
        """Test that ensure_dir creates a new directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, "new", "nested", "directory")
            ensure_dir(new_dir)
            assert os.path.exists(new_dir)
            assert os.path.isdir(new_dir)
    
    def test_ensure_dir_existing_directory(self):
        """Test that ensure_dir doesn't fail if directory already exists"""
        with tempfile.TemporaryDirectory() as tmpdir:
            ensure_dir(tmpdir)
            # Should not raise an error
            assert os.path.exists(tmpdir)
    
    def test_ensure_dir_nested_paths(self):
        """Test that ensure_dir creates nested directories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "level1", "level2", "level3")
            ensure_dir(nested)
            assert os.path.exists(nested)
            assert os.path.isdir(nested)


class TestGetFiles:
    """Tests for get_files()"""
    
    def test_get_files_all_extensions(self):
        """Test get_files() with default extensions"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            test_files = [
                "test1.pdf",
                "test2.png",
                "test3.jpg",
                "test4.jpeg",
                "test5.tiff",
                "test6.bmp",
                "test7.txt",  # Should be ignored
                "test8.doc"   # Should be ignored
            ]
            
            for filename in test_files:
                filepath = os.path.join(tmpdir, filename)
                with open(filepath, 'w') as f:
                    f.write("test content")
            
            files = get_files(tmpdir)
            
            # Should find 6 image/PDF files, sorted
            assert len(files) == 6
            assert all(any(ext in f.lower() for ext in ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp']) 
                      for f in files)
            assert "test7.txt" not in str(files)
            assert "test8.doc" not in str(files)
    
    def test_get_files_custom_extensions(self):
        """Test get_files() with custom extensions"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            test_files = [
                "test1.pdf",
                "test2.png",
                "test3.txt",
                "test4.doc"
            ]
            
            for filename in test_files:
                filepath = os.path.join(tmpdir, filename)
                with open(filepath, 'w') as f:
                    f.write("test content")
            
            files = get_files(tmpdir, extensions=['.txt', '.doc'])
            
            # Should find only .txt and .doc files
            assert len(files) == 2
            assert all(any(ext in f for ext in ['.txt', '.doc']) for f in files)
            assert "test1.pdf" not in str(files)
            assert "test2.png" not in str(files)
    
    def test_get_files_case_insensitive(self):
        """Test that get_files() is case-insensitive"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files with different cases
            test_files = [
                "test1.PDF",
                "test2.Png",
                "test3.JPG",
                "test4.jpeg"
            ]
            
            for filename in test_files:
                filepath = os.path.join(tmpdir, filename)
                with open(filepath, 'w') as f:
                    f.write("test content")
            
            files = get_files(tmpdir, extensions=['.pdf', '.png', '.jpg', '.jpeg'])
            
            # Should find all files regardless of case
            assert len(files) == 4
    
    def test_get_files_nested_directories(self):
        """Test get_files() with nested directories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested structure
            subdir1 = os.path.join(tmpdir, "subdir1")
            subdir2 = os.path.join(tmpdir, "subdir2", "nested")
            os.makedirs(subdir1)
            os.makedirs(subdir2)
            
            # Create files in different locations
            files_to_create = [
                os.path.join(tmpdir, "root.pdf"),
                os.path.join(subdir1, "sub1.png"),
                os.path.join(subdir2, "sub2.jpg")
            ]
            
            for filepath in files_to_create:
                with open(filepath, 'w') as f:
                    f.write("test content")
            
            files = get_files(tmpdir)
            
            # Should find all 3 files
            assert len(files) == 3
            assert any("root.pdf" in f for f in files)
            assert any("sub1.png" in f for f in files)
            assert any("sub2.jpg" in f for f in files)
    
    def test_get_files_empty_directory(self):
        """Test get_files() with empty directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = get_files(tmpdir)
            assert files == []


class TestGetOutputPath:
    """Tests for get_output_path()"""
    
    def test_get_output_path_basic(self):
        """Test get_output_path() without suffix"""
        input_path = "/input/documents/image.png"
        output_dir = "/output/processed"
        result = get_output_path(input_path, output_dir)
        
        expected = os.path.join("/output/processed", "image.png")
        assert os.path.normpath(result) == os.path.normpath(expected)
    
    def test_get_output_path_with_suffix(self):
        """Test get_output_path() with suffix"""
        input_path = "/input/documents/image.png"
        output_dir = "/output/processed"
        result = get_output_path(input_path, output_dir, suffix="processed")
        
        expected = os.path.join("/output/processed", "image_processed.png")
        assert os.path.normpath(result) == os.path.normpath(expected)
    
    def test_get_output_path_preserves_extension(self):
        """Test that get_output_path() preserves file extension"""
        input_path = "/input/documents/document.pdf"
        output_dir = "/output/processed"
        result = get_output_path(input_path, output_dir)
        
        assert result.endswith(".pdf")
        assert "document.pdf" in result
    
    def test_get_output_path_with_complex_filename(self):
        """Test get_output_path() with complex filename"""
        input_path = "/input/documents/my-document_file-2024.png"
        output_dir = "/output/processed"
        result = get_output_path(input_path, output_dir, suffix="enhanced")
        
        expected = os.path.join("/output/processed", "my-document_file-2024_enhanced.png")
        assert os.path.normpath(result) == os.path.normpath(expected)
    
    def test_get_output_path_relative_paths(self):
        """Test get_output_path() with relative paths"""
        input_path = "image.jpg"
        output_dir = "output"
        result = get_output_path(input_path, output_dir)
        
        expected = os.path.join("output", "image.jpg")
        assert os.path.normpath(result) == os.path.normpath(expected)

