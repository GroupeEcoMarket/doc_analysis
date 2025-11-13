"""
Unit tests for pdf_handler.py
"""

import pytest
import tempfile
import os
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.utils.pdf_handler import is_pdf, pdf_to_images, pdf_buffer_to_images, save_image_from_pdf


class TestIsPdf:
    """Tests for is_pdf()"""
    
    def test_is_pdf_with_pdf_extension(self):
        """Test is_pdf() with .pdf extension"""
        assert is_pdf("document.pdf") is True
        assert is_pdf("/path/to/document.pdf") is True
        assert is_pdf("DOCUMENT.PDF") is True  # Case insensitive
    
    def test_is_pdf_without_pdf_extension(self):
        """Test is_pdf() with non-PDF extensions"""
        assert is_pdf("document.png") is False
        assert is_pdf("document.jpg") is False
        assert is_pdf("document.txt") is False
        assert is_pdf("document") is False


class TestPdfToImages:
    """Tests for pdf_to_images()"""
    
    @patch('src.utils.pdf_handler.PYMUPDF_AVAILABLE', True)
    @patch('src.utils.pdf_handler.fitz')
    def test_pdf_to_images_with_pymupdf(self, mock_fitz):
        """Test pdf_to_images() using PyMuPDF"""
        # Mock PyMuPDF document
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_pix = MagicMock()
        
        # Setup mock pixmap
        mock_pix.height = 100
        mock_pix.width = 200
        mock_pix.n = 3  # RGB
        mock_pix.samples = np.ones((100, 200, 3), dtype=np.uint8).tobytes()
        
        mock_page.get_pixmap.return_value = mock_pix
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz.open.return_value = mock_doc
        
        # Create a temporary PDF file path
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            pdf_path = tmp_file.name
        
        try:
            images = pdf_to_images(pdf_path, dpi=300)
            
            assert len(images) == 1
            assert isinstance(images[0], np.ndarray)
            assert images[0].shape == (100, 200, 3)  # BGR format
            mock_doc.close.assert_called_once()
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)
    
    @patch('src.utils.pdf_handler.PYMUPDF_AVAILABLE', False)
    @patch('src.utils.pdf_handler.PDF2IMAGE_AVAILABLE', False)
    def test_pdf_to_images_no_library_available(self):
        """Test pdf_to_images() when no PDF library is available"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            pdf_path = tmp_file.name
        
        try:
            with pytest.raises(RuntimeError, match="No PDF conversion library available"):
                pdf_to_images(pdf_path)
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)
    
    @patch('src.utils.pdf_handler.PYMUPDF_AVAILABLE', True)
    @patch('src.utils.pdf_handler.fitz')
    def test_pdf_to_images_multiple_pages(self, mock_fitz):
        """Test pdf_to_images() with multiple pages"""
        mock_doc = MagicMock()
        mock_page1 = MagicMock()
        mock_page2 = MagicMock()
        mock_pix1 = MagicMock()
        mock_pix2 = MagicMock()
        
        # Setup mock pixmaps
        for mock_pix, h, w in [(mock_pix1, 100, 200), (mock_pix2, 150, 250)]:
            mock_pix.height = h
            mock_pix.width = w
            mock_pix.n = 3
            mock_pix.samples = np.ones((h, w, 3), dtype=np.uint8).tobytes()
        
        mock_page1.get_pixmap.return_value = mock_pix1
        mock_page2.get_pixmap.return_value = mock_pix2
        mock_doc.__len__.return_value = 2
        mock_doc.__getitem__.side_effect = [mock_page1, mock_page2]
        mock_fitz.open.return_value = mock_doc
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            pdf_path = tmp_file.name
        
        try:
            images = pdf_to_images(pdf_path, dpi=300)
            
            assert len(images) == 2
            assert images[0].shape == (100, 200, 3)
            assert images[1].shape == (150, 250, 3)
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)
    
    @patch('src.utils.pdf_handler.PYMUPDF_AVAILABLE', True)
    @patch('src.utils.pdf_handler.fitz')
    def test_pdf_to_images_rgba_format(self, mock_fitz):
        """Test pdf_to_images() with RGBA format pixmap"""
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_pix = MagicMock()
        
        mock_pix.height = 100
        mock_pix.width = 200
        mock_pix.n = 4  # RGBA
        mock_pix.samples = np.ones((100, 200, 4), dtype=np.uint8).tobytes()
        
        mock_page.get_pixmap.return_value = mock_pix
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz.open.return_value = mock_doc
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            pdf_path = tmp_file.name
        
        try:
            images = pdf_to_images(pdf_path, dpi=300)
            assert len(images) == 1
            assert images[0].shape == (100, 200, 3)  # Converted to BGR
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)
    
    @patch('src.utils.pdf_handler.PYMUPDF_AVAILABLE', True)
    @patch('src.utils.pdf_handler.fitz')
    def test_pdf_to_images_grayscale_format(self, mock_fitz):
        """Test pdf_to_images() with grayscale format pixmap"""
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_pix = MagicMock()
        
        mock_pix.height = 100
        mock_pix.width = 200
        mock_pix.n = 1  # Grayscale
        mock_pix.samples = np.ones((100, 200), dtype=np.uint8).tobytes()
        
        mock_page.get_pixmap.return_value = mock_pix
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz.open.return_value = mock_doc
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            pdf_path = tmp_file.name
        
        try:
            images = pdf_to_images(pdf_path, dpi=300)
            assert len(images) == 1
            assert images[0].shape == (100, 200, 3)  # Converted to BGR
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)
    
    @patch('src.utils.pdf_handler.PYMUPDF_AVAILABLE', True)
    @patch('src.utils.pdf_handler.fitz')
    def test_pdf_to_images_pymupdf_error(self, mock_fitz):
        """Test pdf_to_images() when PyMuPDF raises an error"""
        mock_fitz.open.side_effect = Exception("PyMuPDF error")
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            pdf_path = tmp_file.name
        
        try:
            with pytest.raises(RuntimeError, match="PyMuPDF conversion failed"):
                pdf_to_images(pdf_path, dpi=300)
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)


class TestSaveImageFromPdf:
    """Tests for save_image_from_pdf()"""
    
    @patch('src.utils.pdf_handler.pdf_to_images')
    @patch('src.utils.pdf_handler.cv2')
    def test_save_image_from_pdf_success(self, mock_cv2, mock_pdf_to_images):
        """Test save_image_from_pdf() successfully saves an image"""
        # Mock pdf_to_images to return a test image
        test_image = np.ones((100, 200, 3), dtype=np.uint8) * 128
        mock_pdf_to_images.return_value = [test_image]
        mock_cv2.imwrite.return_value = True
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = os.path.join(tmpdir, "test.pdf")
            output_path = os.path.join(tmpdir, "output.png")
            
            # Create dummy PDF file
            with open(pdf_path, 'w') as f:
                f.write("dummy pdf content")
            
            result = save_image_from_pdf(pdf_path, output_path, page_num=0, dpi=300)
            
            assert result == output_path
            mock_cv2.imwrite.assert_called_once_with(output_path, test_image)
    
    @patch('src.utils.pdf_handler.pdf_to_images')
    def test_save_image_from_pdf_invalid_page(self, mock_pdf_to_images):
        """Test save_image_from_pdf() with invalid page number"""
        # Mock pdf_to_images to return only 1 page
        test_image = np.ones((100, 200, 3), dtype=np.uint8) * 128
        mock_pdf_to_images.return_value = [test_image]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = os.path.join(tmpdir, "test.pdf")
            output_path = os.path.join(tmpdir, "output.png")
            
            # Create dummy PDF file
            with open(pdf_path, 'w') as f:
                f.write("dummy pdf content")
            
            with pytest.raises(ValueError, match="Page 5 n'existe pas"):
                save_image_from_pdf(pdf_path, output_path, page_num=5, dpi=300)


class TestPdfBufferToImages:
    """Tests for pdf_buffer_to_images()"""
    
    @patch('src.utils.pdf_handler.PYMUPDF_AVAILABLE', True)
    @patch('src.utils.pdf_handler.fitz')
    def test_pdf_buffer_to_images_with_pymupdf(self, mock_fitz):
        """Test pdf_buffer_to_images() using PyMuPDF"""
        # Mock PyMuPDF document
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_pix = MagicMock()
        
        # Setup mock pixmap
        mock_pix.height = 100
        mock_pix.width = 200
        mock_pix.n = 3  # RGB
        mock_pix.samples = np.ones((100, 200, 3), dtype=np.uint8).tobytes()
        
        mock_page.get_pixmap.return_value = mock_pix
        mock_doc.__len__.return_value = 1
        mock_doc.__iter__.return_value = iter([mock_page])
        mock_fitz.open.return_value = mock_doc
        mock_fitz.Matrix.return_value = MagicMock()
        
        # Create a dummy PDF buffer
        pdf_buffer = b"%PDF-1.4\n1 0 obj\nendobj\nxref\ntrailer\n%%EOF"
        
        images = pdf_buffer_to_images(pdf_buffer, dpi=300)
        
        assert len(images) == 1
        assert isinstance(images[0], np.ndarray)
        assert images[0].shape == (100, 200, 3)  # BGR format
        mock_doc.close.assert_called_once()
    
    @patch('src.utils.pdf_handler.PYMUPDF_AVAILABLE', False)
    def test_pdf_buffer_to_images_no_library_available(self):
        """Test pdf_buffer_to_images() when PyMuPDF is not available"""
        pdf_buffer = b"%PDF-1.4\n1 0 obj\nendobj\nxref\ntrailer\n%%EOF"
        
        with pytest.raises(RuntimeError, match="PyMuPDF n'est pas installé"):
            pdf_buffer_to_images(pdf_buffer, dpi=300)
    
    @patch('src.utils.pdf_handler.PYMUPDF_AVAILABLE', True)
    @patch('src.utils.pdf_handler.fitz')
    def test_pdf_buffer_to_images_empty_pdf(self, mock_fitz):
        """Test pdf_buffer_to_images() with empty PDF"""
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 0
        mock_fitz.open.return_value = mock_doc
        
        pdf_buffer = b"%PDF-1.4\n%%EOF"
        
        with pytest.raises(ValueError, match="Le PDF est vide"):
            pdf_buffer_to_images(pdf_buffer, dpi=300)
        
        mock_doc.close.assert_called_once()
    
    @patch('src.utils.pdf_handler.PYMUPDF_AVAILABLE', True)
    @patch('src.utils.pdf_handler.fitz')
    def test_pdf_buffer_to_images_multiple_pages(self, mock_fitz):
        """Test pdf_buffer_to_images() with multiple pages"""
        mock_doc = MagicMock()
        mock_page1 = MagicMock()
        mock_page2 = MagicMock()
        mock_pix1 = MagicMock()
        mock_pix2 = MagicMock()
        
        # Setup mock pixmaps
        for mock_pix, h, w in [(mock_pix1, 100, 200), (mock_pix2, 150, 250)]:
            mock_pix.height = h
            mock_pix.width = w
            mock_pix.n = 3
            mock_pix.samples = np.ones((h, w, 3), dtype=np.uint8).tobytes()
        
        mock_page1.get_pixmap.return_value = mock_pix1
        mock_page2.get_pixmap.return_value = mock_pix2
        mock_doc.__len__.return_value = 2
        mock_doc.__iter__.return_value = iter([mock_page1, mock_page2])
        mock_fitz.open.return_value = mock_doc
        mock_fitz.Matrix.return_value = MagicMock()
        
        pdf_buffer = b"%PDF-1.4\n1 0 obj\nendobj\nxref\ntrailer\n%%EOF"
        
        images = pdf_buffer_to_images(pdf_buffer, dpi=300)
        
        assert len(images) == 2
        assert images[0].shape == (100, 200, 3)
        assert images[1].shape == (150, 250, 3)
    
    @patch('src.utils.pdf_handler.PYMUPDF_AVAILABLE', True)
    @patch('src.utils.pdf_handler.fitz')
    def test_pdf_buffer_to_images_with_min_dpi(self, mock_fitz):
        """Test pdf_buffer_to_images() respects min_dpi parameter"""
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_pix = MagicMock()
        
        mock_pix.height = 100
        mock_pix.width = 200
        mock_pix.n = 3
        mock_pix.samples = np.ones((100, 200, 3), dtype=np.uint8).tobytes()
        
        mock_page.get_pixmap.return_value = mock_pix
        mock_doc.__len__.return_value = 1
        mock_doc.__iter__.return_value = iter([mock_page])
        mock_fitz.open.return_value = mock_doc
        mock_matrix = MagicMock()
        mock_fitz.Matrix.return_value = mock_matrix
        
        pdf_buffer = b"%PDF-1.4\n1 0 obj\nendobj\nxref\ntrailer\n%%EOF"
        
        # Test avec min_dpi plus élevé que dpi
        pdf_buffer_to_images(pdf_buffer, dpi=200, min_dpi=300)
        
        # Vérifier que Matrix a été appelé avec le bon DPI (max(200, 300) = 300)
        mock_fitz.Matrix.assert_called()
        # Le premier argument devrait être 300/72 (min_dpi est utilisé)
        call_args = mock_fitz.Matrix.call_args[0]
        assert call_args[0] == pytest.approx(300 / 72)
    
    @patch('src.utils.pdf_handler.PYMUPDF_AVAILABLE', True)
    @patch('src.utils.pdf_handler.fitz')
    def test_pdf_buffer_to_images_error_handling(self, mock_fitz):
        """Test pdf_buffer_to_images() error handling"""
        mock_fitz.open.side_effect = Exception("PyMuPDF error")
        
        pdf_buffer = b"%PDF-1.4\n1 0 obj\nendobj\nxref\ntrailer\n%%EOF"
        
        with pytest.raises(RuntimeError, match="Impossible de convertir le PDF"):
            pdf_buffer_to_images(pdf_buffer, dpi=300)

