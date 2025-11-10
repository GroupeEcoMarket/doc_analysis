"""
PDF handling utilities
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[np.ndarray]:
    """
    Convertit un PDF en liste d'images
    
    Args:
        pdf_path: Chemin vers le fichier PDF
        dpi: Résolution DPI pour la conversion (défaut: 200)
        
    Returns:
        list: Liste d'images numpy (une par page)
    """
    images = []
    
    # Essayer d'abord avec pdf2image
    if PDF2IMAGE_AVAILABLE:
        try:
            pil_images = convert_from_path(pdf_path, dpi=dpi)
            for pil_img in pil_images:
                # Convertir PIL Image en numpy array (BGR pour OpenCV)
                img_array = np.array(pil_img)
                if len(img_array.shape) == 3:
                    # Convertir RGB vers BGR
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                images.append(img_array)
            return images
        except Exception as e:
            # pdf2image nécessite poppler, mais PyMuPDF fonctionne sans dépendances externes
            # On bascule silencieusement sur PyMuPDF qui est plus fiable
            pass
    
    # Fallback sur PyMuPDF (utilisé par défaut, plus fiable)
    if PYMUPDF_AVAILABLE:
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Rendre la page en image à 300 DPI minimum via get_pixmap
                # S'assurer que le DPI est au moins 300
                actual_dpi = max(dpi, 300)
                mat = fitz.Matrix(actual_dpi / 72, actual_dpi / 72)  # 72 est le DPI par défaut
                pix = page.get_pixmap(matrix=mat)
                # Convertir en numpy array
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                if pix.n == 4:  # RGBA
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                elif pix.n == 3:  # RGB
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                elif pix.n == 1:  # Grayscale
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                images.append(img_array)
            doc.close()
            return images
        except Exception as e:
            raise RuntimeError(f"PyMuPDF conversion failed: {e}")
    
    raise RuntimeError(
        "No PDF conversion library available. Install pdf2image or PyMuPDF: "
        "pip install pdf2image or pip install PyMuPDF"
    )


def is_pdf(file_path: str) -> bool:
    """
    Vérifie si un fichier est un PDF
    
    Args:
        file_path: Chemin vers le fichier
        
    Returns:
        bool: True si c'est un PDF
    """
    return Path(file_path).suffix.lower() == '.pdf'


def save_image_from_pdf(pdf_path: str, output_path: str, page_num: int = 0, dpi: int = 300) -> str:
    """
    Convertit une page d'un PDF en image et la sauvegarde
    
    Args:
        pdf_path: Chemin vers le PDF
        output_path: Chemin de sortie pour l'image
        page_num: Numéro de page (0-indexed)
        dpi: Résolution DPI
        
    Returns:
        str: Chemin de l'image sauvegardée
    """
    images = pdf_to_images(pdf_path, dpi=dpi)
    if page_num >= len(images):
        raise ValueError(f"Page {page_num} n'existe pas. Le PDF a {len(images)} page(s).")
    
    cv2.imwrite(output_path, images[page_num])
    return output_path

