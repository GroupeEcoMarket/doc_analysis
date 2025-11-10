"""
Tests pour la gestion des PDFs
"""

import pytest
import os
from pathlib import Path
from src.utils.pdf_handler import is_pdf, pdf_to_images
from src.pipeline.geometry import GeometryNormalizer


def test_is_pdf():
    """Test de détection PDF"""
    assert is_pdf("test.pdf") == True
    assert is_pdf("test.PDF") == True
    assert is_pdf("test.png") == False
    assert is_pdf("test.jpg") == False


@pytest.mark.skipif(
    not os.path.exists("data/input/test.pdf"),
    reason="Fichier PDF de test non trouvé"
)
def test_pdf_to_images():
    """Test de conversion PDF en images"""
    pdf_path = "data/input/test.pdf"
    images = pdf_to_images(pdf_path)
    assert len(images) > 0
    assert images[0].shape[2] == 3  # BGR image


@pytest.mark.skipif(
    not os.path.exists("data/input/test.pdf"),
    reason="Fichier PDF de test non trouvé"
)
def test_geometry_normalizer_with_pdf():
    """Test du pipeline géométrique avec un PDF"""
    normalizer = GeometryNormalizer()
    
    input_pdf = "data/input/test.pdf"
    output_dir = "data/output/test_geometry"
    os.makedirs(output_dir, exist_ok=True)
    
    results = normalizer.process_batch("data/input", output_dir)
    
    # Vérifier qu'il y a des résultats
    assert len(results) > 0
    
    # Vérifier que les fichiers de sortie existent
    for result in results:
        if 'output_path' in result:
            assert os.path.exists(result['output_path'])

