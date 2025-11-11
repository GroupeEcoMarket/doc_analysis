"""
Tests pour la gestion des PDFs
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
from src.utils.pdf_handler import is_pdf, pdf_to_images
from src.pipeline.preprocessing import PreprocessingNormalizer
from src.pipeline.geometry import GeometryNormalizer


def test_is_pdf():
    """Test de détection PDF"""
    assert is_pdf("test.pdf") == True
    assert is_pdf("test.PDF") == True
    assert is_pdf("test.png") == False
    assert is_pdf("test.jpg") == False


def test_pdf_to_images():
    """Test de conversion PDF en images à partir d'un fichier fixture."""
    fixtures_dir = Path(__file__).parent.parent / "fixtures"
    pdf_path = fixtures_dir / "test.pdf"

    if not pdf_path.exists():
        pytest.skip("Le fichier PDF de test 'tests/fixtures/test.pdf' est manquant.")

    images = pdf_to_images(str(pdf_path))

    assert len(images) > 0
    assert images[0].shape[2] == 3  # Doit être une image BGR


def test_geometry_normalizer_with_pdf(app_config):
    """
    Teste l'enchaînement complet : Preprocessing (PDF->Image) -> Geometry.
    
    Utilise la VRAIE configuration pour s'assurer que l'application
    fonctionne correctement avec sa configuration réelle.
    """
    from src.utils.capture_classifier import CaptureClassifier
    from src.models.registry import ModelRegistry
    
    # 1. Initialiser les normaliseurs avec la vraie configuration
    capture_classifier = CaptureClassifier(
        white_level_threshold=app_config.geometry.capture_classifier_white_level_threshold,
        white_percentage_threshold=app_config.geometry.capture_classifier_white_percentage_threshold,
        enabled=app_config.geometry.capture_classifier_enabled
    )
    
    preprocessor = PreprocessingNormalizer(
        capture_classifier=capture_classifier,
        pdf_config=app_config.pdf
    )
    
    model_registry = ModelRegistry(lazy_load=app_config.performance.lazy_load_models)
    
    geometry_normalizer = GeometryNormalizer(
        geo_config=app_config.geometry,
        qa_config=app_config.qa,
        perf_config=app_config.performance,
        output_config=app_config.output,
        model_registry=model_registry
    )
    
    # 2. Préparer les répertoires temporaires
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        preproc_output_dir = Path(tmpdir) / "preprocessed"
        geometry_output_dir = Path(tmpdir) / "geometry_output"
        
        input_dir.mkdir()
        preproc_output_dir.mkdir()
        geometry_output_dir.mkdir()

        # 3. Copier notre fichier de test PDF dans le répertoire d'input
        fixture_path = Path(__file__).parent.parent / "fixtures" / "test.pdf"
        
        if not fixture_path.exists():
            pytest.skip("Le fichier PDF de test 'tests/fixtures/test.pdf' est manquant.")
        
        shutil.copy(fixture_path, input_dir)

        # --- ÉTAPE 1: EXÉCUTER LE PREPROCESSING ---
        preproc_results = preprocessor.process_batch(str(input_dir), str(preproc_output_dir))

        # Vérifier que le preprocessing a bien créé une image
        assert len(preproc_results) > 0
        assert preproc_results[0].status == 'success'

        # --- ÉTAPE 2: EXÉCUTER LA GÉOMÉTRIE SUR LA SORTIE DU PREPROCESSING ---
        # Le répertoire d'entrée pour la géométrie est la sortie du preprocessing
        geometry_results = geometry_normalizer.process_batch(str(preproc_output_dir), str(geometry_output_dir))

        # 4. Valider les résultats finaux
        # Note: process_batch retourne List[GeometryOutput]
        assert len(geometry_results) > 0, "GeometryNormalizer n'a retourné aucun résultat."
        assert geometry_results[0].status == 'success', f"Le traitement de la géométrie a échoué: {getattr(geometry_results[0], 'error', 'Unknown error')}"
        # process_batch retourne GeometryOutput avec output_transformed_path ou output_path
        output_path = geometry_results[0].output_transformed_path or geometry_results[0].output_path
        assert output_path, "Le chemin de sortie n'a pas été défini."
        assert Path(output_path).exists(), f"Le fichier de sortie de la géométrie n'a pas été créé: {output_path}"

