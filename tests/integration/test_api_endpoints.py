"""
Tests for API endpoints
"""

import pytest
import os
import numpy as np
import cv2
from pathlib import Path
from fastapi.testclient import TestClient
from src.api.app import app


@pytest.fixture(scope="session")
def client():
    """
    Fixture that creates a test client for the API.
    Executes once per test session.
    """
    return TestClient(app)


@pytest.fixture(scope="session")
def test_file_path():
    """
    Fixture that provides the path to a test file.
    - Searches for an image or PDF file in tests/fixtures/.
    - If none is found, creates a basic one (test_image.png).
    Executes once per test session.
    """
    fixtures_dir = Path(__file__).parent.parent / "fixtures"
    
    # Search for an existing file first
    for ext in ['.jpg', '.png', '.jpeg', '.pdf']:
        found_files = list(fixtures_dir.glob(f"*{ext}"))
        if found_files:
            return found_files[0]  # Return the first file found
    
    # If no file is found, create one
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    fallback_path = fixtures_dir / "test_image.png"
    
    # Only create it if it doesn't really exist
    if not fallback_path.exists():
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        cv2.imwrite(str(fallback_path), test_image)
        
    return fallback_path


def test_root(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()


def test_health(client):
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_pipeline_status(client):
    """Test pipeline status endpoint"""
    response = client.get("/api/v1/pipeline/status")
    assert response.status_code == 200
    assert "stages" in response.json()


def test_pipeline_geometry_endpoint(client, test_file_path):
    """
    Test the geometry normalization endpoint using a fixture.
    
    This integration test:
    - Uploads a real file (PNG or PDF)
    - Verifies that processing completes successfully
    - Verifies the JSON response structure
    """
    # Determine MIME type based on file extension
    mime_type = "image/png"
    if test_file_path.suffix.lower() == '.pdf':
        mime_type = "application/pdf"
    elif test_file_path.suffix.lower() in ['.jpg', '.jpeg']:
        mime_type = "image/jpeg"
    
    with open(test_file_path, "rb") as test_file:
        response = client.post(
            "/api/v1/pipeline/geometry?sync=true",  # Utiliser le mode synchrone pour le test d'intégration
            files={"file": (test_file_path.name, test_file, mime_type)}
        )
    
    # Verify response status
    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"
    
    # Verify that the response is valid JSON
    json_response = response.json()
    assert isinstance(json_response, dict)
    
    # Verify expected keys in the response
    assert "status" in json_response, "Missing 'status' key in response"
    assert json_response["status"] == "success", f"Expected 'success', got '{json_response.get('status')}'"
    
    assert "input_filename" in json_response, "Missing 'input_filename' key in response"
    assert "output_files" in json_response, "Missing 'output_files' key in response"
    assert "metadata" in json_response, "Missing 'metadata' key in response"
    
    # Verify output_files structure
    output_files = json_response["output_files"]
    assert isinstance(output_files, dict), "output_files should be a dictionary"
    assert "transformed" in output_files, "Missing 'transformed' key in output_files"
    
    # Verify metadata structure
    metadata = json_response["metadata"]
    assert isinstance(metadata, dict), "metadata should be a dictionary"
    assert "crop_applied" in metadata, "Missing 'crop_applied' key in metadata"
    assert "deskew_applied" in metadata, "Missing 'deskew_applied' key in metadata"
    assert "rotation_applied" in metadata, "Missing 'rotation_applied' key in metadata"
    assert "orientation_angle" in metadata, "Missing 'orientation_angle' key in metadata"
    assert "capture_type" in metadata, "Missing 'capture_type' key in metadata"
    assert "processing_time" in metadata, "Missing 'processing_time' key in metadata"
    
    # Verify types of values in metadata
    assert isinstance(metadata["crop_applied"], bool), "crop_applied should be a boolean"
    assert isinstance(metadata["deskew_applied"], bool), "deskew_applied should be a boolean"
    assert isinstance(metadata["rotation_applied"], bool), "rotation_applied should be a boolean"
    assert isinstance(metadata["orientation_angle"], (int, float)), "orientation_angle should be a number"
    assert isinstance(metadata["capture_type"], str), "capture_type should be a string"
    assert isinstance(metadata["processing_time"], (int, float)), "processing_time should be a number"
    
    # Verify that qa_flags is present (can be an empty dict)
    assert "qa_flags" in json_response, "Missing 'qa_flags' key in response"


def test_pipeline_geometry_endpoint_invalid_file(client):
    """
    Test que l'endpoint retourne une erreur 400 pour un fichier invalide
    """
    # Créer un fichier avec une extension non supportée
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_file:
        tmp_file.write(b"This is not an image")
        tmp_file_path = tmp_file.name
    
    try:
        with open(tmp_file_path, 'rb') as f:
            response = client.post(
                "/api/v1/pipeline/geometry",
                files={"file": ("test.txt", f, "text/plain")}
            )
        
        # Vérifier que l'erreur est retournée
        assert response.status_code == 400, f"Expected 400 for invalid file, got {response.status_code}"
        assert "detail" in response.json(), "Error response should contain 'detail'"
        
    finally:
        # Nettoyer le fichier temporaire
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


def test_pipeline_geometry_endpoint_empty_file(client):
    """
    Test que l'endpoint retourne une erreur 400 pour un fichier vide
    """
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        # Fichier vide
        tmp_file_path = tmp_file.name
    
    try:
        with open(tmp_file_path, 'rb') as f:
            response = client.post(
                "/api/v1/pipeline/geometry",
                files={"file": ("empty.png", f, "image/png")}
            )
        
        # Vérifier que l'erreur est retournée
        assert response.status_code == 400, f"Expected 400 for empty file, got {response.status_code}"
        
    finally:
        # Nettoyer le fichier temporaire
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
