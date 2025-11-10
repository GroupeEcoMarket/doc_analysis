"""
Tests for API endpoints
"""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()


def test_health():
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_pipeline_status():
    """Test pipeline status endpoint"""
    response = client.get("/api/v1/pipeline/status")
    assert response.status_code == 200
    assert "stages" in response.json()

