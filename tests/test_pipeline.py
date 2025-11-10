"""
Tests for pipeline stages
"""

import pytest
from src.pipeline import ColometryNormalizer, GeometryNormalizer, FeatureExtractor
from src.models.registry import ModelRegistry


def test_colometry_normalizer():
    """Test ColometryNormalizer initialization"""
    normalizer = ColometryNormalizer()
    assert normalizer is not None


def test_geometry_normalizer():
    """Test GeometryNormalizer initialization"""
    # GeometryNormalizer should work without parameters (creates ModelRegistry internally)
    normalizer = GeometryNormalizer()
    assert normalizer is not None
    assert normalizer.model_registry is not None
    assert isinstance(normalizer.model_registry, ModelRegistry)


def test_geometry_normalizer_with_model_registry():
    """Test GeometryNormalizer with injected ModelRegistry"""
    # Test with injected ModelRegistry
    registry = ModelRegistry()
    normalizer = GeometryNormalizer(model_registry=registry)
    assert normalizer is not None
    assert normalizer.model_registry is registry


def test_feature_extractor():
    """Test FeatureExtractor initialization"""
    extractor = FeatureExtractor()
    assert extractor is not None

