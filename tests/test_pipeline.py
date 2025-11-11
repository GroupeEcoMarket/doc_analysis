"""
Tests for pipeline stages
"""

import pytest
from src.pipeline.colometry import ColometryNormalizer
from src.pipeline.geometry import GeometryNormalizer
from src.pipeline.features import FeatureExtractor
from src.models.registry import ModelRegistry


def test_colometry_normalizer(mock_app_config):
    """Test ColometryNormalizer initialization"""
    normalizer = ColometryNormalizer(app_config=mock_app_config)
    assert normalizer is not None
    assert normalizer.config is not None


def test_geometry_normalizer(
    mock_geometry_config,
    mock_qa_config,
    mock_performance_config,
    mock_output_config,
    mock_model_registry
):
    """Test GeometryNormalizer initialization"""
    normalizer = GeometryNormalizer(
        geo_config=mock_geometry_config,
        qa_config=mock_qa_config,
        perf_config=mock_performance_config,
        output_config=mock_output_config,
        model_registry=mock_model_registry
    )
    assert normalizer is not None
    assert normalizer.model_registry is not None
    assert isinstance(normalizer.model_registry, ModelRegistry)


def test_geometry_normalizer_with_model_registry(
    mock_geometry_config,
    mock_qa_config,
    mock_performance_config,
    mock_output_config
):
    """Test GeometryNormalizer with injected ModelRegistry"""
    # Test with injected ModelRegistry
    registry = ModelRegistry(lazy_load=True)
    normalizer = GeometryNormalizer(
        geo_config=mock_geometry_config,
        qa_config=mock_qa_config,
        perf_config=mock_performance_config,
        output_config=mock_output_config,
        model_registry=registry
    )
    assert normalizer is not None
    assert normalizer.model_registry is registry


def test_feature_extractor(mock_app_config):
    """Test FeatureExtractor initialization"""
    extractor = FeatureExtractor(app_config=mock_app_config)
    assert extractor is not None
    assert extractor.config is not None

