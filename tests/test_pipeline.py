"""
Tests for pipeline stages
"""

import pytest
from src.pipeline import ColometryNormalizer, GeometryNormalizer, FeatureExtractor


def test_colometry_normalizer():
    """Test ColometryNormalizer initialization"""
    normalizer = ColometryNormalizer()
    assert normalizer is not None


def test_geometry_normalizer():
    """Test GeometryNormalizer initialization"""
    normalizer = GeometryNormalizer()
    assert normalizer is not None


def test_feature_extractor():
    """Test FeatureExtractor initialization"""
    extractor = FeatureExtractor()
    assert extractor is not None

