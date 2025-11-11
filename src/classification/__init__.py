"""
Module de classification de documents.

Ce module contient la logique de classification de type de document
basée sur des embeddings sémantiques et positionnels.
"""

from src.classification.feature_engineering import (
    extract_document_embedding,
    filter_ocr_lines,
    create_multimodal_embedding,
    aggregate_line_embeddings,
    FeatureEngineer
)
from src.classification.classifier_service import DocumentClassifier

__all__ = [
    'extract_document_embedding',
    'filter_ocr_lines',
    'create_multimodal_embedding',
    'aggregate_line_embeddings',
    'FeatureEngineer',
    'DocumentClassifier',
]

