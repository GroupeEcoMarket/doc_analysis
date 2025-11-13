# services/ocr_service/__init__.py
"""
Microservice OCR isolé.

Ce package contient un service OCR isolé qui peut être déployé indépendamment
du reste du monorepo doc_analysis.

Le service expose des acteurs Dramatiq pour le traitement OCR via PaddleOCR.
"""

__version__ = "0.1.0"

# Exporter les acteurs principaux pour faciliter l'importation
from .actors import (
    perform_ocr_task,
    warmup_ocr_worker,
    init_ocr_engine,
    extract_ocr_lines
)

__all__ = [
    'perform_ocr_task',
    'warmup_ocr_worker',
    'init_ocr_engine',
    'extract_ocr_lines'
]

# Note: Pour utiliser ces acteurs depuis l'application principale,
# utilisez plutôt src.utils.ocr_client qui contient les acteurs proxy
# configurés pour pointer vers la queue ocr-queue.

