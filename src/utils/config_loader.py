"""
Configuration Loader
Charge et valide la configuration du pipeline depuis config.yaml
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class GeometryConfig:
    """Configuration pour la normalisation géométrique"""
    # Capture Classifier
    capture_classifier_enabled: bool
    capture_classifier_white_level_threshold: int
    capture_classifier_white_percentage_threshold: float
    capture_classifier_skip_crop_if_scan: bool
    
    # Orientation
    orientation_min_confidence: float
    
    # Crop
    crop_enabled: bool
    crop_min_area_ratio: float
    crop_max_margin_ratio: float
    
    # Deskew
    deskew_enabled: bool
    deskew_min_confidence: float
    deskew_max_angle: float
    deskew_min_angle: float
    deskew_hough_threshold: int
    
    # Quality
    quality_min_contrast: int
    quality_min_resolution_width: int
    quality_min_resolution_height: int


@dataclass
class PDFConfig:
    """Configuration pour le traitement PDF"""
    dpi: int
    min_dpi: int


@dataclass
class QAConfig:
    """Configuration pour les flags QA"""
    low_confidence_orientation: float
    overcrop_risk: float
    low_contrast: int
    too_small_width: int
    too_small_height: int


@dataclass
class PerformanceConfig:
    """Configuration pour les performances"""
    batch_size: int
    max_workers: int
    parallelization_threshold: int
    lazy_load_models: bool


@dataclass
class OutputConfig:
    """Configuration pour les sorties"""
    save_original: bool
    save_transformed: bool
    save_qa_flags: bool
    save_transforms: bool
    image_format: str
    jpeg_quality: int


class Config:
    """Classe principale de configuration"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialise la configuration
        
        Args:
            config_path: Chemin vers le fichier de configuration (défaut: config.yaml à la racine)
        """
        if config_path is None:
            # Chercher config.yaml à la racine du projet
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent
            config_path = project_root / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config_data = self._load_config()
        
        # Initialiser les sous-configurations
        self.geometry = self._load_geometry_config()
        self.pdf = self._load_pdf_config()
        self.qa = self._load_qa_config()
        self.performance = self._load_performance_config()
        self.output = self._load_output_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Charge le fichier YAML de configuration"""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Fichier de configuration introuvable: {self.config_path}\n"
                f"Veuillez créer un fichier config.yaml à la racine du projet."
            )
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        if not config_data:
            raise ValueError(f"Fichier de configuration vide: {self.config_path}")
        
        return config_data
    
    def _load_geometry_config(self) -> GeometryConfig:
        """Charge la configuration de normalisation géométrique"""
        geo = self._config_data.get('geometry', {})
        
        return GeometryConfig(
            capture_classifier_enabled=geo.get('capture_classifier', {}).get('enabled', True),
            capture_classifier_white_level_threshold=geo.get('capture_classifier', {}).get('white_level_threshold', 245),
            capture_classifier_white_percentage_threshold=geo.get('capture_classifier', {}).get('white_percentage_threshold', 0.70),
            capture_classifier_skip_crop_if_scan=geo.get('capture_classifier', {}).get('skip_crop_if_scan', True),
            orientation_min_confidence=geo.get('orientation', {}).get('min_confidence', 0.70),
            crop_enabled=geo.get('crop', {}).get('enabled', True),
            crop_min_area_ratio=geo.get('crop', {}).get('min_area_ratio', 0.85),
            crop_max_margin_ratio=geo.get('crop', {}).get('max_margin_ratio', 0.08),
            deskew_enabled=geo.get('deskew', {}).get('enabled', True),
            deskew_min_confidence=geo.get('deskew', {}).get('min_confidence', 0.20),
            deskew_max_angle=geo.get('deskew', {}).get('max_angle', 15.0),
            deskew_min_angle=geo.get('deskew', {}).get('min_angle', 0.5),
            deskew_hough_threshold=geo.get('deskew', {}).get('hough_threshold', 100),
            quality_min_contrast=geo.get('quality', {}).get('min_contrast', 50),
            quality_min_resolution_width=geo.get('quality', {}).get('min_resolution_width', 1200),
            quality_min_resolution_height=geo.get('quality', {}).get('min_resolution_height', 1600),
        )
    
    def _load_pdf_config(self) -> PDFConfig:
        """Charge la configuration PDF"""
        pdf = self._config_data.get('pdf', {})
        
        return PDFConfig(
            dpi=pdf.get('dpi', 300),
            min_dpi=pdf.get('min_dpi', 300)
        )
    
    def _load_qa_config(self) -> QAConfig:
        """
        Charge la configuration QA.
        """
        # Récupérer les valeurs de geometry (source unique de vérité)
        geo = self._config_data.get('geometry', {})
        geo_orientation = geo.get('orientation', {})
        geo_crop = geo.get('crop', {})
        geo_quality = geo.get('quality', {})
        
        # Utiliser TOUJOURS les valeurs de geometry (la section qa.* est ignorée)
        return QAConfig(
            low_confidence_orientation=geo_orientation.get('min_confidence', 0.70),
            overcrop_risk=geo_crop.get('max_margin_ratio', 0.08),
            low_contrast=geo_quality.get('min_contrast', 50),
            too_small_width=geo_quality.get('min_resolution_width', 1200),
            too_small_height=geo_quality.get('min_resolution_height', 1600)
        )
    
    def _load_performance_config(self) -> PerformanceConfig:
        """Charge la configuration de performance"""
        perf = self._config_data.get('performance', {})
        
        return PerformanceConfig(
            batch_size=perf.get('batch_size', 10),
            max_workers=perf.get('max_workers', 4),
            parallelization_threshold=perf.get('parallelization_threshold', 2),
            lazy_load_models=perf.get('lazy_load_models', True)
        )
    
    def _load_output_config(self) -> OutputConfig:
        """Charge la configuration de sortie"""
        output = self._config_data.get('output', {})
        
        return OutputConfig(
            save_original=output.get('save_original', True),
            save_transformed=output.get('save_transformed', True),
            save_qa_flags=output.get('save_qa_flags', True),
            save_transforms=output.get('save_transforms', True),
            image_format=output.get('image_format', 'png'),
            jpeg_quality=output.get('jpeg_quality', 95)
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Récupère une valeur de configuration par clé (support des clés imbriquées).
        
        Args:
            key: Clé de configuration (ex: 'features.ocr.enabled')
            default: Valeur par défaut si la clé n'existe pas
            
        Returns:
            Valeur de configuration ou valeur par défaut
        """
        keys = key.split('.')
        value = self._config_data
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)  # On ne met pas de valeur par défaut ici
                if value is None:  # On vérifie si la clé n'a pas été trouvée
                    return default  # Et on sort immédiatement
            else:
                return default  # La clé parente n'était pas un dictionnaire
        
        return value if value is not None else default
    
    def reload(self) -> None:
        """Recharge la configuration depuis le fichier"""
        self._config_data = self._load_config()
        self.geometry = self._load_geometry_config()
        self.pdf = self._load_pdf_config()
        self.qa = self._load_qa_config()
        self.performance = self._load_performance_config()
        self.output = self._load_output_config()


# Instance globale de configuration (singleton)
_global_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Récupère l'instance globale de configuration
    
    Args:
        config_path: Chemin vers le fichier de configuration (utilisé uniquement au premier appel)
    
    Returns:
        Instance de Config
    """
    global _global_config
    
    if _global_config is None:
        _global_config = Config(config_path)
    
    return _global_config


def reload_config() -> None:
    """Recharge la configuration globale"""
    global _global_config
    
    if _global_config is not None:
        _global_config.reload()


if __name__ == "__main__":
    # Test du chargement de configuration
    config = get_config()
    
    print("=== Configuration chargée ===")
    print(f"\n[Geometry]")
    print(f"  Orientation min confidence: {config.geometry.orientation_min_confidence}")
    print(f"  Deskew enabled: {config.geometry.deskew_enabled}")
    print(f"  Deskew min confidence: {config.geometry.deskew_min_confidence}")
    print(f"  Deskew min angle: {config.geometry.deskew_min_angle}°")
    print(f"  Deskew max angle: {config.geometry.deskew_max_angle}°")
    
    print(f"\n[PDF]")
    print(f"  DPI: {config.pdf.dpi}")
    
    print(f"\n[QA]")
    print(f"  Low confidence orientation: {config.qa.low_confidence_orientation}")
    print(f"  Overcrop risk: {config.qa.overcrop_risk}")
    
    print(f"\n[Performance]")
    print(f"  Batch size: {config.performance.batch_size}")
    print(f"  Max workers: {config.performance.max_workers}")
    
    print(f"\n[Output]")
    print(f"  Image format: {config.output.image_format}")
    print(f"  Save QA flags: {config.output.save_qa_flags}")

