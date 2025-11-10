"""
Transform handling utilities
Gère la sauvegarde et le chargement des transformations appliquées
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
from src.utils.file_handler import ensure_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Transform:
    """
    Représente une transformation appliquée à une image
    """
    
    def __init__(self, transform_type: str, params: Dict, order: int = 0):
        """
        Initialise une transformation
        
        Args:
            transform_type: Type de transformation ('crop', 'deskew', 'rotation', etc.)
            params: Paramètres de la transformation
            order: Ordre d'application (0 = première transformation)
        """
        self.transform_type = transform_type
        self.params = params
        self.order = order
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit la transformation en dictionnaire"""
        return {
            'transform_type': self.transform_type,
            'params': self.params,
            'order': self.order
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Transform':
        """Crée une transformation depuis un dictionnaire"""
        return cls(
            transform_type=data['transform_type'],
            params=data['params'],
            order=data.get('order', 0)
        )


class TransformSequence:
    """
    Séquence de transformations appliquées à une image
    """
    
    def __init__(self, input_path: str, output_path: str, output_original_path: Optional[str] = None):
        """
        Initialise une séquence de transformations
        
        Args:
            input_path: Chemin du fichier d'entrée original
            output_path: Chemin du fichier de sortie transformé
            output_original_path: Chemin du fichier de sortie original (optionnel)
        """
        self.input_path = input_path
        self.output_path = output_path
        self.output_original_path = output_original_path
        self.transforms: List[Transform] = []
    
    def add_transform(self, transform: Transform) -> None:
        """Ajoute une transformation à la séquence"""
        self.transforms.append(transform)
        # Trier par ordre
        self.transforms.sort(key=lambda t: t.order)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit la séquence en dictionnaire"""
        # Créer un dictionnaire ordonné avec l'ordre souhaité
        result = {
            'input_path': self.input_path,
            'output_path': self.output_path,
        }
        # Ajouter output_original_path si disponible
        if self.output_original_path:
            result['output_original_path'] = self.output_original_path
        # Ajouter transforms à la fin
        result['transforms'] = [t.to_dict() for t in self.transforms]
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransformSequence':
        """Crée une séquence depuis un dictionnaire"""
        # Support de l'ancien format avec output_transformed_path et du nouveau format
        output_path = data.get('output_path', data.get('output_transformed_path', ''))
        output_original = data.get('output_original_path', None)
        seq = cls(
            input_path=data['input_path'],
            output_path=output_path,
            output_original_path=output_original
        )
        for t_data in data.get('transforms', []):
            seq.add_transform(Transform.from_dict(t_data))
        return seq
    
    def save(self, file_path: str) -> None:
        """Sauvegarde la séquence dans un fichier JSON"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, file_path: str) -> 'TransformSequence':
        """Charge une séquence depuis un fichier JSON"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


def get_transform_file_path(output_path: str) -> str:
    """
    Génère le chemin du fichier de transformation à partir du chemin de sortie
    
    Args:
        output_path: Chemin du fichier de sortie
        
    Returns:
        Chemin du fichier de transformation (.json)
    """
    path = Path(output_path)
    return str(path.with_suffix('.transform.json'))


def save_transforms(output_path: str, transform_sequence: TransformSequence) -> None:
    """
    Sauvegarde les transformations dans un fichier JSON
    
    Args:
        output_path: Chemin du fichier de sortie
        transform_sequence: Séquence de transformations à sauvegarder
    """
    try:
        transform_file = get_transform_file_path(output_path)
        # S'assurer que le répertoire existe
        ensure_dir(os.path.dirname(transform_file))
        transform_sequence.save(transform_file)
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde des transformations pour {output_path}", exc_info=True)


def load_transforms(output_path: str) -> Optional[TransformSequence]:
    """
    Charge les transformations depuis un fichier JSON
    
    Args:
        output_path: Chemin du fichier de sortie
        
    Returns:
        Séquence de transformations ou None si le fichier n'existe pas
    """
    transform_file = get_transform_file_path(output_path)
    if os.path.exists(transform_file):
        return TransformSequence.load(transform_file)
    return None

