"""
File handling utilities
"""

import os
from pathlib import Path
from typing import List, Optional


def ensure_dir(path: str) -> None:
    """
    Crée un répertoire s'il n'existe pas
    
    Args:
        path: Chemin du répertoire
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def get_files(input_dir: str, extensions: Optional[List[str]] = None) -> List[str]:
    """
    Récupère la liste des fichiers dans un répertoire
    
    Args:
        input_dir: Répertoire à scanner
        extensions: Liste des extensions à filtrer (ex: ['.pdf', '.png', '.jpg'])
        
    Returns:
        list: Liste des chemins de fichiers
    """
    if extensions is None:
        extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp']
    
    files = []
    for root, dirs, filenames in os.walk(input_dir):
        for filename in filenames:
            if any(filename.lower().endswith(ext.lower()) for ext in extensions):
                files.append(os.path.join(root, filename))
    
    return sorted(files)


def get_output_path(input_path: str, output_dir: str, suffix: str = "") -> str:
    """
    Génère le chemin de sortie basé sur le chemin d'entrée
    
    Args:
        input_path: Chemin du fichier d'entrée
        output_dir: Répertoire de sortie
        suffix: Suffixe optionnel à ajouter au nom de fichier
        
    Returns:
        str: Chemin de sortie
    """
    filename = Path(input_path).stem
    extension = Path(input_path).suffix
    
    if suffix:
        output_filename = f"{filename}_{suffix}{extension}"
    else:
        output_filename = f"{filename}{extension}"
    
    return os.path.join(output_dir, output_filename)

