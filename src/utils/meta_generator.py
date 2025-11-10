"""
Meta Generator
Génère un fichier meta.json global avec toutes les métadonnées
"""

import json
import os
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from src.utils.qa_flags import load_qa_flags
from src.utils.transform_handler import load_transforms


def generate_meta_json(output_dir: str, meta_file: str = "meta.json"):
    """
    Génère un fichier meta.json global avec toutes les métadonnées
    
    Args:
        output_dir: Répertoire contenant les fichiers traités
        meta_file: Nom du fichier meta.json de sortie
    """
    output_path = Path(output_dir)
    meta_data = {
        'generated_at': datetime.now().isoformat(),
        'total_pages': 0,
        'pages': []
    }
    
    # Chercher tous les fichiers .qa.json
    qa_files = list(output_path.rglob("*.qa.json"))
    meta_data['total_pages'] = len(qa_files)
    
    for qa_file in qa_files:
        # Fichier image correspondant
        image_file = qa_file.with_suffix('').with_suffix('.png')
        if not image_file.exists():
            image_file = qa_file.with_suffix('').with_suffix('.jpg')
        
        if not image_file.exists():
            continue
        
        # Charger les flags QA
        qa_flags = load_qa_flags(str(image_file))
        if qa_flags is None:
            continue
        
        # Charger les transformations
        transform_sequence = load_transforms(str(image_file))
        
        # Créer l'entrée de page
        page_entry = {
            'page_name': image_file.stem,
            'image_path': str(image_file),
            'qa_file': str(qa_file),
            'flags': qa_flags.to_dict(),
            'transforms': transform_sequence.to_dict() if transform_sequence else None
        }
        
        meta_data['pages'].append(page_entry)
    
    # Calculer les statistiques globales
    meta_data['statistics'] = _calculate_global_statistics(meta_data['pages'])
    
    # Sauvegarder
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, indent=2, ensure_ascii=False)
    
    print(f"Fichier meta.json genere: {meta_file}")
    return meta_data


def _calculate_global_statistics(pages: List[Dict]) -> Dict:
    """Calcule les statistiques globales"""
    if not pages:
        return {}
    
    total = len(pages)
    
    stats = {
        'orientation_accuracy': sum(1 for p in pages 
                                   if not p['flags'].get('low_confidence_orientation', False)) / total,
        'overcrop_risk_count': sum(1 for p in pages 
                                  if p['flags'].get('overcrop_risk', False)),
        'overcrop_risk_rate': sum(1 for p in pages 
                                 if p['flags'].get('overcrop_risk', False)) / total,
        'no_quad_detected_count': sum(1 for p in pages 
                                     if p['flags'].get('no_quad_detected', False)),
        'no_quad_detected_rate': sum(1 for p in pages 
                                    if p['flags'].get('no_quad_detected', False)) / total,
        'dewarp_applied_count': sum(1 for p in pages 
                                   if p['flags'].get('dewarp_applied', False)),
        'low_contrast_count': sum(1 for p in pages 
                                 if p['flags'].get('low_contrast_after_enhance', False)),
        'too_small_count': sum(1 for p in pages 
                              if p['flags'].get('too_small_final', False)),
        'avg_processing_time': sum(p['flags'].get('processing_time', 0) for p in pages) / total,
        'pages_with_flags': sum(1 for p in pages 
                               if any([
                                   p['flags'].get('low_confidence_orientation', False),
                                   p['flags'].get('overcrop_risk', False),
                                   p['flags'].get('no_quad_detected', False),
                                   p['flags'].get('dewarp_applied', False),
                                   p['flags'].get('low_contrast_after_enhance', False),
                                   p['flags'].get('too_small_final', False)
                               ]))
    }
    
    return stats

