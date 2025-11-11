"""
Script pour préparer le dataset d'entraînement.

Ce script aide à organiser les données d'entraînement en structurant
les fichiers JSON OCR par type de document dans des sous-dossiers.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import argparse


def organize_training_data(
    input_dir: str,
    output_dir: str,
    metadata_file: Optional[str] = None
) -> None:
    """
    Organise les données d'entraînement par type de document.
    
    Structure attendue en sortie :
    training_data/
    ├── Attestation_CEE/
    │   ├── doc1.json
    │   ├── doc2.json
    │   └── ...
    ├── Facture/
    │   ├── doc1.json
    │   └── ...
    └── ...
    
    Args:
        input_dir: Répertoire contenant les fichiers JSON OCR à organiser.
        output_dir: Répertoire de sortie où créer la structure par type.
        metadata_file: Fichier JSON optionnel avec mapping {filename: document_type}.
                      Si None, le type doit être dans le nom du fichier ou le JSON.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Répertoire d'entrée introuvable: {input_dir}")
    
    # Créer le répertoire de sortie
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Charger le mapping si fourni
    metadata = {}
    if metadata_file:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    
    # Compter les fichiers par type
    type_counts: Dict[str, int] = {}
    
    # Parcourir les fichiers JSON
    json_files = list(input_path.glob('*.json'))
    print(f"Traitement de {len(json_files)} fichiers JSON...")
    
    for json_file in json_files:
        # Déterminer le type de document
        doc_type = None
        
        # 1. Chercher dans le metadata
        if metadata_file and json_file.name in metadata:
            doc_type = metadata[json_file.name]
        
        # 2. Chercher dans le contenu JSON
        if doc_type is None:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    doc_type = data.get('document_type')
            except Exception as e:
                print(f"Erreur lors de la lecture de {json_file.name}: {e}")
                continue
        
        # 3. Chercher dans le nom du fichier (format: type_docname.json)
        if doc_type is None:
            parts = json_file.stem.split('_', 1)
            if len(parts) > 1:
                doc_type = parts[0]
        
        if doc_type is None:
            print(f"⚠️  Type de document non trouvé pour {json_file.name}, ignoré.")
            continue
        
        # Créer le sous-dossier pour ce type
        type_dir = output_path / doc_type
        type_dir.mkdir(parents=True, exist_ok=True)
        
        # Copier le fichier
        dest_file = type_dir / json_file.name
        shutil.copy2(json_file, dest_file)
        
        # Compter
        type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
    
    # Afficher le résumé
    print("\n✅ Organisation terminée !")
    print(f"\nRésumé par type de document:")
    for doc_type, count in sorted(type_counts.items()):
        print(f"  {doc_type}: {count} fichiers")
    print(f"\nTotal: {sum(type_counts.values())} fichiers organisés dans {output_dir}")


def main():
    """Point d'entrée principal du script."""
    parser = argparse.ArgumentParser(
        description="Organise les données d'entraînement par type de document"
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help="Répertoire contenant les fichiers JSON OCR à organiser"
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help="Répertoire de sortie pour la structure organisée"
    )
    parser.add_argument(
        '--metadata',
        type=str,
        default=None,
        help="Fichier JSON optionnel avec mapping {filename: document_type}"
    )
    
    args = parser.parse_args()
    
    try:
        organize_training_data(args.input_dir, args.output_dir, args.metadata)
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

