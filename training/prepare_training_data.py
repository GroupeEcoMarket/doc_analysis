"""
Script pour préparer les données d'entraînement en extrayant les features OCR des images.

Ce script :
1. Scanne les sous-dossiers contenant des images PNG
2. Extrait les features OCR de chaque image avec FeatureExtractor
3. Sauvegarde les résultats dans des fichiers JSON
4. Maintient la structure de dossiers (un dossier = un type de document)
"""

import sys
from pathlib import Path
import argparse
import logging

# Ajouter le répertoire parent au path pour importer les modules src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.features import FeatureExtractor
from src.utils.config_loader import get_config

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_training_data(
    input_dir: str,
    output_dir: str,
    config_path: str = None
) -> None:
    """
    Extrait les features OCR des images et crée les fichiers JSON pour l'entraînement.
    
    Structure attendue en entrée :
    training/
    ├── Type1/
    │   ├── image1.png
    │   ├── image2.png
    │   └── ...
    ├── Type2/
    │   └── image3.png
    └── ...
    
    Structure générée en sortie :
    training_data/
    ├── Type1/
    │   ├── image1.json
    │   ├── image2.json
    │   └── ...
    ├── Type2/
    │   └── image3.json
    └── ...
    
    Args:
        input_dir: Répertoire contenant les sous-dossiers d'images par type.
        output_dir: Répertoire de sortie pour les fichiers JSON.
        config_path: Chemin optionnel vers le fichier de configuration.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Répertoire d'entrée introuvable: {input_dir}")
    
    # Créer le répertoire de sortie
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialiser l'extracteur de features
    logger.info("Initialisation de FeatureExtractor...")
    config = get_config(config_path)
    extractor = FeatureExtractor(app_config=config)
    
    # Parcourir les sous-dossiers (chaque sous-dossier = un type de document)
    type_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    if not type_dirs:
        raise ValueError(
            f"Aucun sous-dossier trouvé dans {input_dir}. "
            "Structure attendue: input_dir/TypeDocument/*.png"
        )
    
    print(f"\n[INFO] Decouverte de {len(type_dirs)} types de documents")
    
    total_processed = 0
    total_errors = 0
    type_counts = {}
    
    for type_dir in sorted(type_dirs):
        doc_type = type_dir.name
        print(f"\n[INFO] Traitement du type: {doc_type}")
        
        # Créer le sous-dossier de sortie pour ce type
        output_type_dir = output_path / doc_type
        output_type_dir.mkdir(parents=True, exist_ok=True)
        
        # Trouver toutes les images PNG dans ce dossier
        image_files = list(type_dir.glob('*.png'))
        print(f"   Trouvé {len(image_files)} images")
        
        success_count = 0
        error_count = 0
        
        for i, image_file in enumerate(image_files, 1):
            try:
                # Créer le chemin de sortie JSON
                output_json = output_type_dir / f"{image_file.stem}.json"
                
                # Vérifier si le fichier a déjà été traité
                if output_json.exists():
                    print(f"   [{i}/{len(image_files)}] SKIP {image_file.name} (déjà traité)")
                    success_count += 1  # Compter comme succès car déjà présent
                    continue
                
                # Extraire les features
                result = extractor.process(str(image_file), str(output_json))
                
                if result.status == 'success':
                    success_count += 1
                    print(f"   [{i}/{len(image_files)}] OK {image_file.name}")
                else:
                    error_count += 1
                    logger.warning(f"   [{i}/{len(image_files)}] WARN {image_file.name}: {result.error}")
            
            except Exception as e:
                error_count += 1
                logger.error(f"   [{i}/{len(image_files)}] ERROR {image_file.name}: {e}")
        
        print(f"   Succès: {success_count}, Erreurs: {error_count}")
        type_counts[doc_type] = success_count
        total_processed += success_count
        total_errors += error_count
    
    # Afficher le résumé
    print(f"\n{'='*60}")
    print("[SUCCESS] Preparation terminee !")
    print(f"{'='*60}")
    print(f"\nResume par type de document:")
    for doc_type, count in sorted(type_counts.items()):
        print(f"  {doc_type}: {count} fichiers JSON crees")
    print(f"\nTotal: {total_processed} fichiers traites avec succes")
    if total_errors > 0:
        print(f"[WARNING] {total_errors} erreurs rencontrees")
    print(f"\nLes donnees sont pretes dans: {output_dir}")
    print(f"\nVous pouvez maintenant entrainer le modele avec:")
    print(f"  poetry run python training/train_classifier.py {output_dir}")


def main():
    """Point d'entrée principal du script."""
    parser = argparse.ArgumentParser(
        description="Prépare les données d'entraînement en extrayant les features OCR des images"
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help="Répertoire contenant les sous-dossiers d'images par type"
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help="Répertoire de sortie pour les fichiers JSON"
    )
    parser.add_argument(
        '--config',
        '-c',
        type=str,
        default=None,
        help="Chemin optionnel vers le fichier de configuration"
    )
    
    args = parser.parse_args()
    
    try:
        prepare_training_data(args.input_dir, args.output_dir, args.config)
    except Exception as e:
        logger.error(f"[ERROR] Erreur: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

