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
import gc
import tempfile
import shutil
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# Ajouter le répertoire parent au path pour importer les modules src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.features import FeatureExtractor
from src.utils.config_loader import get_config

# Configuration du logging
logging.basicConfig(
    level=logging.WARNING,  # Seulement les warnings et erreurs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Mais garder INFO pour ce script


def process_single_image(args):
    """
    Traite une seule image (fonction worker pour multiprocessing).
    
    Args:
        args: Tuple de (image_file, output_json, config_path, doc_type, index, total)
        
    Returns:
        Tuple de (success: bool, image_name: str, error_msg: str or None)
    """
    image_file, output_json, config_path, doc_type, index, total = args
    
    try:
        # Vérifier si le fichier a déjà été traité
        if output_json.exists():
            return (True, image_file.name, "skip")
        
        # Créer l'extracteur dans chaque worker (chaque processus a sa propre instance)
        config = get_config(config_path)
        extractor = FeatureExtractor(app_config=config)
        
        # Extraire les features
        result = extractor.process(str(image_file), str(output_json))
        
        if result.status == 'success':
            return (True, image_file.name, None)
        else:
            return (False, image_file.name, result.error)
    
    except Exception as e:
        return (False, image_file.name, f"{type(e).__name__}: {e}")


def prepare_training_data(
    input_dir: str,
    output_dir: str,
    config_path: str = None,
    num_workers: int = None
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
    
    # Déterminer le nombre de workers (laisser 1 CPU libre pour le système)
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    logger.info(f"Utilisation de {num_workers} workers pour le traitement parallèle (sur {cpu_count()} CPUs disponibles)")
    
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
        
        if not image_files:
            continue
        
        # Préparer les arguments pour chaque image
        tasks = [
            (image_file, output_type_dir / f"{image_file.stem}.json", config_path, doc_type, i, len(image_files))
            for i, image_file in enumerate(image_files, 1)
        ]
        
        # Traiter les images en parallèle
        success_count = 0
        error_count = 0
        
        # --- AMÉLIORATION : Utiliser imap_unordered avec tqdm ---
        print(f"   Lancement du traitement parallèle...")
        with Pool(processes=num_workers) as pool:
            # tqdm va automatiquement créer et mettre à jour une barre de progression
            # imap_unordered retourne les résultats dès qu'ils sont prêts
            results = list(tqdm(pool.imap_unordered(process_single_image, tasks), total=len(tasks), desc=f"   {doc_type}"))
        
        # Le code ici ne s'exécute qu'une fois TOUTES les tâches terminées,
        # mais tqdm aura déjà affiché la progression.
        
        # Analyser les résultats
        for success, image_name, error_msg in results:
            if success:
                success_count += 1
            else:
                error_count += 1
                # On peut logger l'erreur ici si on veut un résumé à la fin
                if error_msg != "skip":
                    logger.warning(f"Erreur sur {image_name}: {error_msg}")
        
        # Nettoyage mémoire après chaque type
        gc.collect()
        
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
    parser.add_argument(
        '--workers',
        '-w',
        type=int,
        default=None,
        help="Nombre de workers parallèles (défaut: cpu_count() - 1)"
    )
    
    args = parser.parse_args()
    
    try:
        prepare_training_data(args.input_dir, args.output_dir, args.config, args.workers)
    except Exception as e:
        logger.error(f"[ERROR] Erreur: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

