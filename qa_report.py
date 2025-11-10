"""
Script de génération de rapport QA
Génère un rapport HTML avec galerie avant/après et statistiques
"""

import sys
import argparse
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.qa_report import QAReportGenerator
from src.utils.meta_generator import generate_meta_json


def main():
    parser = argparse.ArgumentParser(description='Génère un rapport QA HTML et meta.json')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/output',
        help='Répertoire contenant les fichiers traités (défaut: data/output)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='qa_report.html',
        help='Fichier HTML de sortie (défaut: qa_report.html)'
    )
    parser.add_argument(
        '--meta',
        type=str,
        default='meta.json',
        help='Fichier meta.json de sortie (défaut: meta.json)'
    )
    
    args = parser.parse_args()
    
    print(f"Generation du rapport QA...")
    print(f"   Repertoire: {args.output_dir}")
    print(f"   Rapport HTML: {args.output}")
    print(f"   Meta JSON: {args.meta}")
    
    # Générer meta.json
    print("\nGeneration de meta.json...")
    generate_meta_json(args.output_dir, args.meta)
    
    # Créer le générateur de rapport HTML
    generator = QAReportGenerator(args.output_dir)
    
    # Collecter les données
    print("\nCollecte des donnees pour le rapport HTML...")
    generator.collect_data()
    
    print(f"   {len(generator.pages_data)} pages trouvees")
    
    # Générer le rapport HTML
    print("\nGeneration du rapport HTML...")
    generator.generate_html_report(args.output)
    
    print(f"\nRapport genere avec succes:")
    print(f"   - Rapport HTML: {args.output}")
    print(f"   - Meta JSON: {args.meta}")


if __name__ == "__main__":
    main()

