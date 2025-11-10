"""
Exemple d'utilisation : Réappliquer les transformations sauvegardées
"""

import cv2
import sys
import os
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.transform_handler import load_transforms
from src.utils.transform_applier import apply_transform_sequence


def main():
    """Exemple de réapplication des transformations"""
    
    # Chemin vers l'image transformée
    transformed_image_path = "data/output/geometry/document_normalized.png"
    transform_file = "data/output/geometry/document_normalized.transform.json"
    
    # Vérifier que les fichiers existent
    if not os.path.exists(transformed_image_path):
        print(f"❌ Image non trouvée: {transformed_image_path}")
        return
    
    if not os.path.exists(transform_file):
        print(f"❌ Fichier de transformation non trouvé: {transform_file}")
        return
    
    # Charger l'image transformée
    print(f"📖 Chargement de l'image: {transformed_image_path}")
    image = cv2.imread(transformed_image_path)
    
    if image is None:
        print("❌ Impossible de charger l'image")
        return
    
    # Charger les transformations
    print(f"📖 Chargement des transformations: {transform_file}")
    transform_sequence = load_transforms(transformed_image_path)
    
    if transform_sequence is None:
        print("❌ Impossible de charger les transformations")
        return
    
    # Afficher les transformations
    print("\n📋 Transformations chargées:")
    for i, transform in enumerate(transform_sequence.transforms, 1):
        print(f"  {i}. {transform.transform_type} (ordre: {transform.order})")
        print(f"     Paramètres: {transform.params}")
    
    # Note: Pour réappliquer les transformations inverses, il faudrait
    # implémenter les transformations inverses (inverse de crop, deskew, rotation)
    print("\n💡 Pour réappliquer les transformations inverses, utilisez:")
    print("   from src.utils.transform_applier import apply_transform_sequence")
    print("   result = apply_transform_sequence(original_image, transform_sequence)")


if __name__ == "__main__":
    main()

