# Guide d'utilisation des Transformations

## Vue d'ensemble

Le pipeline sauvegarde automatiquement toutes les transformations appliquées dans un fichier `.transform.json` à côté de chaque image de sortie. Cela permet de :
- Retracer toutes les transformations appliquées
- Réappliquer les transformations à d'autres images
- Créer des fonctions de transformation T pour des usages futurs

## Format des fichiers de transformation

Chaque fichier `.transform.json` contient :

```json
{
  "input_path": "chemin/vers/input.pdf",
  "output_path": "chemin/vers/output.png",
  "transforms": [
    {
      "transform_type": "crop",
      "order": 1,
      "params": {
        "area_ratio": 0.85,
        "transform_matrix": [[...], [...], [...]],
        "source_points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
        "destination_points": [[...], [...], [...], [...]],
        "output_size": [width, height]
      }
    },
    {
      "transform_type": "deskew",
      "order": 2,
      "params": {
        "angle": 1.7,
        "transform_matrix": [[...], [...]],
        "center": [cx, cy],
        "output_size": [width, height],
        "original_size": [width, height]
      }
    },
    {
      "transform_type": "rotation",
      "order": 3,
      "params": {
        "angle": 90,
        "rotation_type": "standard"
      }
    }
  ]
}
```

## Utilisation

### 1. Charger les transformations

```python
from src.utils.transform_handler import load_transforms

# Charger depuis un fichier de sortie
transform_sequence = load_transforms("data/output/geometry/document.png")

if transform_sequence:
    print(f"Input: {transform_sequence.input_path}")
    print(f"Output: {transform_sequence.output_path}")
    for transform in transform_sequence.transforms:
        print(f"  - {transform.transform_type}: {transform.params}")
```

### 2. Réappliquer les transformations

```python
import cv2
from src.utils.transform_applier import apply_transform_sequence

# Charger une nouvelle image
new_image = cv2.imread("nouvelle_image.png")

# Charger les transformations
transform_sequence = load_transforms("data/output/geometry/document.png")

# Appliquer les mêmes transformations
result = apply_transform_sequence(new_image, transform_sequence)

# Sauvegarder
cv2.imwrite("nouvelle_image_transformed.png", result)
```

### 3. Créer une fonction T de transformation

```python
from src.utils.transform_handler import load_transforms
from src.utils.transform_applier import apply_transform_sequence
import cv2
import numpy as np

def create_transform_function(transform_file_path: str):
    """
    Crée une fonction de transformation T à partir d'un fichier de transformation
    
    Args:
        transform_file_path: Chemin vers le fichier .transform.json
        
    Returns:
        Fonction T(image) -> transformed_image
    """
    transform_sequence = load_transforms(transform_file_path)
    
    if transform_sequence is None:
        raise ValueError(f"Fichier de transformation non trouvé: {transform_file_path}")
    
    def T(image: np.ndarray) -> np.ndarray:
        """Applique les transformations à une image"""
        return apply_transform_sequence(image, transform_sequence)
    
    return T

# Utilisation
T = create_transform_function("data/output/geometry/document.transform.json")
image = cv2.imread("input.png")
transformed = T(image)
```

### 4. Appliquer une transformation spécifique

```python
from src.utils.transform_applier import apply_single_transform
from src.utils.transform_handler import Transform

# Créer une transformation personnalisée
custom_transform = Transform(
    transform_type='rotation',
    params={'angle': 90, 'rotation_type': 'standard'},
    order=0
)

# Appliquer
image = cv2.imread("input.png")
result = apply_single_transform(image, custom_transform)
```

## Structure des transformations

### Crop (Perspective Transform)
- **Type**: `crop`
- **Matrice**: Matrice 3x3 de transformation de perspective
- **Points source**: 4 points du quadrilatère original
- **Points destination**: 4 points du rectangle de sortie
- **Taille sortie**: [width, height]

### Deskew (Affine Transform)
- **Type**: `deskew`
- **Matrice**: Matrice 2x3 de transformation affine
- **Centre**: Point central de rotation
- **Angle**: Angle de correction en degrés
- **Tailles**: Originale et de sortie

### Rotation
- **Type**: `rotation`
- **Angle**: Angle de rotation (0, 90, 180, 270 ou arbitraire)
- **Type rotation**: `standard` ou `arbitrary`

## Exemple complet

```python
import cv2
from src.utils.transform_handler import load_transforms
from src.utils.transform_applier import apply_transform_sequence

# 1. Traiter un document (génère automatiquement le fichier .transform.json)
from src.pipeline.geometry import GeometryNormalizer

normalizer = GeometryNormalizer()
result = normalizer.process("input.pdf", "output.png")

# 2. Charger les transformations sauvegardées
transform_sequence = load_transforms("output.png")

# 3. Appliquer les mêmes transformations à une autre image
other_image = cv2.imread("autre_document.png")
transformed = apply_transform_sequence(other_image, transform_sequence)
cv2.imwrite("autre_document_transformed.png", transformed)
```

## Notes

- Les fichiers `.transform.json` sont créés automatiquement lors du traitement
- Les transformations sont sauvegardées dans l'ordre d'application
- Toutes les matrices et paramètres nécessaires sont inclus pour réapplication
- Les transformations peuvent être combinées pour créer des pipelines personnalisés

