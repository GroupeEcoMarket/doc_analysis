# Guide de Test du Pipeline

## Test avec un PDF

### Prérequis

1. **Installer les dépendances** :
```bash
pip install -r requirements.txt
```

2. **Installer les dépendances système pour pdf2image** (si vous utilisez pdf2image) :
   - **Windows** : Télécharger et installer [poppler](https://github.com/oschwartz10612/poppler-windows/releases/)
   - **Linux** : `sudo apt-get install poppler-utils`
   - **Mac** : `brew install poppler`

### Méthode 1 : Test via CLI

1. **Placer votre PDF dans le dossier d'entrée** :
```bash
# Copier votre PDF
copy votre_document.pdf data\input\
# ou sur Linux/Mac
cp votre_document.pdf data/input/
```

2. **Exécuter le pipeline géométrique** :
```bash
# Windows
python -m src.cli.main geometry --input data/input --output data/output/geometry

# Linux/Mac
python -m src.cli.main geometry --input data/input --output data/output/geometry
```

3. **Vérifier les résultats** :
   - Les images traitées seront dans `data/output/geometry/`
   - Pour un PDF multi-pages, chaque page sera sauvegardée avec le suffixe `_page1.png`, `_page2.png`, etc.

### Méthode 2 : Test via Python

Créez un fichier `test_pdf.py` :

```python
from src.pipeline.geometry import GeometryNormalizer

# Initialiser le normaliseur
normalizer = GeometryNormalizer()

# Traiter un PDF
input_pdf = "data/input/votre_document.pdf"
output_path = "data/output/document_normalise.png"

result = normalizer.process(input_pdf, output_path)

print(f"Résultat: {result}")
print(f"Image sauvegardée: {result['output_path']}")
print(f"Crop appliqué: {result.get('crop_applied', False)}")
print(f"Deskew appliqué: {result.get('deskew_applied', False)}")
print(f"Rotation appliquée: {result.get('rotation_applied', False)}")
```

Exécutez :
```bash
python test_pdf.py
```

### Méthode 3 : Test avec plusieurs pages

Pour traiter toutes les pages d'un PDF :

```python
from src.pipeline.geometry import GeometryNormalizer
from src.utils.pdf_handler import pdf_to_images
import cv2
import os

normalizer = GeometryNormalizer()
input_pdf = "data/input/document_multi_pages.pdf"
output_dir = "data/output/pages"

os.makedirs(output_dir, exist_ok=True)

# Traiter toutes les pages
results = normalizer.process_batch("data/input", output_dir)

for result in results:
    if result.get('page_num'):
        print(f"Page {result['page_num']}/{result['total_pages']}: {result['output_path']}")
```

### Méthode 4 : Test via API

1. **Démarrer le serveur API** :
```bash
python -m src.api.app
```

2. **Envoyer un PDF via curl** :
```bash
curl -X POST "http://localhost:8000/api/v1/pipeline/geometry" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/input/votre_document.pdf"
```

Ou avec Python requests :
```python
import requests

url = "http://localhost:8000/api/v1/pipeline/geometry"
files = {"file": open("data/input/votre_document.pdf", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## Structure des résultats

Le pipeline retourne un dictionnaire avec les informations suivantes :

```python
{
    'input_path': 'chemin/vers/input.pdf',
    'output_path': 'chemin/vers/output.png',
    'crop_applied': True/False,
    'crop_metadata': {
        'area_ratio': 0.85,
        'status': 'cropped'
    },
    'deskew_applied': True/False,
    'deskew_angle': 1.7,
    'deskew_metadata': {
        'angle': 1.7,
        'status': 'success'
    },
    'orientation_detected': True,
    'angle': 0,  # 0, 90, 180, ou 270
    'rotation_applied': True/False,
    'status': 'success',
    'page_num': 1,  # Si PDF multi-pages
    'total_pages': 3  # Si PDF multi-pages
}
```

## Dépannage

### Erreur : "No PDF conversion library available"

**Solution** : Installer pdf2image ou PyMuPDF :
```bash
pip install pdf2image
# ou
pip install PyMuPDF
```

### Erreur : "poppler not found" (avec pdf2image)

**Solution** : Installer poppler (voir Prérequis ci-dessus)

### Le PDF n'est pas traité

**Vérifications** :
1. Le fichier est bien un PDF valide
2. Le chemin est correct
3. Les permissions de lecture sont OK

### Les images de sortie sont vides ou corrompues

**Vérifications** :
1. Vérifier que le répertoire de sortie existe et est accessible en écriture
2. Vérifier les logs d'erreur dans la console
3. Tester avec une image simple d'abord (PNG/JPG)

## Exemples de fichiers de test

Vous pouvez créer des PDFs de test avec :
- Un scanner
- Un outil de conversion (ex: ImageMagick)
- Un générateur de PDF (ex: LaTeX, Word export)

Pour tester différents cas :
- PDF avec document penché
- PDF avec document sur arrière-plan
- PDF multi-pages
- PDF avec différentes orientations (portrait/paysage)

