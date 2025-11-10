# Guide du Système QA

## Vue d'ensemble

Le système QA (Quality Assurance) génère automatiquement des flags de qualité et des rapports pour chaque document traité.

## Flags de Qualité

### Flags disponibles

1. **low_confidence_orientation** (conf < 0.70)
   - Détecte quand la confiance de détection d'orientation est faible
   - Seuil configurable via `orientation_confidence_threshold`

2. **overcrop_risk** (recadrage > 8% d'un bord)
   - Détecte un risque de sur-crop (document trop proche des bords)
   - Seuil configurable via `overcrop_threshold` (défaut: 8.0%)

3. **no_quad_detected** (pas de quadrilatère fiable)
   - Détecte quand aucun quadrilatère n'a été détecté par doctr
   - Status: `no_detection`, `invalid_geometry`, `no_valid_detection`

4. **dewarp_applied** (correction non plane)
   - Détecte quand une transformation de perspective (dewarp) a été appliquée
   - Indique que le document n'était pas plan

5. **low_contrast_after_enhance** (contraste toujours faible)
   - Détecte un contraste insuffisant après traitement
   - Seuil configurable via `contrast_threshold` (défaut: 30.0)

6. **too_small_final** (résolution cible non atteinte)
   - Détecte quand la résolution finale est trop petite
   - Seuil configurable via `min_resolution` (défaut: [800, 1000])

## Fichiers générés

Pour chaque image traitée (`output.png`), 3 fichiers sont créés :

1. **`output.png`** : Image finale traitée
2. **`output.transform.json`** : Séquence de transformations appliquées
3. **`output.qa.json`** : Flags QA et métriques de qualité

## KPIs

### KPIs calculés

- **Orientation Accuracy** : Pourcentage de pages avec confiance d'orientation ≥ 0.70
- **Pages avec Overcrop Risk** : Nombre et pourcentage de pages à risque
- **Taux No Quad Detected** : Pourcentage de pages sans détection de quadrilatère
- **Temps Moyen/Page** : Temps de traitement moyen en secondes
- **Pages avec Flags** : Nombre total de pages avec au moins un flag actif

## Génération du rapport

### Commande

```bash
python qa_report.py --output-dir data/output --output qa_report.html --meta meta.json
```

### Fichiers générés

1. **`qa_report.html`** : Rapport HTML interactif avec :
   - Statistiques agrégées (KPIs)
   - Tableau des flags par page
   - Galerie avant/après avec 3 vignettes par page :
     - Source (image originale)
     - Masque/Contour (détection du document)
     - Final (image traitée)
   - Badges de flags pour chaque page

2. **`meta.json`** : Fichier JSON global avec :
   - Métadonnées de toutes les pages
   - Flags QA pour chaque page
   - Transformations appliquées
   - Statistiques globales

## Structure de meta.json

```json
{
  "generated_at": "2024-01-01T12:00:00",
  "total_pages": 10,
  "statistics": {
    "orientation_accuracy": 0.95,
    "overcrop_risk_count": 2,
    "overcrop_risk_rate": 0.20,
    "no_quad_detected_count": 1,
    "no_quad_detected_rate": 0.10,
    "avg_processing_time": 2.5,
    "pages_with_flags": 3
  },
  "pages": [
    {
      "page_name": "document_page1",
      "image_path": "data/output/document_page1.png",
      "qa_file": "data/output/document_page1.qa.json",
      "flags": {
        "low_confidence_orientation": false,
        "overcrop_risk": false,
        "no_quad_detected": false,
        "dewarp_applied": true,
        "low_contrast_after_enhance": false,
        "too_small_final": false,
        "orientation_confidence": 0.95,
        "crop_margins": {"top": 5.2, "bottom": 4.8, "left": 6.1, "right": 5.9},
        "final_resolution": [1200, 1600],
        "processing_time": 2.3
      },
      "transforms": { ... }
    }
  ]
}
```

## Configuration

Les seuils peuvent être configurés lors de l'initialisation du `GeometryNormalizer` :

```python
config = {
    'orientation_confidence_threshold': 0.70,
    'overcrop_threshold': 8.0,
    'min_resolution': [800, 1000],
    'contrast_threshold': 30.0
}

normalizer = GeometryNormalizer(config=config)
```

## Utilisation

### 1. Traiter des documents

```bash
python -m src.cli.main geometry --input data/input --output data/output/geometry
```

Les fichiers `.qa.json` sont générés automatiquement.

### 2. Générer le rapport

```bash
python qa_report.py --output-dir data/output/geometry
```

### 3. Consulter les résultats

- Ouvrir `qa_report.html` dans un navigateur
- Consulter `meta.json` pour les métadonnées complètes
- Examiner les fichiers `.qa.json` individuels pour chaque page

## Exemple d'utilisation programmatique

```python
from src.utils.qa_flags import load_qa_flags

# Charger les flags QA d'une page
qa_flags = load_qa_flags("data/output/document.png")

if qa_flags:
    print(f"Overcrop risk: {qa_flags.overcrop_risk}")
    print(f"Processing time: {qa_flags.processing_time:.2f}s")
    print(f"Final resolution: {qa_flags.final_resolution}")
```

## Monitoring

Le temps de traitement est loggé pour chaque page et peut être utilisé pour :
- Identifier les pages lentes
- Optimiser le pipeline
- Définir des objectifs de performance (non bloquants)

