# Configuration du Pipeline

Le pipeline de document analysis utilise un fichier de configuration `config.yaml` situé à la racine du projet pour définir tous les seuils et paramètres.

## Fichier de Configuration

### Emplacement
- **Fichier principal**: `config.yaml` (à la racine du projet)
- **Chargement automatique**: Le fichier est chargé automatiquement au démarrage du pipeline

### Structure

```yaml
geometry:
  orientation:
    min_confidence: 0.70
  crop:
    min_area_ratio: 0.85
    max_margin_ratio: 0.08
  deskew:
    enabled: true
    min_confidence: 0.20
    max_angle: 15.0
    min_angle: 0.5
    hough_threshold: 100
  quality:
    min_contrast: 50
    min_resolution_width: 1200
    min_resolution_height: 1600

pdf:
  dpi: 300
  min_dpi: 300

qa:
  flags:
    low_confidence_orientation: 0.70
    overcrop_risk: 0.08
    low_contrast: 50
    too_small_width: 1200
    too_small_height: 1600
```

## Paramètres Détaillés

### 1. Geometry - Normalisation Géométrique

#### Orientation (ONNX Model)
- `min_confidence` (0.0-1.0): Confiance minimale pour accepter l'orientation détectée
  - **Défaut**: 0.70 (70%)
  - **Impact**: Si la confiance est inférieure, le flag `low_confidence_orientation` sera levé

#### Intelligent Crop (doctr)
- `min_area_ratio` (0.0-1.0): Ratio de surface minimum du document dans l'image
  - **Défaut**: 0.85 (85%)
  - **Impact**: Si le document occupe moins de 85% de l'image, un recadrage sera appliqué
  
- `max_margin_ratio` (0.0-1.0): Marge maximale acceptable après crop
  - **Défaut**: 0.08 (8%)
  - **Impact**: Si le recadrage enlève plus de 8% d'un bord, le flag `overcrop_risk` sera levé

#### Deskew (Correction d'Inclinaison)
- `enabled` (bool): Activer/désactiver la correction d'inclinaison
  - **Défaut**: true
  - **Impact**: Si false, aucun deskew ne sera appliqué

- `min_confidence` (0.0-1.0): **Confiance minimale pour appliquer le deskew**
  - **Défaut**: 0.20 (20%)
  - **Impact**: ⚠️ **Paramètre important** - Si la confiance détectée est inférieure à ce seuil, le deskew ne sera PAS appliqué
  - **Exemple**: Si `min_confidence: 0.20`, un deskew avec 15% de confiance sera ignoré

- `max_angle` (degrés): Angle maximum acceptable pour le deskew
  - **Défaut**: 15.0°
  - **Impact**: Les angles supérieurs à cette valeur seront ignorés

- `min_angle` (degrés): Angle minimum pour justifier un deskew
  - **Défaut**: 0.5°
  - **Impact**: Les petites inclinaisons inférieures à ce seuil ne seront pas corrigées

- `hough_threshold` (int): Seuil pour la détection de lignes Hough
  - **Défaut**: 100
  - **Impact**: Plus le seuil est élevé, moins de lignes seront détectées (détection plus stricte)

#### Quality
- `min_contrast`: Contraste minimum après enhancement
  - **Défaut**: 50
  
- `min_resolution_width`: Largeur minimale de l'image finale (pixels)
  - **Défaut**: 1200
  
- `min_resolution_height`: Hauteur minimale de l'image finale (pixels)
  - **Défaut**: 1600

### 2. PDF Processing

- `dpi`: DPI pour la conversion PDF vers image
  - **Défaut**: 300
  
- `min_dpi`: DPI minimum garanti
  - **Défaut**: 300

### 3. QA Flags

Les seuils pour les flags de qualité (doivent correspondre aux valeurs de `geometry`):

- `low_confidence_orientation`: Seuil pour le flag orientation (0.0-1.0)
- `overcrop_risk`: Seuil pour le flag overcrop (0.0-1.0, en ratio)
- `low_contrast`: Seuil pour le flag contraste faible
- `too_small_width`: Largeur minimale pour le flag résolution
- `too_small_height`: Hauteur minimale pour le flag résolution

## Exemples d'Utilisation

### Exemple 1: Désactiver le Deskew

```yaml
geometry:
  deskew:
    enabled: false
```

### Exemple 2: Deskew Plus Strict (Confiance 40%)

Pour ne deskewer que si la confiance est supérieure à 40%:

```yaml
geometry:
  deskew:
    enabled: true
    min_confidence: 0.40  # Augmenter le seuil à 40%
    max_angle: 10.0       # Réduire l'angle max à 10°
    min_angle: 1.0        # Ignorer les angles < 1°
```

### Exemple 3: Deskew Plus Permissif (Confiance 10%)

Pour accepter des deskew avec une confiance plus faible:

```yaml
geometry:
  deskew:
    enabled: true
    min_confidence: 0.10  # Réduire le seuil à 10%
    max_angle: 20.0       # Accepter des angles plus grands
    hough_threshold: 50   # Détecter plus de lignes
```

### Exemple 4: Crop Plus Agressif

```yaml
geometry:
  crop:
    min_area_ratio: 0.75    # Crop si le document occupe < 75%
    max_margin_ratio: 0.05  # Flag si marge > 5%
```

### Exemple 5: Résolution Plus Élevée

```yaml
geometry:
  quality:
    min_resolution_width: 1920
    min_resolution_height: 2560

pdf:
  dpi: 400  # Augmenter le DPI pour plus de qualité

qa:
  flags:
    too_small_width: 1920
    too_small_height: 2560
```

## Utilisation Programmatique

### Charger la Configuration

```python
from src.utils.config_loader import get_config

# Charger la configuration
config = get_config()

# Accéder aux paramètres
print(f"Deskew min confidence: {config.geometry.deskew_min_confidence}")
print(f"Orientation min confidence: {config.geometry.orientation_min_confidence}")
```

### Recharger la Configuration

```python
from src.utils.config_loader import reload_config

# Modifier config.yaml...

# Recharger
reload_config()
```

### Spécifier un Fichier Différent

```python
from src.utils.config_loader import get_config

# Charger depuis un autre fichier
config = get_config(config_path="custom_config.yaml")
```

## Impact sur le Pipeline

### GeometryNormalizer

Le `GeometryNormalizer` charge automatiquement la configuration:

```python
from src.pipeline import GeometryNormalizer

# Utilise config.yaml automatiquement
normalizer = GeometryNormalizer()

# Ou avec une config personnalisée (ancien format)
normalizer = GeometryNormalizer(config={
    'deskew_min_confidence': 0.30,
    'enable_deskew': True
})
```

### QADetector

Le `QADetector` charge aussi automatiquement la configuration:

```python
from src.utils.qa_flags import QADetector

# Utilise config.yaml automatiquement
detector = QADetector()
```

## Logs et Debug

Pour voir quels paramètres sont chargés, exécutez:

```bash
python -m src.utils.config_loader
```

Cela affichera tous les paramètres chargés depuis `config.yaml`.

## Bonnes Pratiques

1. **Toujours committer `config.yaml`** dans le repository avec des valeurs par défaut raisonnables
2. **Documenter les changements** de configuration dans le code ou les commits
3. **Tester** l'impact des changements de seuils sur un échantillon représentatif
4. **Monitorer les KPIs** dans le QA report après modification des seuils
5. **Créer des profils** de configuration pour différents types de documents:
   - `config_strict.yaml` - Seuils élevés pour haute qualité
   - `config_permissive.yaml` - Seuils bas pour documents difficiles
   - `config_production.yaml` - Seuils optimisés pour la production

## Dépannage

### Erreur: Fichier de configuration introuvable

```
FileNotFoundError: Fichier de configuration introuvable: config.yaml
```

**Solution**: Créez un fichier `config.yaml` à la racine du projet en copiant l'exemple ci-dessus.

### Deskew Jamais Appliqué

Vérifiez:
1. `geometry.deskew.enabled` est à `true`
2. `geometry.deskew.min_confidence` n'est pas trop élevé (ex: 0.20)
3. `geometry.deskew.min_angle` n'est pas trop élevé (ex: 0.5)

### Trop de Flags QA

Si trop de pages sont flaggées, ajustez les seuils:
- Réduire `qa.flags.low_confidence_orientation` (ex: 0.60 au lieu de 0.70)
- Augmenter `qa.flags.overcrop_risk` (ex: 0.10 au lieu de 0.08)
- Réduire `qa.flags.low_contrast` (ex: 40 au lieu de 50)

