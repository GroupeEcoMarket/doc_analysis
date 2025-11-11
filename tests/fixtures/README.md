# Test Fixtures

Ce dossier contient les fichiers de test utilisés par les tests d'intégration.

## Fichiers attendus

- `test_image.png` : Une petite image PNG pour tester l'endpoint de géométrie
- `test.pdf` : Un PDF de test (optionnel)
- `attestation_cee.png` : Image de test pour une Attestation CEE (générée automatiquement dans les tests)
- `facture.png` : Image de test pour une Facture (générée automatiquement dans les tests)

## Génération automatique

Si aucun fichier n'est trouvé, les tests créeront automatiquement un fichier PNG minimal pour les tests.

Les images de test pour la classification (attestation_cee.png, facture.png) sont créées automatiquement
par les fixtures pytest dans `test_api_classification.py`.

