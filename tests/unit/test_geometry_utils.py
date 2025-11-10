"""
Tests unitaires pour les utilitaires géométriques
Tests rapides sans dépendances externes (pas de lecture de fichiers)
"""

import pytest
import numpy as np
from src.pipeline.geometry import fit_quad_to_pts


def test_fit_quad_to_pts_orders_points_correctly():
    """
    Test que fit_quad_to_pts ordonne correctement les points d'un quadrilatère.
    
    Ce test est un vrai test unitaire :
    - Pas de lecture de fichier
    - Pas de dépendances externes
    - Exécution en quelques millisecondes
    - Valide un pur algorithme
    """
    # Créer 4 points désordonnés représentant un rectangle
    # Format: (x, y)
    # Points dans un ordre aléatoire : bottom-right, top-left, bottom-left, top-right
    points_disordered = np.array([
        [100, 100],  # bottom-right
        [0, 0],      # top-left
        [0, 100],    # bottom-left
        [100, 0]     # top-right
    ], dtype=np.float32)
    
    # Appeler la fonction
    points_ordered = fit_quad_to_pts(points_disordered)
    
    # Vérifier que le résultat est un numpy array
    assert isinstance(points_ordered, np.ndarray)
    assert points_ordered.shape == (4, 2)
    assert points_ordered.dtype == np.float32
    
    # Vérifier l'ordre attendu : top-left, top-right, bottom-right, bottom-left
    expected_order = np.array([
        [0, 0],      # top-left
        [100, 0],    # top-right
        [100, 100],  # bottom-right
        [0, 100]     # bottom-left
    ], dtype=np.float32)
    
    # Vérifier que les points sont dans le bon ordre
    np.testing.assert_array_almost_equal(points_ordered, expected_order)


def test_fit_quad_to_pts_with_different_rectangle():
    """
    Test avec un rectangle différent pour vérifier la robustesse
    """
    # Rectangle plus grand et décalé
    points_disordered = np.array([
        [200, 300],  # bottom-right
        [50, 50],    # top-left
        [50, 300],   # bottom-left
        [200, 50]    # top-right
    ], dtype=np.float32)
    
    points_ordered = fit_quad_to_pts(points_disordered)
    
    # Vérifier la structure
    assert points_ordered.shape == (4, 2)
    
    # Vérifier l'ordre : top-left doit avoir la somme minimale
    sums = np.sum(points_ordered, axis=1)
    assert sums[0] == np.min(sums)  # top-left
    
    # bottom-right doit avoir la somme maximale
    assert sums[2] == np.max(sums)  # bottom-right
    
    # top-right doit avoir x > top-left et y similaire
    assert points_ordered[1][0] > points_ordered[0][0]  # top-right.x > top-left.x
    assert abs(points_ordered[1][1] - points_ordered[0][1]) < 1  # y similaire
    
    # bottom-left doit avoir x < bottom-right et y > top-left
    assert points_ordered[3][0] < points_ordered[2][0]  # bottom-left.x < bottom-right.x
    assert points_ordered[3][1] > points_ordered[0][1]  # bottom-left.y > top-left.y


def test_fit_quad_to_pts_raises_error_with_wrong_number_of_points():
    """
    Test que la fonction lève une erreur si le nombre de points est incorrect
    """
    # Test avec 3 points
    points_3 = np.array([[0, 0], [100, 0], [100, 100]], dtype=np.float32)
    with pytest.raises(ValueError, match="exactement 4 points"):
        fit_quad_to_pts(points_3)
    
    # Test avec 5 points
    points_5 = np.array([
        [0, 0], [100, 0], [100, 100], [0, 100], [50, 50]
    ], dtype=np.float32)
    with pytest.raises(ValueError, match="exactement 4 points"):
        fit_quad_to_pts(points_5)


def test_fit_quad_to_pts_preserves_points():
    """
    Test que tous les points d'entrée sont présents dans la sortie
    """
    points_input = np.array([
        [100, 100],
        [0, 0],
        [0, 100],
        [100, 0]
    ], dtype=np.float32)
    
    points_output = fit_quad_to_pts(points_input)
    
    # Vérifier que tous les points d'entrée sont dans la sortie
    # (peuvent être dans un ordre différent, mais doivent être présents)
    for point_in in points_input:
        found = False
        for point_out in points_output:
            if np.allclose(point_in, point_out):
                found = True
                break
        assert found, f"Point {point_in} non trouvé dans la sortie"

