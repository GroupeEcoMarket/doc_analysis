"""
Script d'entraînement pour le classifieur de documents.

Ce script :
1. Scanne un dossier de données d'entraînement (JSON OCR triés par type)
2. Utilise feature_engineering.py pour vectoriser chaque document
3. Entraîne un modèle scikit-learn ou lightgbm
4. Sauvegarde le modèle et un rapport de performance
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import joblib
from datetime import datetime
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# LightGBM (optionnel)
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Import du module de classification
import sys

# Ajouter le répertoire racine au path pour les imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.classification.feature_engineering import FeatureEngineer
from src.pipeline.models import FeaturesOutput, OCRLine
from src.utils.logger import get_logger
from src.utils.config_loader import get_config

logger = get_logger(__name__)


def process_json_file(file_path_tuple):
    """
    Worker function to load and vectorize a single JSON file.
    Initializes its own FeatureEngineer.
    
    Args:
        file_path_tuple: Tuple of (json_file, doc_type, class_idx, semantic_model_name, min_confidence)
    
    Returns:
        Tuple of (embedding, class_idx) or None if error
    """
    json_file, doc_type, class_idx, semantic_model_name, min_confidence = file_path_tuple
    
    try:
        # Chaque worker initialise son propre FeatureEngineer. C'est plus robuste.
        feature_engineer = FeatureEngineer(
            semantic_model_name=semantic_model_name,
            min_confidence=min_confidence
        )
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        embedding = feature_engineer.extract_document_embedding(data)
        
        return (embedding, class_idx)
    except Exception as e:
        # On log l'erreur mais on ne fait pas planter le pool
        logger.warning(f"Erreur lors du traitement de {json_file}: {e}")
        return None


def load_training_data(
    data_dir: str,
    feature_engineer_params: dict,
    num_workers: int = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Charge et vectorise les données d'entraînement en parallèle.
    
    Structure attendue :
    data_dir/
    ├── Attestation_CEE/
    │   ├── doc1.json
    │   └── ...
    ├── Facture/
    │   └── ...
    └── ...
    
    Args:
        data_dir: Répertoire contenant les sous-dossiers par type de document.
        feature_engineer_params: Dictionnaire avec les paramètres pour FeatureEngineer
            (semantic_model_name, min_confidence).
        num_workers: Nombre de processus workers (défaut: cpu_count() - 1).
    
    Returns:
        Tuple de (X, y, class_names) :
        - X: Array numpy des embeddings (n_samples, n_features)
        - y: Array numpy des labels (n_samples,)
        - class_names: Liste des noms de classes dans l'ordre
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Répertoire de données introuvable: {data_dir}")
    
    # Parcourir les sous-dossiers (chaque sous-dossier = un type de document)
    type_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    
    if not type_dirs:
        raise ValueError(
            f"Aucun sous-dossier trouvé dans {data_dir}. "
            "Structure attendue: data_dir/TypeDocument/*.json"
        )
    
    print(f"📁 Découverte de {len(type_dirs)} types de documents")
    
    # Préparer la liste de toutes les tâches à effectuer
    tasks = []
    class_names = []
    class_to_idx: Dict[str, int] = {}
    
    for type_dir in type_dirs:
        doc_type = type_dir.name
        
        # Ajouter le type à la liste des classes
        if doc_type not in class_to_idx:
            class_to_idx[doc_type] = len(class_names)
            class_names.append(doc_type)
        
        class_idx = class_to_idx[doc_type]
        
        # Parcourir les fichiers JSON dans ce dossier
        json_files = list(type_dir.glob('*.json'))
        print(f"  {doc_type}: {len(json_files)} fichiers")
        
        for json_file in json_files:
            tasks.append((
                json_file,
                doc_type,
                class_idx,
                feature_engineer_params['semantic_model_name'],
                feature_engineer_params['min_confidence']
            ))
    
    if not tasks:
        raise ValueError("Aucun document .json trouvé dans les données d'entraînement")
    
    print(f"\n⚡ Vectorisation de {len(tasks)} documents en parallèle avec {num_workers} workers...")
    
    X_list = []
    y_list = []
    
    # Utiliser le Pool pour exécuter les tâches en parallèle
    with Pool(processes=num_workers) as pool:
        # tqdm ajoute une barre de progression
        # imap_unordered retourne les résultats dès qu'ils sont prêts pour un feedback en temps réel
        results = list(tqdm(pool.imap_unordered(process_json_file, tasks), total=len(tasks), desc="Vectorisation"))
    
    # Collecter les résultats
    for result in results:
        if result is not None:
            embedding, class_idx = result
            X_list.append(embedding)
            y_list.append(class_idx)
    
    if not X_list:
        raise ValueError("La vectorisation n'a produit aucun résultat valide.")
    
    # Convertir en arrays numpy
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"\n✅ Données chargées: {len(X)} documents, {len(class_names)} classes")
    print(f"   Dimensions des embeddings: {X.shape[1]}")
    print(f"   Distribution des classes:")
    for class_name in class_names:
        count = np.sum(y == class_to_idx[class_name])
        print(f"     {class_name}: {count} documents")
    
    return X, y, class_names


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = 'lightgbm',
    **model_params
) -> Any:
    """
    Entraîne un modèle de classification.
    
    Args:
        X_train: Features d'entraînement (n_samples, n_features)
        y_train: Labels d'entraînement (n_samples,)
        model_type: Type de modèle ('lightgbm', 'logistic_regression', 'random_forest')
        **model_params: Paramètres additionnels pour le modèle
    
    Returns:
        Modèle entraîné
    """
    print(f"\n🔧 Entraînement d'un modèle {model_type}...")
    
    if model_type == 'lightgbm':
        if not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "LightGBM n'est pas installé. "
                "Installez-le avec: pip install lightgbm"
            )
        
        # Paramètres par défaut pour LightGBM
        default_params = {
            'objective': 'multiclass',
            'num_class': len(np.unique(y_train)),
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        default_params.update(model_params)
        
        # Créer le dataset LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)
        
        # Entraîner
        model = lgb.train(
            default_params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(period=10)]
        )
    
    elif model_type == 'logistic_regression':
        default_params = {
            'max_iter': 1000,
            'multi_class': 'multinomial',
            'solver': 'lbfgs',
            'random_state': 42
        }
        default_params.update(model_params)
        
        model = LogisticRegression(**default_params)
        model.fit(X_train, y_train)
    
    elif model_type == 'random_forest':
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(model_params)
        
        model = RandomForestClassifier(**default_params)
        model.fit(X_train, y_train)
    
    else:
        raise ValueError(
            f"Type de modèle inconnu: {model_type}. "
            "Choix: 'lightgbm', 'logistic_regression', 'random_forest'"
        )
    
    print("✅ Modèle entraîné avec succès")
    return model


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    model_type: str
) -> Dict[str, Any]:
    """
    Évalue le modèle sur les données de test.
    
    Args:
        model: Modèle entraîné
        X_test: Features de test
        y_test: Labels de test
        class_names: Noms des classes
        model_type: Type de modèle
    
    Returns:
        Dict avec les métriques d'évaluation
    """
    # Prédictions
    # Détecter automatiquement le type de modèle plutôt que de se fier à model_type
    if LIGHTGBM_AVAILABLE and isinstance(model, lgb.Booster):
        # Comportement spécifique à LightGBM
        y_pred_proba = model.predict(X_test)
        # S'assurer que y_pred_proba est bien en 2D, même pour un seul échantillon
        if y_pred_proba.ndim == 1:
            y_pred_proba = y_pred_proba.reshape(1, -1)
        y_pred = np.argmax(y_pred_proba, axis=1)
    elif hasattr(model, 'predict_proba'):
        # Comportement pour les modèles sklearn qui ont predict_proba
        y_pred_proba = model.predict_proba(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
    else:
        # Fallback pour les modèles qui n'ont que .predict()
        y_pred = model.predict(X_test)
    
    # Métriques
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    feature_importance_results = {}
    
    # Détecter automatiquement le type de modèle pour les feature importances
    if LIGHTGBM_AVAILABLE and isinstance(model, lgb.Booster) and hasattr(model, 'feature_importance'):
        feature_importances = model.feature_importance(importance_type='gain')
        
        # La dimension totale est la largeur de X_test. On soustrait les 4 features de position.
        total_features = X_test.shape[1]
        semantic_dim = total_features - 4

        # Gérer la division par zéro
        total_importance = np.sum(feature_importances)
        if total_importance > 0:
            text_importance = np.sum(feature_importances[:semantic_dim])
            pos_importance = np.sum(feature_importances[semantic_dim:])
            
            # Remplir le dictionnaire de résultats
            feature_importance_results = {
                'text_percent': (text_importance / total_importance) * 100,
                'position_percent': (pos_importance / total_importance) * 100,
            }

    # Si c'est un modèle sklearn avec feature importances (ex: RandomForest)
    elif hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
        
        # Calcul dynamique de la dimension sémantique
        total_features = X_test.shape[1]
        semantic_dim = total_features - 4
        
        total_importance = np.sum(feature_importances)
        if total_importance > 0:
            text_importance = np.sum(feature_importances[:semantic_dim])
            pos_importance = np.sum(feature_importances[semantic_dim:])
            
            # Remplir le dictionnaire de résultats
            feature_importance_results = {
                'text_percent': (text_importance / total_importance) * 100,
                'position_percent': (pos_importance / total_importance) * 100,
            }

    # Rapport de classification
    # zero_division=0 pour éviter les warnings et retourner 0.0 pour les métriques indéfinies
    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Vérification manuelle : détecter les classes avec support non nul mais F1-score à 0
    # Cela indique un problème potentiel (ex: classe présente dans le test mais jamais prédite)
    for class_name in class_names:
        if class_name in report:
            class_metrics = report[class_name]
            support = class_metrics.get('support', 0)
            f1_score = class_metrics.get('f1-score', 0.0)
            
            # Si la classe a des échantillons dans le test mais une F1-score de 0
            if support > 0 and f1_score == 0.0:
                logger.warning(
                    f"⚠️  Classe '{class_name}' présente dans le jeu de test ({support} échantillons) "
                    f"mais jamais prédite correctement (F1-score = 0.0). "
                    f"Précision: {class_metrics['precision']:.2f}, Rappel: {class_metrics['recall']:.2f}. "
                    f"Vérifiez la qualité du modèle, l'équilibrage des classes et les données d'entraînement."
                )
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    
    # CORRECTION : Retourner le dictionnaire, qui sera vide si pas d'importance
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,
        'feature_importance': feature_importance_results  
    }


def save_model_and_report(
    model: Any,
    class_names: List[str],
    metrics: Dict[str, Any],
    model_path: str,
    report_path: str
) -> None:
    """
    Sauvegarde le modèle et le rapport de performance.
    
    Args:
        model: Modèle entraîné
        class_names: Noms des classes
        metrics: Métriques d'évaluation
        model_path: Chemin pour sauvegarder le modèle
        report_path: Chemin pour sauvegarder le rapport
    """
    # Sauvegarder le modèle avec les noms de classes
    model_data = {
        'model': model,
        'class_names': class_names
    }
    
    model_file = Path(model_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model_data, model_file)
    print(f"💾 Modèle sauvegardé: {model_path}")
    
    # Sauvegarder le rapport
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'model_path': str(model_path),
        'metrics': {
            'accuracy': metrics['accuracy'],
            'f1_score': metrics['f1_score']
        },
        'classification_report': metrics['classification_report'],
        'confusion_matrix': metrics['confusion_matrix'],
        'class_names': class_names,
        'feature_importances': metrics.get('feature_importance', {})
    }
    
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"📊 Rapport sauvegardé: {report_path}")
    
    # Afficher un résumé
    print("\n" + "="*60)
    print("📈 RÉSUMÉ DES PERFORMANCES")
    print("="*60)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-Score (weighted): {metrics['f1_score']:.4f}")
    print("\nRapport par classe:")
    for class_name in class_names:
        if class_name in metrics['classification_report']:
            cls_metrics = metrics['classification_report'][class_name]
            print(f"  {class_name}:")
            print(f"    Precision: {cls_metrics.get('precision', 0):.4f}")
            print(f"    Recall: {cls_metrics.get('recall', 0):.4f}")
            print(f"    F1-Score: {cls_metrics.get('f1-score', 0):.4f}")

    feature_importances = metrics.get('feature_importance', {})
    if feature_importances:
        print("\n--- Importance des Features ---")
        print(f"  Importance relative du Texte    : {feature_importances.get('text_percent', 0):.2f}%")
        print(f"  Importance relative de la Position : {feature_importances.get('position_percent', 0):.2f}%")

def main():
    """Point d'entrée principal du script d'entraînement."""
    # Charger la config pour obtenir les chemins par défaut
    config = get_config()
    default_data_dir = config.get('paths.training_processed_dir', 'training_data/processed')
    default_model_dir = config.get('paths.training_artifacts_dir', 'training_data/artifacts')
    
    # Récupérer les valeurs par défaut depuis la config pour classification
    classification_config = config.get('classification', {})
    default_embedding_model = classification_config.get('embedding_model', 'antoinelouis/french-me5-base')
    default_min_confidence = classification_config.get('min_confidence', 0.70)
    
    parser = argparse.ArgumentParser(
        description="Entraîne un modèle de classification de documents"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=default_data_dir,
        help=f"Répertoire contenant les données d'entraînement (sous-dossiers par type) (défaut: {default_data_dir})"
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=str(Path(default_model_dir) / 'document_classifier.joblib'),
        help=f"Chemin pour sauvegarder le modèle (défaut: {Path(default_model_dir) / 'document_classifier.joblib'})"
    )
    parser.add_argument(
        '--report-path',
        type=str,
        default=str(Path(default_model_dir) / 'training_report.json'),
        help=f"Chemin pour sauvegarder le rapport (défaut: {Path(default_model_dir) / 'training_report.json'})"
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='lightgbm',
        choices=['lightgbm', 'logistic_regression', 'random_forest'],
        help="Type de modèle à entraîner (défaut: lightgbm)"
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help="Proportion des données pour le test (défaut: 0.2)"
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help="Seed aléatoire pour la reproductibilité (défaut: 42)"
    )
    parser.add_argument(
        '--embedding-model',
        type=str,
        default=default_embedding_model,
        help=f"Modèle sentence-transformers à utiliser (défaut depuis config.yaml: {default_embedding_model})"
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=default_min_confidence,
        help=f"Seuil de confiance minimum pour filtrer les lignes OCR (défaut depuis config.yaml: {default_min_confidence})"
    )
    parser.add_argument(
        '--workers',
        '-w',
        type=int,
        default=None,
        help="Nombre de workers parallèles pour la vectorisation (défaut: cpu_count() - 1)"
    )
    
    args = parser.parse_args()
    
    try:
        # On passe les paramètres à la place de l'objet FeatureEngineer
        feature_engineer_params = {
            'semantic_model_name': args.embedding_model,
            'min_confidence': args.min_confidence
        }
        
        # Charger et vectoriser les données en parallèle
        print("\n📚 Chargement et vectorisation des données d'entraînement...")
        X, y, class_names = load_training_data(args.data_dir, feature_engineer_params, num_workers=args.workers)
        
        # Vérifier si le jeu de test sera assez grand pour la stratification
        n_samples = len(X)
        n_classes = len(class_names)
        test_samples = int(np.ceil(n_samples * args.test_size))
        
        if test_samples < n_classes:
            raise ValueError(
                f"Le jeu de données est trop petit pour créer un jeu de test stratifié. "
                f"Avec {n_samples} documents et un test_size de {args.test_size}, "
                f"le jeu de test n'aurait que {test_samples} échantillon(s), "
                f"ce qui est inférieur au nombre de classes ({n_classes}). "
                f"Augmentez la taille de votre jeu de données ou ajustez test_size."
            )
        
        # Split train/test
        print(f"\n✂️  Division train/test (test_size={args.test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y  # Stratifier pour maintenir la distribution des classes
        )
        print(f"   Train: {len(X_train)} échantillons")
        print(f"   Test: {len(X_test)} échantillons")
        
        # Entraîner le modèle
        model = train_model(
            X_train,
            y_train,
            model_type=args.model_type
        )
        
        # Évaluer le modèle
        print("\n📊 Évaluation du modèle...")
        metrics = evaluate_model(model, X_test, y_test, class_names, args.model_type)
        
        # Sauvegarder le modèle et le rapport
        print("\n💾 Sauvegarde...")
        save_model_and_report(
            model,
            class_names,
            metrics,
            args.model_path,
            args.report_path
        )
        
        print("\n✅ Entraînement terminé avec succès !")
        return 0
    
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement: {e}", exc_info=True)
        print(f"\n❌ Erreur: {e}")
        return 1


if __name__ == '__main__':
    exit(main())

