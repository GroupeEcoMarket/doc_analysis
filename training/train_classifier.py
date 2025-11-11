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

logger = get_logger(__name__)


def load_training_data(
    data_dir: str,
    feature_engineer: FeatureEngineer
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Charge et vectorise les données d'entraînement.
    
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
        feature_engineer: Instance de FeatureEngineer pour vectoriser les documents.
    
    Returns:
        Tuple de (X, y, class_names) :
        - X: Array numpy des embeddings (n_samples, n_features)
        - y: Array numpy des labels (n_samples,)
        - class_names: Liste des noms de classes dans l'ordre
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Répertoire de données introuvable: {data_dir}")
    
    X_list = []
    y_list = []
    class_names = []
    class_to_idx: Dict[str, int] = {}
    
    # Parcourir les sous-dossiers (chaque sous-dossier = un type de document)
    type_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    
    if not type_dirs:
        raise ValueError(
            f"Aucun sous-dossier trouvé dans {data_dir}. "
            "Structure attendue: data_dir/TypeDocument/*.json"
        )
    
    print(f"📁 Découverte de {len(type_dirs)} types de documents")
    
    for type_dir in sorted(type_dirs):
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
            try:
                # Charger le JSON
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Vectoriser le document
                embedding = feature_engineer.extract_document_embedding(data)
                
                X_list.append(embedding)
                y_list.append(class_idx)
            
            except Exception as e:
                logger.warning(f"Erreur lors du traitement de {json_file}: {e}")
                continue
    
    if not X_list:
        raise ValueError("Aucun document valide trouvé dans les données d'entraînement")
    
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
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Évalue le modèle sur les données de test.
    
    Args:
        model: Modèle entraîné
        X_test: Features de test
        y_test: Labels de test
        class_names: Noms des classes
    
    Returns:
        Dict avec les métriques d'évaluation
    """
    # Prédictions
    if hasattr(model, 'predict'):
        y_pred = model.predict(X_test)
    else:
        # LightGBM utilise predict avec reshape
        y_pred = model.predict(X_test).argmax(axis=1)
    
    # Métriques
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Rapport de classification
    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        output_dict=True
    )
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names
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
        'class_names': class_names
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


def main():
    """Point d'entrée principal du script d'entraînement."""
    parser = argparse.ArgumentParser(
        description="Entraîne un modèle de classification de documents"
    )
    parser.add_argument(
        'data_dir',
        type=str,
        help="Répertoire contenant les données d'entraînement (sous-dossiers par type)"
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/document_classifier.joblib',
        help="Chemin pour sauvegarder le modèle (défaut: models/document_classifier.joblib)"
    )
    parser.add_argument(
        '--report-path',
        type=str,
        default='models/training_report.json',
        help="Chemin pour sauvegarder le rapport (défaut: models/training_report.json)"
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
        '--semantic-model',
        type=str,
        default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        help="Modèle sentence-transformers à utiliser (défaut: paraphrase-multilingual-MiniLM-L12-v2)"
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.70,
        help="Seuil de confiance minimum pour filtrer les lignes OCR (défaut: 0.70)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialiser le feature engineer
        print("🔧 Initialisation du FeatureEngineer...")
        feature_engineer = FeatureEngineer(
            semantic_model_name=args.semantic_model,
            min_confidence=args.min_confidence
        )
        
        # Charger et vectoriser les données
        print("\n📚 Chargement des données d'entraînement...")
        X, y, class_names = load_training_data(args.data_dir, feature_engineer)
        
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
        metrics = evaluate_model(model, X_test, y_test, class_names)
        
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

