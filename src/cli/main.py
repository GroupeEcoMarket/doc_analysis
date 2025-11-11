"""
Command-line interface for document analysis pipeline
"""

from src.utils.bootstrap import configure_paddle_environment
configure_paddle_environment()  # Doit être appelé avant toutes les autres importations

import click
import os
from pathlib import Path
from typing import Optional, Tuple
from src.cli.dependencies import (
    get_preprocessing_normalizer,
    get_geometry_normalizer,
    get_colometry_normalizer,
    get_feature_extractor,
    get_document_classifier,
    get_app_config
)
from src.utils.config_loader import Config


@click.group()
def cli() -> None:
    """Document Analysis Pipeline CLI"""
    pass


@cli.command()
@click.option("--input", "-i", required=True, help="Répertoire d'entrée")
@click.option("--output", "-o", required=True, help="Répertoire de sortie")
@click.option("--config", "-c", help="Fichier de configuration")
def colometry(input: str, output: str, config: Optional[str]) -> None:
    """Normalisation colométrie"""
    click.echo(f"Normalisation colométrie: {input} -> {output}")
    
    # Utiliser l'injection de dépendances
    normalizer = get_colometry_normalizer(config_path=config)
    results = normalizer.process_batch(input, output)
    
    click.echo(f"Traitement terminé: {len(results)} documents traités")


@cli.command()
@click.option("--input", "-i", required=True, help="Répertoire d'entrée")
@click.option("--output", "-o", required=True, help="Répertoire de sortie")
@click.option("--config", "-c", help="Fichier de configuration")
def geometry(input: str, output: str, config: Optional[str]) -> None:
    """Normalisation géométrie"""
    click.echo(f"Normalisation géométrie: {input} -> {output}")
    
    # Utiliser l'injection de dépendances
    normalizer = get_geometry_normalizer(config_path=config)
    results = normalizer.process_batch(input, output)
    
    click.echo(f"Traitement terminé: {len(results)} documents traités")


@cli.command()
@click.option("--input", "-i", required=True, help="Répertoire d'entrée")
@click.option("--output", "-o", required=True, help="Répertoire de sortie")
@click.option("--config", "-c", help="Fichier de configuration")
def preprocessing(input: str, output: str, config: Optional[str]) -> None:
    """Étape de prétraitement : amélioration contraste et classification."""
    click.echo(f"Prétraitement des images : {input} -> {output}")
    
    # Utiliser l'injection de dépendances
    normalizer = get_preprocessing_normalizer(config_path=config)
    results = normalizer.process_batch(input, output)
    
    click.echo(f"Traitement terminé: {len(results)} documents prétraités")


@cli.command()
@click.option("--input", "-i", required=True, help="Répertoire d'entrée")
@click.option("--output", "-o", required=True, help="Répertoire de sortie")
@click.option("--config", "-c", help="Fichier de configuration")
def features(input: str, output: str, config: Optional[str]) -> None:
    """Extraction de features"""
    click.echo(f"Extraction de features: {input} -> {output}")
    
    # Utiliser l'injection de dépendances
    extractor = get_feature_extractor(config_path=config)
    results = extractor.process_batch(input, output)
    
    click.echo(f"Traitement terminé: {len(results)} documents traités")


@cli.command()
@click.option("--input", "-i", required=True, help="Répertoire d'entrée (JSON OCR)")
@click.option("--output", "-o", required=True, help="Répertoire de sortie")
@click.option("--config", "-c", help="Fichier de configuration")
def classification(input: str, output: str, config: Optional[str]) -> None:
    """Classification de type de document"""
    import json
    from pathlib import Path
    from src.utils.file_handler import ensure_dir, get_files
    
    click.echo(f"Classification de documents: {input} -> {output}")
    
    try:
        # Utiliser l'injection de dépendances
        classifier = get_document_classifier(config_path=config)
        
        # Créer le répertoire de sortie
        ensure_dir(output)
        
        # Parcourir les fichiers JSON d'entrée
        json_files = get_files(input, extensions=['.json'])
        
        if not json_files:
            click.echo(f"Aucun fichier JSON trouvé dans {input}")
            return
        
        click.echo(f"Traitement de {len(json_files)} fichiers...")
        
        results = []
        for json_file in json_files:
            try:
                # Charger le JSON OCR
                with open(json_file, 'r', encoding='utf-8') as f:
                    ocr_data = json.load(f)
                
                # Classifier
                classification_result = classifier.predict(ocr_data)
                
                # Sauvegarder le résultat
                output_file = Path(output) / Path(json_file).name
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(classification_result, f, indent=2, ensure_ascii=False)
                
                results.append({
                    'file': str(json_file),
                    'document_type': classification_result.get('document_type'),
                    'confidence': classification_result.get('confidence', 0.0)
                })
                
            except Exception as e:
                click.echo(f"Erreur lors du traitement de {json_file}: {e}", err=True)
                continue
        
        click.echo(f"\nTraitement terminé: {len(results)} documents classifiés")
        
        # Afficher un résumé
        if results:
            click.echo("\nRésumé:")
            type_counts = {}
            for result in results:
                doc_type = result['document_type'] or 'Unknown'
                type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            
            for doc_type, count in sorted(type_counts.items()):
                click.echo(f"  {doc_type}: {count}")
    
    except ValueError as e:
        click.echo(f"Erreur de configuration: {e}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"Erreur lors de la classification: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option("--input", "-i", required=True, help="Répertoire d'entrée")
@click.option("--output", "-o", required=True, help="Répertoire de sortie")
@click.option("--config", "-c", help="Fichier de configuration")
@click.option("--stages", "-s", multiple=True, 
              type=click.Choice(["preprocessing", "colometry", "geometry", "features", "classification"]),
              help="Étapes à exécuter (peut être répété plusieurs fois)")
def pipeline(input: str, output: str, config: Optional[str], stages: Tuple[str, ...]) -> None:
    """Exécute le pipeline complet ou des étapes spécifiques"""
    # Charger la configuration une seule fois
    app_config = get_app_config(config_path=config)
    
    # Si aucune étape spécifiée, exécuter toutes les étapes par défaut
    if not stages:
        stages = ["preprocessing", "geometry", "features", "classification"]
    
    # Ordre logique des étapes (pour référence)
    # preprocessing → geometry → features
    # colometry peut être exécuté indépendamment ou avant geometry
    
    current_input = input
    
    if "preprocessing" in stages:
        click.echo("Étape 1: Prétraitement des images...")
        # Utiliser l'injection de dépendances
        preproc_normalizer = get_preprocessing_normalizer(config_path=config)
        preproc_output = os.path.join(output, "processed", "preprocessing")
        os.makedirs(preproc_output, exist_ok=True)
        preproc_normalizer.process_batch(current_input, preproc_output)
        current_input = preproc_output  # La sortie de cette étape est l'entrée de la suivante
    
    if "colometry" in stages:
        click.echo("Normalisation colométrie...")
        # Utiliser l'injection de dépendances
        colometry_normalizer = get_colometry_normalizer(config_path=config)
        colometry_output = os.path.join(output, "colometry")
        os.makedirs(colometry_output, exist_ok=True)
        colometry_normalizer.process_batch(current_input, colometry_output)
        current_input = colometry_output  # Mettre à jour current_input pour les étapes suivantes
    
    if "geometry" in stages:
        click.echo("Étape 2: Normalisation géométrie...")
        # Utiliser l'injection de dépendances
        geometry_normalizer = get_geometry_normalizer(config_path=config)
        geometry_output = os.path.join(output, "geometry")
        os.makedirs(geometry_output, exist_ok=True)
        geometry_normalizer.process_batch(current_input, geometry_output)
        current_input = geometry_output  # Mettre à jour current_input pour les étapes suivantes
    
    if "features" in stages:
        click.echo("Extraction de features...")
        # Utiliser l'injection de dépendances
        feature_extractor = get_feature_extractor(config_path=config)
        features_output = os.path.join(output, "features")
        os.makedirs(features_output, exist_ok=True)
        feature_extractor.process_batch(current_input, features_output)
        current_input = features_output  # Mettre à jour pour la classification
    
    if "classification" in stages:
        click.echo("Classification de documents...")
        try:
            # Utiliser l'injection de dépendances
            classifier = get_document_classifier(config_path=config)
            
            import json
            from pathlib import Path
            from src.utils.file_handler import ensure_dir, get_files
            
            classification_output = os.path.join(output, "classification")
            os.makedirs(classification_output, exist_ok=True)
            
            # Parcourir les fichiers JSON d'entrée
            json_files = get_files(current_input, extensions=['.json'])
            
            if not json_files:
                click.echo(f"Aucun fichier JSON trouvé dans {current_input}")
            else:
                click.echo(f"Traitement de {len(json_files)} fichiers...")
                
                for json_file in json_files:
                    try:
                        # Charger le JSON OCR
                        with open(json_file, 'r', encoding='utf-8') as f:
                            ocr_data = json.load(f)
                        
                        # Classifier
                        classification_result = classifier.predict(ocr_data)
                        
                        # Sauvegarder le résultat
                        output_file = Path(classification_output) / Path(json_file).name
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(classification_result, f, indent=2, ensure_ascii=False)
                    
                    except Exception as e:
                        click.echo(f"Erreur lors du traitement de {json_file}: {e}", err=True)
                        continue
                
                click.echo(f"Classification terminée: {len(json_files)} documents traités")
        
        except ValueError as e:
            click.echo(f"⚠️  Classification ignorée: {e}", err=True)
            # Ne pas faire échouer le pipeline si la classification n'est pas activée
    
    click.echo("Pipeline terminé!")


if __name__ == "__main__":
    cli()

