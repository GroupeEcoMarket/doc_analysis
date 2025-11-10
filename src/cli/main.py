"""
Command-line interface for document analysis pipeline
"""

import click
import os
from pathlib import Path
from src.pipeline import PreprocessingNormalizer, ColometryNormalizer, GeometryNormalizer, FeatureExtractor
from src.utils.config import get_config


@click.group()
def cli():
    """Document Analysis Pipeline CLI"""
    pass


@cli.command()
@click.option("--input", "-i", required=True, help="Répertoire d'entrée")
@click.option("--output", "-o", required=True, help="Répertoire de sortie")
@click.option("--config", "-c", help="Fichier de configuration")
def colometry(input, output, config):
    """Normalisation colométrie"""
    click.echo(f"Normalisation colométrie: {input} -> {output}")
    
    normalizer = ColometryNormalizer(config=config)
    results = normalizer.process_batch(input, output)
    
    click.echo(f"Traitement terminé: {len(results)} documents traités")


@cli.command()
@click.option("--input", "-i", required=True, help="Répertoire d'entrée")
@click.option("--output", "-o", required=True, help="Répertoire de sortie")
@click.option("--config", "-c", help="Fichier de configuration")
def geometry(input, output, config):
    """Normalisation géométrie"""
    click.echo(f"Normalisation géométrie: {input} -> {output}")
    
    normalizer = GeometryNormalizer(config=config)
    results = normalizer.process_batch(input, output)
    
    click.echo(f"Traitement terminé: {len(results)} documents traités")


@cli.command()
@click.option("--input", "-i", required=True, help="Répertoire d'entrée")
@click.option("--output", "-o", required=True, help="Répertoire de sortie")
@click.option("--config", "-c", help="Fichier de configuration")
def preprocessing(input, output, config):
    """Étape de prétraitement : amélioration contraste et classification."""
    click.echo(f"Prétraitement des images : {input} -> {output}")
    
    normalizer = PreprocessingNormalizer(config=config)
    results = normalizer.process_batch(input, output)
    
    click.echo(f"Traitement terminé: {len(results)} documents prétraités")


@cli.command()
@click.option("--input", "-i", required=True, help="Répertoire d'entrée")
@click.option("--output", "-o", required=True, help="Répertoire de sortie")
@click.option("--config", "-c", help="Fichier de configuration")
def features(input, output, config):
    """Extraction de features"""
    click.echo(f"Extraction de features: {input} -> {output}")
    
    extractor = FeatureExtractor(config=config)
    results = extractor.process_batch(input, output)
    
    click.echo(f"Traitement terminé: {len(results)} documents traités")


@cli.command()
@click.option("--input", "-i", required=True, help="Répertoire d'entrée")
@click.option("--output", "-o", required=True, help="Répertoire de sortie")
@click.option("--config", "-c", help="Fichier de configuration")
@click.option("--stages", "-s", multiple=True, 
              type=click.Choice(["preprocessing", "colometry", "geometry", "features"]),
              help="Étapes à exécuter (peut être répété plusieurs fois)")
def pipeline(input, output, config, stages):
    """Exécute le pipeline complet ou des étapes spécifiques"""
    config_obj = get_config(config_file=config) if config else get_config()
    
    # Si aucune étape spécifiée, exécuter toutes les étapes par défaut
    if not stages:
        stages = ["preprocessing", "geometry", "features"]
    
    # Ordre logique des étapes (pour référence)
    # preprocessing → geometry → features
    # colometry peut être exécuté indépendamment ou avant geometry
    
    current_input = input
    
    if "preprocessing" in stages:
        click.echo("Étape 1: Prétraitement des images...")
        preproc_normalizer = PreprocessingNormalizer(config=config_obj)
        preproc_output = os.path.join(output, "processed", "preprocessing")
        os.makedirs(preproc_output, exist_ok=True)
        preproc_normalizer.process_batch(current_input, preproc_output)
        current_input = preproc_output  # La sortie de cette étape est l'entrée de la suivante
    
    if "colometry" in stages:
        click.echo("Normalisation colométrie...")
        colometry_normalizer = ColometryNormalizer(config=config_obj)
        colometry_output = os.path.join(output, "colometry")
        os.makedirs(colometry_output, exist_ok=True)
        colometry_normalizer.process_batch(current_input, colometry_output)
        current_input = colometry_output  # Mettre à jour current_input pour les étapes suivantes
    
    if "geometry" in stages:
        click.echo("Étape 2: Normalisation géométrie...")
        geometry_normalizer = GeometryNormalizer(config=config_obj)
        geometry_output = os.path.join(output, "geometry")
        os.makedirs(geometry_output, exist_ok=True)
        geometry_normalizer.process_batch(current_input, geometry_output)
        current_input = geometry_output  # Mettre à jour current_input pour les étapes suivantes
    
    if "features" in stages:
        click.echo("Extraction de features...")
        feature_extractor = FeatureExtractor(config=config_obj)
        features_output = os.path.join(output, "features")
        os.makedirs(features_output, exist_ok=True)
        feature_extractor.process_batch(current_input, features_output)
        # Pas besoin de mettre à jour current_input car c'est la dernière étape
    
    click.echo("Pipeline terminé!")


if __name__ == "__main__":
    cli()

