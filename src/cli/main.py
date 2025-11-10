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
@click.option("--input", "-i", required=True, help="Répertoire d'entrée")
@click.option("--output", "-o", required=True, help="Répertoire de sortie")
@click.option("--config", "-c", help="Fichier de configuration")
@click.option("--stages", "-s", multiple=True, 
              type=click.Choice(["preprocessing", "colometry", "geometry", "features"]),
              help="Étapes à exécuter (peut être répété plusieurs fois)")
def pipeline(input: str, output: str, config: Optional[str], stages: Tuple[str, ...]) -> None:
    """Exécute le pipeline complet ou des étapes spécifiques"""
    # Charger la configuration une seule fois
    app_config = get_app_config(config_path=config)
    
    # Si aucune étape spécifiée, exécuter toutes les étapes par défaut
    if not stages:
        stages = ["preprocessing", "geometry", "features"]
    
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
        # Pas besoin de mettre à jour current_input car c'est la dernière étape
    
    click.echo("Pipeline terminé!")


if __name__ == "__main__":
    cli()

