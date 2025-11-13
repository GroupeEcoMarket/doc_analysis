"""
Système de stockage temporaire basé sur des identifiants abstraits.

Ce module gère le stockage temporaire de fichiers pour les workers Dramatiq,
remplaçant le système de passage de données en Base64 par un système basé sur
des identifiants abstraits (noms de fichiers). Cela réduit drastiquement la charge
sur l'API, le broker Redis et le réseau, tout en rendant la communication entre
services plus robuste et portable.

Supporte différents backends de stockage (local, S3, MinIO, etc.) via une
interface abstraite BaseStorage. Le backend est configuré via la configuration
(env.storage.backend, défaut: "local").
"""

import uuid
import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from src.utils.logger import get_logger
from src.config.manager import get_config_manager

logger = get_logger(__name__)

# Répertoire de stockage temporaire par défaut
_DEFAULT_STORAGE_DIR = "data/temp_storage"


class BaseStorage(ABC):
    """
    Classe abstraite définissant l'interface pour les backends de stockage.
    
    Tous les backends de stockage doivent implémenter ces méthodes pour
    garantir une interface cohérente, que ce soit pour un stockage local,
    S3, MinIO, ou tout autre système de stockage distribué.
    """
    
    @abstractmethod
    def save_file(self, file_content: bytes, filename: Optional[str] = None, prefix: str = "file_") -> str:
        """
        Sauvegarde un fichier et retourne son identifiant.
        
        Args:
            file_content: Contenu du fichier en bytes
            filename: Nom original du fichier (optionnel, pour l'extension)
            prefix: Préfixe pour l'identifiant (défaut: "file_")
        
        Returns:
            str: Identifiant du fichier sauvegardé (nom du fichier uniquement)
        """
        pass
    
    @abstractmethod
    def save_image(self, image_np, page_index: int, task_id: Optional[str] = None) -> str:
        """
        Sauvegarde une image numpy et retourne son identifiant.
        
        Args:
            image_np: Image numpy array
            page_index: Index de la page (pour le nom de fichier)
            task_id: ID de la tâche (optionnel, pour organisation)
        
        Returns:
            str: Identifiant de l'image sauvegardée (nom du fichier uniquement)
        """
        pass
    
    @abstractmethod
    def load_file(self, identifier: str) -> bytes:
        """
        Charge un fichier depuis son identifiant.
        
        Args:
            identifier: Identifiant du fichier (nom du fichier)
        
        Returns:
            bytes: Contenu du fichier
        
        Raises:
            FileNotFoundError: Si le fichier n'existe pas
            ValueError: Si l'identifiant est invalide
        """
        pass
    
    @abstractmethod
    def load_image(self, identifier: str):
        """
        Charge une image depuis son identifiant.
        
        Args:
            identifier: Identifiant de l'image (nom du fichier)
        
        Returns:
            np.ndarray: Image numpy array
        
        Raises:
            FileNotFoundError: Si l'image n'existe pas
            ValueError: Si l'identifiant est invalide
        """
        pass
    
    @abstractmethod
    def delete_file(self, identifier: str) -> bool:
        """
        Supprime un fichier et ses métadonnées.
        
        Args:
            identifier: Identifiant du fichier à supprimer (nom du fichier)
        
        Returns:
            bool: True si le fichier a été supprimé, False sinon
        """
        pass
    
    def cleanup_old_files(self, max_age_hours: Optional[int] = None) -> int:
        """
        Nettoie les fichiers plus anciens que max_age_hours.
        
        Cette méthode peut être surchargée par les implémentations spécifiques,
        mais fournit une implémentation par défaut qui peut ne pas être applicable
        à tous les backends (ex: S3 avec lifecycle policies).
        
        Args:
            max_age_hours: Nombre d'heures maximum
        
        Returns:
            int: Nombre de fichiers supprimés
        """
        # Implémentation par défaut vide (peut être surchargée)
        return 0


class LocalStorage(BaseStorage):
    """
    Implémentation de stockage local basée sur le système de fichiers.
    
    Les fichiers sont stockés localement avec un identifiant unique (nom du fichier)
    et peuvent être récupérés par cet identifiant. Un système de nettoyage automatique
    supprime les fichiers après un certain délai.
    
    Cette implémentation utilise des identifiants abstraits (noms de fichiers) plutôt
    que des chemins absolus, rendant la communication entre services plus robuste et portable.
    """
    
    def __init__(self, storage_dir: Optional[str] = None, cleanup_after_hours: int = 24):
        """
        Initialise le gestionnaire de stockage temporaire.
        
        Args:
            storage_dir: Répertoire de stockage (défaut: data/temp_storage)
            cleanup_after_hours: Nombre d'heures avant nettoyage automatique (défaut: 24)
        """
        if storage_dir is None:
            # Essayer de récupérer depuis la config, sinon utiliser le défaut
            try:
                config = get_config()
                storage_dir = config.get('paths.temp_storage_dir', _DEFAULT_STORAGE_DIR)
            except:
                storage_dir = _DEFAULT_STORAGE_DIR
        
        self.storage_dir = Path(storage_dir)
        self.cleanup_after_hours = cleanup_after_hours
        
        # Créer le répertoire de stockage s'il n'existe pas
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Stockage local initialisé: {self.storage_dir}")
    
    def save_file(self, file_content: bytes, filename: Optional[str] = None, prefix: str = "file_") -> str:
        """
        Sauvegarde un fichier et retourne son identifiant.
        
        Args:
            file_content: Contenu du fichier en bytes
            filename: Nom original du fichier (optionnel, pour l'extension)
            prefix: Préfixe pour l'identifiant (défaut: "file_")
        
        Returns:
            str: Identifiant du fichier sauvegardé (nom du fichier uniquement)
        """
        # Générer un identifiant unique
        file_id = f"{prefix}{uuid.uuid4().hex}"
        
        # Déterminer l'extension depuis le filename si disponible
        extension = ""
        if filename:
            extension = Path(filename).suffix
        
        # Chemin complet du fichier
        file_path = self.storage_dir / f"{file_id}{extension}"
        
        # Sauvegarder le fichier
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        # Créer un fichier de métadonnées avec la date de création
        metadata_path = self.storage_dir / f"{file_id}.meta"
        metadata = {
            "file_id": file_id,
            "filename": filename,
            "created_at": datetime.now().isoformat(),
            "size_bytes": len(file_content)
        }
        import json
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)
        
        # Le nom du fichier est notre identifiant
        identifier = file_path.name
        
        logger.debug(f"Fichier sauvegardé sous l'identifiant: {identifier} ({len(file_content)} bytes)")
        
        return identifier
    
    def save_image(self, image_np, page_index: int, task_id: Optional[str] = None) -> str:
        """
        Sauvegarde une image numpy et retourne son identifiant.
        
        Args:
            image_np: Image numpy array
            page_index: Index de la page (pour le nom de fichier)
            task_id: ID de la tâche (optionnel, pour organisation)
        
        Returns:
            str: Identifiant de l'image sauvegardée (nom du fichier uniquement)
        """
        import cv2
        
        # Générer un identifiant unique
        file_id = f"page_{page_index}_{uuid.uuid4().hex[:8]}"
        if task_id:
            file_id = f"{task_id}_{file_id}"
        
        # Chemin complet de l'image
        image_path = self.storage_dir / f"{file_id}.png"
        
        # Sauvegarder l'image
        cv2.imwrite(str(image_path), image_np)
        
        # Créer un fichier de métadonnées
        metadata_path = self.storage_dir / f"{file_id}.meta"
        metadata = {
            "file_id": file_id,
            "page_index": page_index,
            "task_id": task_id,
            "created_at": datetime.now().isoformat(),
            "image_shape": list(image_np.shape),
            "size_bytes": Path(image_path).stat().st_size
        }
        import json
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)
        
        # Le nom du fichier est notre identifiant
        identifier = image_path.name
        
        logger.debug(f"Image sauvegardée sous l'identifiant: {identifier} (page {page_index + 1})")
        
        return identifier
    
    def load_file(self, identifier: str) -> bytes:
        """
        Charge un fichier depuis son identifiant.
        
        Args:
            identifier: Identifiant du fichier (nom du fichier)
        
        Returns:
            bytes: Contenu du fichier
        
        Raises:
            FileNotFoundError: Si le fichier n'existe pas
            ValueError: Si l'identifiant est invalide
        """
        # Reconstruire le chemin complet
        file_path = self.storage_dir / identifier
        
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier non trouvé: {file_path}")
        
        with open(file_path, "rb") as f:
            content = f.read()
        
        logger.debug(f"Fichier chargé: {identifier} ({len(content)} bytes)")
        
        return content
    
    def load_image(self, identifier: str):
        """
        Charge une image depuis son identifiant.
        
        Args:
            identifier: Identifiant de l'image (nom du fichier)
        
        Returns:
            np.ndarray: Image numpy array
        
        Raises:
            FileNotFoundError: Si l'image n'existe pas
            ValueError: Si l'identifiant est invalide
        """
        import cv2
        
        # Reconstruire le chemin complet
        image_path = self.storage_dir / identifier
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image non trouvée: {image_path}")
        
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError(f"Impossible de charger l'image: {image_path}")
        
        logger.debug(f"Image chargée: {identifier} (shape: {image.shape})")
        
        return image
    
    def delete_file(self, identifier: str) -> bool:
        """
        Supprime un fichier et ses métadonnées.
        
        Args:
            identifier: Identifiant du fichier à supprimer (nom du fichier)
        
        Returns:
            bool: True si le fichier a été supprimé, False sinon
        """
        # Reconstruire le chemin complet
        file_path = self.storage_dir / identifier
        
        deleted = False
        if file_path.exists():
            try:
                file_path.unlink()
                deleted = True
            except Exception as e:
                logger.warning(f"Impossible de supprimer le fichier {file_path}: {e}")
        
        # Supprimer aussi les métadonnées si elles existent
        metadata_path = self.storage_dir / f"{file_path.stem}.meta"
        if metadata_path.exists():
            try:
                metadata_path.unlink()
            except Exception as e:
                logger.debug(f"Impossible de supprimer les métadonnées {metadata_path}: {e}")
        
        return deleted
    
    def cleanup_old_files(self, max_age_hours: Optional[int] = None) -> int:
        """
        Nettoie les fichiers plus anciens que max_age_hours.
        
        Args:
            max_age_hours: Nombre d'heures maximum (défaut: self.cleanup_after_hours)
        
        Returns:
            int: Nombre de fichiers supprimés
        """
        if max_age_hours is None:
            max_age_hours = self.cleanup_after_hours
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        deleted_count = 0
        
        if not self.storage_dir.exists():
            return 0
        
        for file_path in self.storage_dir.iterdir():
            # Ignorer les fichiers de métadonnées
            if file_path.suffix == ".meta":
                continue
            
            try:
                # Vérifier l'âge du fichier
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_time:
                    # Supprimer le fichier et ses métadonnées
                    self.delete_file(file_path.name)
                    deleted_count += 1
            except Exception as e:
                logger.warning(f"Erreur lors du nettoyage de {file_path}: {e}")
        
        if deleted_count > 0:
            logger.info(f"Nettoyage de {deleted_count} ancien(s) fichier(s) temporaire(s)")
        
        return deleted_count


# Instance globale de stockage
_storage_instance: Optional[BaseStorage] = None


def get_storage() -> BaseStorage:
    """
    Récupère l'instance globale de stockage.
    
    Le backend de stockage est déterminé par la variable d'environnement
    STORAGE_BACKEND (défaut: "local").
    
    Backends supportés:
    - "local": LocalStorage (stockage sur le système de fichiers local)
    - "s3": S3Storage (à implémenter pour AWS S3)
    - "minio": MinIOStorage (à implémenter pour MinIO)
    
    Returns:
        BaseStorage: Instance du gestionnaire de stockage
    """
    global _storage_instance
    
    if _storage_instance is None:
        # Récupérer le backend depuis la configuration
        config = get_config_manager()
        storage_backend = config.get("env", "storage", "backend", default="local").lower()
        
        if storage_backend == "local":
            _storage_instance = LocalStorage()
        elif storage_backend == "s3":
            # TODO: Implémenter S3Storage
            raise NotImplementedError(
                "S3Storage n'est pas encore implémenté. "
                "Utilisez STORAGE_BACKEND=local pour l'instant."
            )
        elif storage_backend == "minio":
            # TODO: Implémenter MinIOStorage
            raise NotImplementedError(
                "MinIOStorage n'est pas encore implémenté. "
                "Utilisez STORAGE_BACKEND=local pour l'instant."
            )
        else:
            logger.warning(
                f"Backend de stockage inconnu: {storage_backend}. "
                f"Utilisation du backend local par défaut."
            )
            _storage_instance = LocalStorage()
        
        logger.info(f"Backend de stockage sélectionné: {storage_backend}")
    
    return _storage_instance

