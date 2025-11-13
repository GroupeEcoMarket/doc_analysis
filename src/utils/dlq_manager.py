"""
Gestionnaire de Dead Letter Queue (DLQ) pour Dramatiq.

Ce module fournit des utilitaires pour inspecter, analyser et rejouer
les messages qui ont échoué et ont été envoyés vers la Dead Letter Queue.
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from dramatiq.brokers.redis import RedisBroker
from dramatiq.message import Message

from src.utils.logger import get_logger
from src.utils.config_loader import get_config

logger = get_logger(__name__)


class DLQManager:
    """
    Gestionnaire pour la Dead Letter Queue de Dramatiq.
    
    Permet d'inspecter, analyser et rejouer les messages échoués.
    """
    
    def __init__(self, broker: Optional[RedisBroker] = None):
        """
        Initialise le gestionnaire DLQ.
        
        Args:
            broker: Broker Dramatiq (si None, utilise le broker par défaut)
        """
        import dramatiq
        
        if broker is None:
            broker = dramatiq.get_broker()
        
        self.broker = broker
        
        # Le nom de la DLQ est fixe dans Dramatiq (géré automatiquement)
        # Dramatiq utilise toujours 'dramatiq:dead_letter' comme nom de queue
        self.dlq_name = 'dramatiq:dead_letter'
        
        # Accès direct à Redis pour inspecter la DLQ
        if isinstance(broker, RedisBroker):
            self.redis_client = broker.client
        else:
            raise ValueError("DLQManager ne supporte que RedisBroker pour l'instant")
        
        logger.info(f"DLQManager initialisé avec DLQ: {self.dlq_name}")
    
    def get_dlq_length(self) -> int:
        """
        Retourne le nombre de messages dans la DLQ.
        
        Returns:
            int: Nombre de messages dans la DLQ
        """
        try:
            length = self.redis_client.llen(self.dlq_name)
            return length
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de la longueur de la DLQ: {e}")
            return 0
    
    def list_messages(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Liste les messages dans la DLQ.
        
        Args:
            limit: Nombre maximum de messages à retourner (défaut: 100)
        
        Returns:
            List[Dict[str, Any]]: Liste des messages avec leurs métadonnées
        """
        messages = []
        
        try:
            # Récupérer les messages depuis Redis (sans les supprimer)
            raw_messages = self.redis_client.lrange(self.dlq_name, 0, limit - 1)
            
            for raw_msg in raw_messages:
                try:
                    # Décoder le message JSON
                    msg_data = json.loads(raw_msg.decode('utf-8'))
                    
                    # Extraire les informations pertinentes
                    message_info = {
                        'message_id': msg_data.get('message_id'),
                        'actor_name': msg_data.get('actor_name'),
                        'args': msg_data.get('args', []),
                        'kwargs': msg_data.get('kwargs', {}),
                        'queue_name': msg_data.get('queue_name'),
                        'retries': msg_data.get('options', {}).get('retries', 0),
                        'max_retries': msg_data.get('options', {}).get('max_retries', 0),
                        'timestamp': msg_data.get('message_timestamp'),
                        'raw_message': msg_data
                    }
                    
                    messages.append(message_info)
                except Exception as e:
                    logger.warning(f"Impossible de parser un message de la DLQ: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des messages de la DLQ: {e}")
        
        return messages
    
    def get_message_details(self, message_id: str) -> Optional[Dict[str, Any]]:
        """
        Récupère les détails d'un message spécifique dans la DLQ.
        
        Args:
            message_id: ID du message à récupérer
        
        Returns:
            Dict[str, Any]: Détails du message, ou None si non trouvé
        """
        messages = self.list_messages(limit=1000)  # Chercher dans les 1000 premiers
        
        for msg in messages:
            if msg['message_id'] == message_id:
                return msg
        
        return None
    
    def replay_message(self, message_id: str) -> bool:
        """
        Rejoue un message depuis la DLQ.
        
        Le message est retiré de la DLQ et renvoyé dans la queue normale
        pour être traité à nouveau.
        
        Args:
            message_id: ID du message à rejouer
        
        Returns:
            bool: True si le message a été rejoué avec succès, False sinon
        """
        try:
            # Récupérer tous les messages de la DLQ
            raw_messages = self.redis_client.lrange(self.dlq_name, 0, -1)
            
            message_to_replay = None
            remaining_messages = []
            
            # Trouver le message à rejouer
            for raw_msg in raw_messages:
                try:
                    msg_data = json.loads(raw_msg.decode('utf-8'))
                    if msg_data.get('message_id') == message_id:
                        message_to_replay = msg_data
                    else:
                        remaining_messages.append(raw_msg)
                except:
                    remaining_messages.append(raw_msg)
            
            if message_to_replay is None:
                logger.warning(f"Message {message_id} non trouvé dans la DLQ")
                return False
            
            # Reconstruire le message Dramatiq
            message = Message(
                message_id=message_to_replay['message_id'],
                queue_name=message_to_replay['queue_name'],
                actor_name=message_to_replay['actor_name'],
                args=message_to_replay.get('args', []),
                kwargs=message_to_replay.get('kwargs', {}),
                options=message_to_replay.get('options', {})
            )
            
            # Réinitialiser le compteur de retries pour permettre un nouveau traitement
            message.options['retries'] = 0
            
            # Envoyer le message dans la queue normale
            self.broker.enqueue(message)
            
            # Supprimer le message de la DLQ et remettre les autres
            self.redis_client.delete(self.dlq_name)
            if remaining_messages:
                self.redis_client.rpush(self.dlq_name, *remaining_messages)
            
            logger.info(f"Message {message_id} rejoué avec succès")
            return True
        
        except Exception as e:
            logger.error(f"Erreur lors du rejeu du message {message_id}: {e}", exc_info=True)
            return False
    
    def delete_message(self, message_id: str) -> bool:
        """
        Supprime définitivement un message de la DLQ.
        
        Args:
            message_id: ID du message à supprimer
        
        Returns:
            bool: True si le message a été supprimé, False sinon
        """
        try:
            # Récupérer tous les messages de la DLQ
            raw_messages = self.redis_client.lrange(self.dlq_name, 0, -1)
            
            remaining_messages = []
            found = False
            
            # Filtrer le message à supprimer
            for raw_msg in raw_messages:
                try:
                    msg_data = json.loads(raw_msg.decode('utf-8'))
                    if msg_data.get('message_id') == message_id:
                        found = True
                    else:
                        remaining_messages.append(raw_msg)
                except:
                    remaining_messages.append(raw_msg)
            
            if not found:
                logger.warning(f"Message {message_id} non trouvé dans la DLQ")
                return False
            
            # Remettre les messages restants dans la DLQ
            self.redis_client.delete(self.dlq_name)
            if remaining_messages:
                self.redis_client.rpush(self.dlq_name, *remaining_messages)
            
            logger.info(f"Message {message_id} supprimé de la DLQ")
            return True
        
        except Exception as e:
            logger.error(f"Erreur lors de la suppression du message {message_id}: {e}", exc_info=True)
            return False
    
    def clear_dlq(self) -> int:
        """
        Vide complètement la DLQ.
        
        Returns:
            int: Nombre de messages supprimés
        """
        try:
            length = self.get_dlq_length()
            self.redis_client.delete(self.dlq_name)
            logger.warning(f"DLQ vidée: {length} message(s) supprimé(s)")
            return length
        except Exception as e:
            logger.error(f"Erreur lors du vidage de la DLQ: {e}", exc_info=True)
            return 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retourne des statistiques sur la DLQ.
        
        Returns:
            Dict[str, Any]: Statistiques (nombre de messages, répartition par actor, etc.)
        """
        messages = self.list_messages(limit=1000)
        
        stats = {
            'total_messages': len(messages),
            'by_actor': {},
            'oldest_message': None,
            'newest_message': None
        }
        
        timestamps = []
        
        for msg in messages:
            actor_name = msg.get('actor_name', 'unknown')
            stats['by_actor'][actor_name] = stats['by_actor'].get(actor_name, 0) + 1
            
            if msg.get('timestamp'):
                timestamps.append(msg['timestamp'])
        
        if timestamps:
            stats['oldest_message'] = min(timestamps)
            stats['newest_message'] = max(timestamps)
        
        return stats

