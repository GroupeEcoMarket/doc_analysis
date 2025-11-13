#!/usr/bin/env python3
"""
Script de healthcheck pour le service OCR dans Docker.

Ce script vérifie que le service OCR est opérationnel en envoyant une tâche
de warm-up et en attendant le résultat. Il est utilisé par Docker Compose
pour déterminer si le service est sain.

Usage:
    python healthcheck.py

Exit codes:
    0: Service OCR opérationnel (moteur initialisé)
    1: Service OCR non opérationnel (timeout, erreur, ou moteur non initialisé)
"""

import os
import sys
import time
from typing import Dict, Any

import dramatiq
from dramatiq.brokers.redis import RedisBroker
from dramatiq.results import Results
from dramatiq.results.backends import RedisBackend

# Configuration du broker Dramatiq
# Utilise REDIS_HOST depuis l'environnement (défini dans docker-compose.yml)
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", "6379"))
# Construire l'URL Redis pour Dramatiq
redis_url = f"redis://{redis_host}:{redis_port}"
redis_broker = RedisBroker(url=redis_url, dead_message_ttl=604800000)
result_backend = RedisBackend(url=redis_url)
redis_broker.add_middleware(Results(backend=result_backend))
dramatiq.set_broker(redis_broker)

# Acteur proxy pour warmup_ocr_worker
# Cet acteur pointe vers la queue ocr-queue du microservice OCR
@dramatiq.actor(
    queue_name="ocr-queue",
    time_limit=30000,  # 30 secondes pour le warm-up
    max_retries=0,
    store_results=True
)
def warmup_ocr_worker(**kwargs) -> Dict[str, Any]:
    """
    Acteur proxy pour envoyer une tâche de warm-up au microservice OCR.
    
    Cette fonction ne devrait jamais être appelée directement.
    Elle sert uniquement de proxy pour envoyer des messages via .send()
    """
    raise NotImplementedError(
        "warmup_ocr_worker est un acteur proxy. "
        "Utilisez warmup_ocr_worker.send() pour envoyer une tâche de warm-up."
    )


def main() -> int:
    """
    Exécute le healthcheck du service OCR.
    
    Returns:
        0 si le service est opérationnel, 1 sinon
    """
    timeout_seconds = 15
    
    try:
        # Envoyer la tâche de warm-up
        message = warmup_ocr_worker.send()
        
        # Attendre le résultat avec timeout
        start_time = time.time()
        result = None
        
        while time.time() - start_time < timeout_seconds:
            try:
                # Essayer de récupérer le résultat (non-bloquant)
                result = message.get_result(block=False)
                break
            except dramatiq.results.ResultMissing:
                # Le résultat n'est pas encore disponible, attendre un peu
                time.sleep(0.5)
                continue
            except Exception as e:
                # Erreur lors de la récupération du résultat
                print(f"ERREUR [healthcheck]: Erreur lors de la récupération du résultat: {e}", file=sys.stderr)
                return 1
        
        # Vérifier si on a un résultat
        if result is None:
            print(f"ERREUR [healthcheck]: Timeout après {timeout_seconds}s - le service OCR n'a pas répondu", file=sys.stderr)
            return 1
        
        # Vérifier le contenu du résultat
        if not isinstance(result, dict):
            print(f"ERREUR [healthcheck]: Résultat invalide (type: {type(result)})", file=sys.stderr)
            return 1
        
        status = result.get('status')
        engine_initialized = result.get('engine_initialized', False)
        
        if status != 'ready':
            print(f"ERREUR [healthcheck]: Statut invalide: {status}", file=sys.stderr)
            return 1
        
        if not engine_initialized:
            print(f"ERREUR [healthcheck]: Moteur OCR non initialisé", file=sys.stderr)
            return 1
        
        # Tout est OK
        worker_pid = result.get('worker_pid', 'unknown')
        print(f"OK [healthcheck]: Service OCR opérationnel (Worker PID: {worker_pid})")
        return 0
        
    except Exception as e:
        print(f"ERREUR [healthcheck]: Exception inattendue: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

