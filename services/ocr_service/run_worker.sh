#!/bin/bash
# Script pour lancer le worker Dramatiq du service OCR isolé
# 
# Ce script configure l'environnement et lance un worker Dramatiq
# qui écoute les tâches OCR dans Redis.

set -e  # Arrêter en cas d'erreur

# Couleurs pour les messages
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Service OCR - Démarrage du Worker${NC}"
echo -e "${GREEN}========================================${NC}"

# Obtenir le répertoire du script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${YELLOW}[1/4] Vérification de l'environnement...${NC}"

# Vérifier que Python est disponible
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERREUR: python3 n'est pas installé ou n'est pas dans le PATH${NC}"
    exit 1
fi

# Vérifier que Poetry est disponible (optionnel, mais recommandé)
if ! command -v poetry &> /dev/null; then
    echo -e "${YELLOW}AVERTISSEMENT: Poetry n'est pas installé.${NC}"
    echo -e "${YELLOW}Assurez-vous que les dépendances sont installées dans votre environnement Python.${NC}"
    USE_POETRY=false
else
    USE_POETRY=true
    echo -e "${GREEN}Poetry détecté.${NC}"
fi

echo -e "${YELLOW}[2/4] Installation/Vérification des dépendances...${NC}"

if [ "$USE_POETRY" = true ]; then
    # Installer les dépendances avec Poetry
    poetry install --no-interaction
    PYTHON_CMD="poetry run python"
else
    # Utiliser Python directement
    PYTHON_CMD="python3"
    echo -e "${YELLOW}Utilisation de Python directement (sans Poetry).${NC}"
fi

echo -e "${YELLOW}[3/4] Vérification de la configuration...${NC}"

# Vérifier que config.yaml existe
if [ ! -f "config.yaml" ]; then
    echo -e "${RED}ERREUR: config.yaml non trouvé dans $SCRIPT_DIR${NC}"
    exit 1
fi

echo -e "${GREEN}Configuration trouvée.${NC}"

echo -e "${YELLOW}[4/4] Lancement du worker Dramatiq...${NC}"

# Variables d'environnement optionnelles
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Lancer le worker Dramatiq
# Le worker va automatiquement charger actors.py et écouter les tâches
echo -e "${GREEN}Worker en cours de démarrage...${NC}"
echo -e "${YELLOW}Appuyez sur Ctrl+C pour arrêter le worker.${NC}"
echo ""

# Récupérer le nombre de processus et threads depuis les variables d'environnement
# Défaut: 1 processus, 1 thread (pour économiser les ressources en développement)
OCR_WORKER_PROCESSES=${OCR_WORKER_PROCESSES:-1}
OCR_WORKER_THREADS=${OCR_WORKER_THREADS:-1}

echo -e "${GREEN}Configuration: ${OCR_WORKER_PROCESSES} processus, ${OCR_WORKER_THREADS} thread(s) par processus${NC}"
echo -e "${GREEN}Total: $((OCR_WORKER_PROCESSES * OCR_WORKER_THREADS)) worker(s) OCR${NC}"
echo ""

python3 -m dramatiq actors --processes ${OCR_WORKER_PROCESSES:-1} --threads ${OCR_WORKER_THREADS:-1} --queues ocr-queue