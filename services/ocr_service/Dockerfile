# services/ocr_service/Dockerfile
# Version pinée pour garantir la reproductibilité des builds

# Stage 1: Builder - Installation des dépendances avec Poetry
FROM python:3.11.9-slim as builder

# Installer les outils de build nécessaires pour compiler certaines dépendances
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installer Poetry
ENV POETRY_VERSION=1.8.3 \
    POETRY_HOME=/opt/poetry \
    POETRY_VENV=/opt/poetry-venv \
    POETRY_CACHE_DIR=/opt/.cache

RUN python3 -m venv $POETRY_VENV && \
    $POETRY_VENV/bin/pip install -U pip setuptools && \
    $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION} && \
    ln -s ${POETRY_VENV}/bin/poetry /usr/local/bin/poetry && \
    chmod +x /usr/local/bin/poetry

RUN poetry config virtualenvs.create false

WORKDIR /app

# Copier uniquement les fichiers de dépendances
COPY pyproject.toml poetry.lock* ./

# Installer les dépendances (sans les dépendances de dev et sans le package root)
RUN poetry install --no-root --only main --no-interaction --no-ansi

# Stage 2: Runtime - Image finale optimisée
FROM python:3.11.9-slim as runtime

# Installer les dépendances système nécessaires pour PaddleOCR et OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Dépendances pour OpenCV
    libgl1-mesa-glx \
    libglib2.0-0 \
    # Dépendances pour PaddleOCR
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Créer un utilisateur non-root pour la sécurité
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

# Copier l'environnement Python depuis le builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Définir le répertoire de travail
WORKDIR /app

# Copier le code source nécessaire
COPY --chown=appuser:appuser __init__.py actors.py healthcheck.py config.yaml run_worker.sh ./
COPY --chown=appuser:appuser pyproject.toml ./

# Rendre le script de lancement exécutable
RUN chmod +x run_worker.sh

# Forcer le téléchargement des modèles PaddleOCR pendant le build
# On exécute un petit script Python en ligne de commande
# qui initialise PaddleOCR. Cette initialisation déclenchera le téléchargement.
# On le fait en tant que root pour que les modèles soient dans un cache système.
RUN python3 -c "from paddleocr import PaddleOCR; print('Pré-téléchargement du modèle OCR...'); ocr = PaddleOCR(lang='fr', use_textline_orientation=False, device='cpu'); print('Modèle OCR pré-téléchargé.')"

# S'assurer que les modèles téléchargés sont accessibles par 'appuser'
# Le cache de PaddleOCR est généralement dans /root/.paddleocr/
RUN if [ -d /root/.paddleocr ]; then chown -R appuser:appuser /root/.paddleocr; fi

# Passer à l'utilisateur non-root
USER appuser

# Le `command` dans docker-compose.yml écrasera ceci, mais c'est une bonne pratique
CMD ["./run_worker.sh"]

