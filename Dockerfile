# Dockerfile multi-stage pour l'application principale (API + Workers)
# Stage 1: Builder - Installation des dépendances avec Poetry
# Version pinée pour garantir la reproductibilité des builds
FROM python:3.11.9-slim as builder

# Variables d'environnement pour Poetry
ENV POETRY_VERSION=1.8.3 \
    POETRY_HOME=/opt/poetry \
    POETRY_VENV=/opt/poetry-venv \
    POETRY_CACHE_DIR=/opt/.cache

# Installer les dépendances système nécessaires pour la compilation
# Inclure les paquets -dev nécessaires pour compiler certaines dépendances Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    # Paquets -dev nécessaires pour la compilation des dépendances Python
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Installer Poetry dans un environnement virtuel isolé
RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION} \
    && ln -s ${POETRY_VENV}/bin/poetry /usr/local/bin/poetry \
    && chmod +x /usr/local/bin/poetry

# Configurer Poetry pour ne pas créer d'environnement virtuel
# (on utilisera l'environnement système)
RUN poetry config virtualenvs.create false

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de dépendances
COPY pyproject.toml poetry.lock* ./

# Installer les dépendances (sans les dépendances de dev)
RUN poetry install --only main --no-interaction --no-ansi

# Stage 2: Runtime - Image finale optimisée
# Version pinée pour garantir la reproductibilité des builds
FROM python:3.11.9-slim as runtime

# Installer uniquement les dépendances runtime nécessaires (sans les paquets -dev)
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Dépendances pour PyMuPDF (librairie d'exécution uniquement, pas -dev)
    libmupdf \
    # Dépendances pour pdf2image (Poppler)
    poppler-utils \
    # Dépendances pour OpenCV (si nécessaire)
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Créer un utilisateur non-root pour la sécurité
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app /app/data && \
    chown -R appuser:appuser /app

# Copier l'environnement Python depuis le builder
# Poetry installe dans l'environnement système Python
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copier uniquement le code source nécessaire (pas les tests, docs, etc.)
WORKDIR /app

# Copier le code source
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser config.yaml ./
COPY --chown=appuser:appuser pyproject.toml ./

# Créer les répertoires de données nécessaires
RUN mkdir -p data/temp_storage data/input data/output data/processed models && \
    chown -R appuser:appuser data models

# Passer à l'utilisateur non-root
USER appuser

# Variables d'environnement par défaut
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8000

# Exposer le port de l'API
EXPOSE 8000

# Commande par défaut (peut être surchargée dans docker-compose)
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

