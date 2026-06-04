# Dockerfile pour pymusouf
FROM python:3.11-slim

WORKDIR /pymusouf

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers du package
COPY . /pymusouf

# Installer le package pymusouf et ses dépendances
RUN pip install --no-cache-dir -e .

# Rendre la commande "py" disponible dans le conteneur
RUN ln -s /usr/local/bin/python /usr/local/bin/py

# Créer les répertoires de travail
RUN mkdir -p /data /struct /output

# Exposer le répertoire de travail
VOLUME ["/data", "/struct", "/output"]

# Répertoire par défaut pour les scripts
WORKDIR /pymusouf

# Commande par défaut : bash interactif
CMD ["/bin/bash"]
