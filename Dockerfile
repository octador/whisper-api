# whisper-backend/Dockerfile

# Étape 1: Image de base Python
FROM python:3.10-slim

# Étape 2: Répertoire de travail
WORKDIR /app

# Étape 3: Installation des dépendances système (pour faster-whisper)
# faster-whisper a besoin de ffmpeg et libsndfile1
RUN apt-get update && apt-get install -y ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

# Étape 4: Installation des dépendances Python
# Copie le fichier requirements.txt (que tu dois créer dans whisper-backend/)
COPY requirements.txt .
# Installe toutes les dépendances listées dans requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Étape 5: Copier le code de l'application
COPY main.py .

# Étape 6: Pré-télécharger le modèle Whisper
# C'est une excellente idée ! Cela évite de le télécharger à chaque démarrage du conteneur.
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('medium', device='cpu', compute_type='int8')"

# Étape 7: Exposer le port sur lequel l'API écoutera
EXPOSE 8000

# Étape 8: Commande de démarrage du serveur API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]