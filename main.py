# main.py (version corrigée pour les timestamps mot par mot)

import os
import tempfile
from fastapi import FastAPI, UploadFile, File, Query
from faster_whisper import WhisperModel
import time
import json # <-- AJOUTE CET IMPORT

# --- Configuration ---
MODEL_SIZE = "medium"
COMPUTE_TYPE = "int8"
# --- Fin de la Configuration ---

app = FastAPI()
model = None

@app.on_event("startup")
def load_model():
    global model
    print(f"Chargement du modèle Whisper '{MODEL_SIZE}'...")
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE)
    print("Modèle chargé et prêt à recevoir des requêtes.")

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(..., description="Le fichier audio à transcrire"),
    user_id: str = Query(None, description="ID de l'utilisateur"),
    niveau_de_langue: str = Query(None, description="Niveau de langue de l'utilisateur")
):
    if not model:
        return {"error": "Le modèle n'est pas encore chargé. Veuillez patienter."}

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        print(f"Traitement du fichier : {file.filename} pour l'utilisateur {user_id}")
        start_time = time.time()

        segments, info = model.transcribe(
            tmp_path,
            beam_size=5,
            language="fr",
            word_timestamps=True # <-- TRÈS IMPORTANT : Active les timestamps mot par mot
        )

        full_text = ""
        transcription_segments = []
        word_level_timestamps = [] # <-- NOUVEAU : Pour stocker les timestamps mot par mot

        for segment in segments:
            full_text += segment.text.strip() + " "
            transcription_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
            if segment.words: # Les mots sont disponibles si word_timestamps=True
                for word in segment.words:
                    word_level_timestamps.append({
                        "word": word.word,
                        "start": word.start,
                        "end": word.end
                    })

        duration = time.time() - start_time
        print(f"Transcription terminée en {duration:.2f} secondes.")

        return {
            "user_id": user_id,
            "niveau_de_langue": niveau_de_langue,
            "language": info.language,
            "language_probability": info.language_probability,
            "full_text": full_text.strip(), # <-- AJOUTE LE TEXTE COMPLET
            "transcription_segments": transcription_segments,
            "word_level_timestamps": word_level_timestamps, # <-- AJOUTE LES TIMESTAMPS MOT PAR MOT
            "processing_time_seconds": duration
        }

    except Exception as e:
        return {"error": f"Une erreur est survenue : {str(e)}"}

    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de transcription Whisper (optimisée pour le français). Utilisez le endpoint POST /transcribe."}