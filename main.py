# main.py (version corrigée pour les constantes globales et la boucle for)

import os
import tempfile
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from faster_whisper import WhisperModel
import time
import json
from typing import Optional, List, Dict, Any
import re
import numpy as np

# --- Configuration ---
MODEL_SIZE = "medium"
COMPUTE_TYPE = "int8"
# --- Fin de la Configuration ---

app = FastAPI()
model = None

# --- Constantes pour l'évaluation (définies globalement) ---
SIMILARITY_THRESHOLD_FACTOR = 0.3 # Max Levenshtein distance comme fraction de la longueur du mot
LOW_CONFIDENCE_PROBABILITY = 0.4 # Probabilité en dessous de laquelle un mot est considéré comme peu confiant
# --- Fin des constantes ---

@app.on_event("startup")
def load_model():
    global model
    print(f"Chargement du modèle Whisper '{MODEL_SIZE}'...")
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE)
    print("Modèle chargé et prêt à recevoir des requêtes.")

# --- Fonctions utilitaires pour l'évaluation ---

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calcule la distance de Levenshtein entre deux chaînes."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def normalize_word_for_comparison(word: str) -> str:
    """Normalise un mot pour la comparaison (minuscules, sans ponctuation, apostrophes standard)."""
    word = word.lower()
    word = word.replace('’', "'") # Normalise les apostrophes (ex: ' vs ’)
    word = re.sub(r'[.,!?;:"]', '', word) # Supprime la ponctuation
    word = re.sub(r'\s+', '', word) # Supprime les espaces internes (ex: "m' appelle" -> "m'appelle")
    return word.strip()

# --- Fin des fonctions utilitaires ---


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(..., description="Le fichier audio à transcrire"),
    user_id: str = Query(None, description="ID de l'utilisateur"),
    language: str = Query("fr", description="Langue de la transcription (ex: fr, en)"),
):
    if not model:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas encore chargé. Veuillez patienter.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        whisper_language = language.lower()
        print(f"Traitement du fichier : {file.filename} pour l'utilisateur {user_id} en langue {whisper_language}")
        start_time = time.time()

        segments, info = model.transcribe(
            tmp_path,
            beam_size=5,
            language=whisper_language,
            word_timestamps=True
        )

        full_text = ""
        transcription_segments = []
        word_level_timestamps = []

        for segment in segments:
            full_text += segment.text.strip() + " "
            transcription_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
            if segment.words:
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
            "language": info.language,
            "language_probability": info.language_probability,
            "full_text": full_text.strip(),
            "transcription_segments": transcription_segments,
            "word_level_timestamps": word_level_timestamps,
            "processing_time_seconds": duration
        }

    except Exception as e:
        print(f"Erreur lors de la transcription: {e}")
        raise HTTPException(status_code=500, detail=f"Une erreur est survenue lors de la transcription: {str(e)}")

    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/evaluate-pronunciation")
async def evaluate_pronunciation(
    file: UploadFile = File(..., description="Le fichier audio de la prononciation de l'utilisateur"),
    expected_prompt: str = Query(..., description="La phrase attendue pour l'exercice"),
    user_id: str = Query(None, description="ID de l'utilisateur"),
    language: str = Query("fr", description="Langue de la transcription (ex: fr, en)"),
):
    if not model:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas encore chargé. Veuillez patienter.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        whisper_language = language.lower()
        print(f"Évaluation prononciation pour user {user_id}, prompt: '{expected_prompt}' en langue {whisper_language}")
        start_time = time.time()

        segments, info = model.transcribe(
            tmp_path,
            beam_size=5,
            language=whisper_language,
            word_timestamps=True
        )

        user_full_text = ""
        user_word_timestamps_data: List[Dict[str, Any]] = []
        for segment in segments:
            user_full_text += segment.text.strip() + " "
            if segment.words:
                for word in segment.words:
                    user_word_timestamps_data.append({
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability
                    })
        user_full_text = user_full_text.strip()

        # --- LOGIQUE D'ÉVALUATION DE PRONONCIATION TRÈS AFFINÉE ---

        # 1. Normalisation des mots attendus
        expected_words_raw = expected_prompt.split()
        expected_words_normalized = [normalize_word_for_comparison(w) for w in expected_words_raw]

        # 2. Normalisation des mots transcrits par l'utilisateur
        user_words_normalized_data = [
            {
                'normalized': normalize_word_for_comparison(w['word']),
                'original': w['word'],
                'probability': w['probability'],
                'start': w['start'],
                'end': w['end']
            }
            for w in user_word_timestamps_data
        ]

        # Gérer le cas où expected_prompt est vide
        if not expected_words_normalized:
            final_score = 0.0
            mismatched_words_details = []
            for i, user_word_obj in enumerate(user_words_normalized_data):
                mismatched_words_details.append({
                    "expected": None,
                    "actual": user_word_obj['original'],
                    "index": i,
                    "reason": "extra",
                    "probability": user_word_obj['probability'],
                    "start": user_word_obj['start'],
                    "end": user_word_obj['end']
                })
        else:
            # --- Algorithme d'alignement basé sur la distance de Levenshtein ---
            n = len(expected_words_normalized)
            m = len(user_words_normalized_data)

            dp = np.zeros((n + 1, m + 1))
            path = np.zeros((n + 1, m + 1), dtype=int) # 0: diag, 1: up (missing), 2: left (extra)

            # Initialisation de la première ligne et colonne
            for i in range(1, n + 1):
                dp[i, 0] = dp[i-1, 0] + 1 # Coût d'un mot manquant
                path[i, 0] = 1 # Vient du haut (mot manquant)
            for j in range(1, m + 1):
                dp[0, j] = dp[0, j-1] + 1 # Coût d'un mot supplémentaire
                path[0, j] = 2 # Vient de la gauche (mot supplémentaire)

            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    expected_word = expected_words_normalized[i-1]
                    user_word_obj = user_words_normalized_data[j-1]
                    user_word = user_word_obj['normalized']

                    # Coût de substitution/match
                    distance = levenshtein_distance(expected_word, user_word)
                    
                    max_len = max(len(expected_word), len(user_word))
                    word_match_cost = distance / max_len if max_len > 0 else 0

                    confidence_penalty = 0
                    if user_word_obj['probability'] < LOW_CONFIDENCE_PROBABILITY:
                        confidence_penalty = 0.2 # Pénalité légère pour faible confiance

                    # Coût total pour un match/substitution
                    cost_diag = dp[i-1, j-1] + word_match_cost + confidence_penalty
                    
                    # Coût pour un mot manquant (expected mais non prononcé)
                    cost_up = dp[i-1, j] + 1 # Pénalité fixe pour un mot manquant
                    
                    # Coût pour un mot supplémentaire (prononcé mais non attendu)
                    cost_left = dp[i, j-1] + 1 # Pénalité fixe pour un mot supplémentaire

                    dp[i, j] = min(cost_diag, cost_up, cost_left)
                    if dp[i, j] == cost_diag:
                        path[i, j] = 0 # Diagonale (match/substitution)
                    elif dp[i, j] == cost_up:
                        path[i, j] = 1 # Haut (mot manquant)
                    else:
                        path[i, j] = 2 # Gauche (mot supplémentaire)

            # Reconstruire l'alignement et calculer le score
            aligned_expected = []
            aligned_user = []
            
            current_i, current_j = n, m
            
            while current_i > 0 or current_j > 0:
                if path[current_i, current_j] == 0: # Match ou substitution
                    expected_word_obj = {
                        "word": expected_words_raw[current_i-1],
                        "normalized": expected_words_normalized[current_i-1]
                    }
                    user_word_obj = user_words_normalized_data[current_j-1]
                    
                    aligned_expected.insert(0, expected_word_obj)
                    aligned_user.insert(0, user_word_obj)
                    
                    current_i -= 1
                    current_j -= 1
                elif path[current_i, current_j] == 1: # Mot manquant (expected mais non prononcé)
                    expected_word_obj = {
                        "word": expected_words_raw[current_i-1],
                        "normalized": expected_words_normalized[current_i-1]
                    }
                    aligned_expected.insert(0, expected_word_obj)
                    aligned_user.insert(0, None) # Pas de mot utilisateur correspondant
                    current_i -= 1
                else: # Mot supplémentaire (prononcé mais non attendu)
                    user_word_obj = user_words_normalized_data[current_j-1]
                    aligned_expected.insert(0, None) # Pas de mot attendu correspondant
                    aligned_user.insert(0, user_word_obj)
                    current_j -= 1
            
            # Calcul du score basé sur l'alignement
            score_sum = 0
            mismatched_words_details = []
            
            for k in range(len(aligned_expected)):
                exp_word = aligned_expected[k]
                usr_word = aligned_user[k]
                
                if exp_word and usr_word: # Match ou substitution
                    distance = levenshtein_distance(exp_word['normalized'], usr_word['normalized'])
                    max_len = max(len(exp_word['normalized']), len(usr_word['normalized']))
                    
                    if distance == 0: # Correspondance exacte
                        score_sum += 1
                        if usr_word['probability'] < LOW_CONFIDENCE_PROBABILITY:
                            mismatched_words_details.append({
                                "expected": exp_word['word'],
                                "actual": usr_word['original'],
                                "index": k,
                                "reason": "low_confidence",
                                "probability": usr_word['probability'],
                                "start": usr_word['start'],
                                "end": usr_word['end']
                            })
                    elif max_len > 0 and distance / max_len <= SIMILARITY_THRESHOLD_FACTOR: # Mal prononcé (similaire)
                        score_sum += (1 - (distance / max_len)) * 0.8 # Crédit partiel basé sur la similarité
                        mismatched_words_details.append({
                            "expected": exp_word['word'],
                            "actual": usr_word['original'],
                            "index": k,
                            "reason": "mispronounced",
                            "distance": distance,
                            "probability": usr_word['probability'],
                            "start": usr_word['start'],
                            "end": usr_word['end']
                        })
                    else: # Substitution complète (très différent)
                        mismatched_words_details.append({
                            "expected": exp_word['word'],
                            "actual": usr_word['original'],
                            "index": k,
                            "reason": "substitution",
                            "distance": distance,
                            "probability": usr_word['probability'],
                            "start": usr_word['start'],
                            "end": usr_word['end']
                        })
                elif exp_word and not usr_word: # Mot manquant
                    mismatched_words_details.append({
                        "expected": exp_word['word'],
                        "actual": None,
                        "index": k,
                        "reason": "missing"
                    })
                elif not exp_word and usr_word: # Mot supplémentaire
                    mismatched_words_details.append({
                        "expected": None,
                        "actual": usr_word['original'],
                        "index": k,
                        "reason": "extra",
                        "probability": usr_word['probability'],
                        "start": usr_word['start'],
                        "end": usr_word['end']
                    })
            
            final_score = score_sum / n
        
        final_score = max(0, min(1, final_score))

        # --- FIN LOGIQUE D'ÉVALUATION TRÈS AFFINÉE ---

        duration = time.time() - start_time
        print(f"Évaluation terminée en {duration:.2f} secondes. Score: {final_score:.2f}")

        return {
            "user_id": user_id,
            "language": info.language,
            "language_probability": info.language_probability,
            "user_transcription": user_full_text,
            "expected_prompt": expected_prompt,
            "score": final_score,
            "errors": mismatched_words_details,
            "word_level_timestamps": user_word_timestamps_data,
            "processing_time_seconds": duration
        }

    except Exception as e:
        print(f"Erreur lors de l'évaluation de prononciation: {e}")
        raise HTTPException(status_code=500, detail=f"Une erreur est survenue lors de l'évaluation: {str(e)}")

    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de transcription Whisper (optimisée pour le français). Utilisez le endpoint POST /transcribe ou POST /evaluate-prononciation."}