import pandas as pd
import requests
import time

# === CONFIGURAZIONE ===
MODEL_NAME = "meta-llama-3.1-8b-instruct"
API_URL = "http://localhost:1234/v1/chat/completions"  # Endpoint API di LM Studio

INPUT_FILE = "prompt_fairness_reduced_FOR_GENERATION_SIMPLE.csv"  # Percorso del file CSV con i prompt
OUTPUT_FILE = "fairness_responses_LMstudio.csv"  # File per salvare le risposte generate

# === CARICA I PROMPT ===
df = pd.read_csv(INPUT_FILE)

# Aggiungi una colonna per le risposte, se non presente
if "response" not in df.columns:
    df["response"] = None

# Intestazioni per la richiesta API
headers = {"Content-Type": "application/json"}

# === LOOP SUI PROMPT ===
for i, row in df.iterrows():
    prompt_text = str(row["prompt"])

    # Se già esiste una risposta, salta
    if pd.notna(row["response"]) and row["response"] != "":
        print(f"[{i+1}/{len(df)}] Già presente, salto.")
        continue

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.0,  # Risposte più stabili
        "max_tokens": 512    # Limite per risposte più lunghe
    }

    try:
        # Invia la richiesta al server locale LM Studio
        response = requests.post(API_URL, json=payload, headers=headers)
        response.raise_for_status()  # Solleva un'eccezione se status != 200

        # Estrai la risposta dal modello
        data = response.json()
        answer = data["choices"][0]["message"]["content"]

        # Salva la risposta nel dataframe
        df.at[i, "response"] = answer

        print(f"[{i+1}/{len(df)}] Risposta generata.")

        # Pausa per ridurre il carico del sistema
        time.sleep(1)  # Pausa di 1 secondo tra ogni richiesta

    except Exception as e:
        print(f"[{i+1}/{len(df)}] ERRORE: {e}")
        df.at[i, "response"] = "ERROR"
        continue

# === SALVA LE RISPOSTE ===
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Risposte generate e salvate in: {OUTPUT_FILE}")
