import pandas as pd
from transformers import pipeline

# Forza l'uso della CPU (evita rallentamenti con MPS su Mac)
device = -1  # -1 significa CPU, 0 per GPU

# Carica il modello per la classificazione della tossicità
toxicity_pipeline = pipeline("text-classification", model="unitary/toxic-bert", device=device)

# Funzione per calcolare la tossicità in batch
def calculate_toxicity_batch(responses):
    try:
        # Calcola la tossicità per tutte le risposte in batch
        results = toxicity_pipeline(responses)
        return results
    except Exception as e:
        print(f"Errore nel calcolare la tossicità: {e}")
        return []

# === CARICA IL FILE CSV ===
# Percorso del file CSV che hai caricato
input_csv = '/Users/giovannicroce/Desktop/uni/tirocinio/script_fairness/fairness_responses_LMstudio.csv'
df = pd.read_csv(input_csv)

# === Aggiungi colonne per i risultati ===
df['Toxicity Label'] = None
df['Toxicity Score'] = None

# Esegui il calcolo della tossicità in batch
batch_size = 10  # Numero di risposte da analizzare per volta
for i in range(0, len(df), batch_size):
    # Seleziona un batch di risposte
    batch_responses = df['prompt'][i:i+batch_size].tolist()
    
    # Calcola la tossicità per il batch
    results = calculate_toxicity_batch(batch_responses)
    
    # Aggiungi i risultati al DataFrame
    for j, result in enumerate(results):
        df.at[i + j, 'Toxicity Label'] = result['label']
        df.at[i + j, 'Toxicity Score'] = result['score']

# === SCRIVI I RISULTATI IN UN NUOVO FILE CSV ===
output_csv = '/Users/giovannicroce/Desktop/uni/tirocinio/script_fairness/toxicity_results.csv'
df.to_csv(output_csv, index=False)

print(f"Il file con le risposte e la tossicità è stato salvato in {output_csv}")