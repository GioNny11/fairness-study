import pandas as pd
from transformers import pipeline

# Forza l'uso della CPU (evita rallentamenti con MPS su Mac)
device = -1  # -1 significa CPU, 0 per GPU

# Carica il modello per la sentiment analysis
sentiment_pipeline = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=device
)

# Funzione per calcolare il sentiment in batch
def calculate_sentiment_batch(responses):
    try:
        # Calcola il sentiment per tutte le risposte in batch
        results = sentiment_pipeline(responses)
        return results
    except Exception as e:
        print(f"Errore nel calcolare il sentiment: {e}")
        return []

# === CARICA IL FILE CSV ===
input_csv = '/Users/giovannicroce/Desktop/uni/tirocinio/script_fairness/fairness_responses_LMstudio.csv'
df = pd.read_csv(input_csv)

# === Aggiungi colonne per i risultati ===
df['Sentiment Label'] = None
df['Sentiment Score'] = None

# Esegui il calcolo del sentiment in batch
batch_size = 10  # Numero di risposte da analizzare per volta
for i in range(0, len(df), batch_size):
    # Seleziona un batch di risposte
    batch_responses = df['prompt'][i:i+batch_size].tolist()
    
    # Calcola il sentiment per il batch
    results = calculate_sentiment_batch(batch_responses)
    
    # Aggiungi i risultati al DataFrame
    for j, result in enumerate(results):
        df.at[i + j, 'Sentiment Label'] = result['label']
        df.at[i + j, 'Sentiment Score'] = result['score']

# === SCRIVI I RISULTATI IN UN NUOVO FILE CSV ===
output_csv = '/Users/giovannicroce/Desktop/uni/tirocinio/script_fairness/sentiment_results.csv'
df.to_csv(output_csv, index=False)

print(f"Il file con le risposte e il sentiment è stato salvato in {output_csv}")
