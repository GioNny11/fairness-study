# Fairness Study Scripts

Questo repository contiene gli script utilizzati per lo studio sulla **fairness** nei modelli linguistici di grandi dimensioni (LLMs). Il progetto analizza la tossicità, il sentiment e altre metriche di fairness delle risposte generate da modelli LLM, utilizzando un dataset di prompt e pattern.

## Contenuti

- **`final_dataset.json`**: Dataset utilizzato per l'analisi.
- **`fairness_pattern_scan.py`**: Script per identificare i pattern nei prompt e generare `prompt_fairness_full.csv`.
- **`clean_fairness_dataset.py`**: Pulisce il dataset rimuovendo richieste di codice o snippet (che avrebbero reso più difficile l'analisi della fairness) e genera `prompt_fairness_clean_FINAL.csv`.
- **`prompt_fairness_sample_FINAL.csv`**: Un campione casuale per la verifica manuale generato sempre all'interno di **`clean_fairness_dataset.py`**.
- **`reduce_prompts_by_pattern.py`**: Riduce il numero di occorrenze per pattern grazie alla FRE (max 100 dove le occorrenze superano questa soglia) e genera `prompt_fairness_reduced_FOR_GENERATION_SIMPLE.csv`.
- **`generate_responses_LMstudio.py`**: Genera risposte tramite il modello **Meta-LLaMA-3.1-8B** usando **LM Studio** e salva i risultati in `fairness_responses_LMstudio.csv`.
- **`toxicity_evaluation.py`**: Analizza la tossicità, con il modello **unitary/toxic-bert**, delle risposte generate. I risultati sono mostrati in **`toxicity_results.csv`**.
- **`toxicity_results.csv`**: Contiene i risultati della tossicità tramite le colonne Toxicity Label (Toxic o No Toxic) e Toxic Score (tra 0 e 1).
- **`toxicity_stats.py`**: Calcola le statistiche sulla tossicità. I risultati sono mostrati nel terminale.
- - **Statistiche** mostrate nel terminale:
    - Kruskal-Wallis p-value
    - Media
    - Mediana
    - Deviazione standard
- **`sentiment_analysis.py`**: Analizza il sentiment delle risposte. I risultati sono mostrati in **`sentiment_results.csv`**.
- **`sentiment_results.csv`**: Mostra i risultati del sentiment tramite le colonne Sentiment Label (POSITIVE o NEGATIVE) e Sentiment Score (da 0 a 1).
- **`sentiment_stats.py`**: Calcola le statistiche sul sentiment. I risultati sono mostrati nel terminale.
- - **Statistiche** mostrate nel terminale:
    - Kruskal-Wallis p-value
    - Media
    - Mediana
    - Deviazione standard
