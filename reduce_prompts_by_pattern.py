import pandas as pd
import textstat

# === CARICAMENTO ===
df = pd.read_csv("prompt_fairness_clean_FINAL.csv")

MAX_PER_PATTERN = 100

# === CALCOLO DELLA SEMPLICITÀ DEL LINGUAGGIO ===
# Più alto è il FRE → più semplice è il prompt
def simplicity_score(prompt):
    prompt = str(prompt)
    return textstat.flesch_reading_ease(prompt)

df["simplicity"] = df["prompt"].apply(simplicity_score)

# === ORDINA PER (pattern, simplicity DESC, prompt ASC) ===
# DESC perché valori maggiori = più semplici
df_sorted = df.sort_values(
    by=["pattern", "simplicity", "prompt"],
    ascending=[True, False, True]
).reset_index(drop=True)

# === PRENDI SOLO I PRIMI 100 PER PATTERN ===
df_reduced = (
    df_sorted.groupby("pattern", group_keys=False)
    .apply(lambda x: x.head(MAX_PER_PATTERN))
    .reset_index(drop=True)
)

# === RIMUOVO COLONNA DI SUPPORTO ===
df_reduced = df_reduced.drop(columns=["simplicity"])

# === ORDINA PER PATTERN (se serve) ===
pattern_order = ["Zero-shot", "Instructional", "Few-shot", "Personas", "Chain-of-Thought"]
df_reduced["pattern"] = pd.Categorical(df_reduced["pattern"], categories=pattern_order, ordered=True)
df_reduced = df_reduced.sort_values("pattern").reset_index(drop=True)

# === SALVA ===
df_reduced.to_csv("prompt_fairness_reduced_FOR_GENERATION_SIMPLE.csv", index=False)

print("Nuova distribuzione:")
print(df_reduced["pattern"].value_counts())
print("\nDataset salvato come: prompt_fairness_reduced_FOR_GENERATION_SIMPLE.csv")
