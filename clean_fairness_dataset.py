import pandas as pd
import re

df = pd.read_csv("prompt_fairness_full.csv")

# === Filtro: richieste di generare codice ===
code_request_terms = [
    "write code", "generate code", "fix this code", "provide code",
    "implement", "write a function", "write a class",
    "convert to code", "debug this", "refactor this", "code only"
]

code_snippet_patterns = [
    r"```.*?```", r"\bclass\s+\w+", r"\bdef\s+\w+",
    r"\bfunction\b", r"\bvar\b|\blet\b|\bconst\b", r"\bimport\b"
]

def asks_for_code(prompt):
    p = str(prompt).lower()
    if any(term in p for term in code_request_terms):
        return True
    for pat in code_snippet_patterns:
        if re.search(pat, prompt, flags=re.DOTALL):
            return True
    return False

# === PULIZIA ===
df_clean = df[~df["prompt"].apply(asks_for_code)]
df_clean = df_clean[df_clean["prompt"].str.split().apply(len) >= 4]
df_clean = df_clean.drop_duplicates(subset=["prompt"])

df_clean.to_csv("prompt_fairness_clean_FINAL.csv", index=False)
print("Pulito:", df_clean["pattern"].value_counts())

# === SAMPLE CASUALE ORDINATO ===
pattern_order = ["Zero-shot", "Instructional", "Few-shot", "Personas", "Chain-of-Thought"]

df_sample = df_clean.groupby("pattern", group_keys=False).apply(
    lambda x: x.sample(n=min(10, len(x)))   # <— NESSUN random_state!
).reset_index(drop=True)

df_sample["pattern"] = pd.Categorical(df_sample["pattern"], categories=pattern_order, ordered=True)
df_sample = df_sample.sort_values("pattern").reset_index(drop=True)

df_sample["Verifica"] = ""

df_sample.to_csv("prompt_fairness_sample_FINAL.csv", index=False)

print("Sample finale:", df_sample["pattern"].value_counts())
