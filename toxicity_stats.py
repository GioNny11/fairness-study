import pandas as pd
from scipy.stats import kruskal

# Carica il CSV con i risultati di toxicity
df = pd.read_csv("toxicity_results.csv")

# Estrai la Toxicity Probability per ciascun pattern
zero = df[df["pattern"] == "Zero-shot"]["Toxicity Score"]
instr = df[df["pattern"] == "Instructional"]["Toxicity Score"]
few = df[df["pattern"] == "Few-shot"]["Toxicity Score"]
persona = df[df["pattern"] == "Personas"]["Toxicity Score"]
cot = df[df["pattern"] == "Chain-of-Thought"]["Toxicity Score"]

# Kruskal-Wallis test
stat, p_value = kruskal(zero, instr, few, persona, cot)

print("Kruskal-Wallis p-value:", p_value)

# Statistiche descrittive (direzioni)
summary = df.groupby("pattern")["Toxicity Score"].agg(["mean", "median", "std"])
print(summary)
