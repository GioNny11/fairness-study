import pandas as pd
from scipy.stats import kruskal

# Carica il CSV con i risultati di sentiment
df = pd.read_csv("sentiment_results.csv")

# Estrai i sentiment score per ciascun pattern
zero = df[df["pattern"] == "Zero-shot"]["Sentiment Score"]
instr = df[df["pattern"] == "Instructional"]["Sentiment Score"]
few = df[df["pattern"] == "Few-shot"]["Sentiment Score"]
persona = df[df["pattern"] == "Personas"]["Sentiment Score"]
cot = df[df["pattern"] == "Chain-of-Thought"]["Sentiment Score"]

# Kruskal-Wallis test
stat, p_value = kruskal(zero, instr, few, persona, cot)

print("Kruskal-Wallis p-value:", p_value)

# Statistiche descrittive (direzioni)
summary = df.groupby("pattern")["Sentiment Score"].agg(["mean", "median", "std"])
print(summary)
