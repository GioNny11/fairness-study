import json
import pandas as pd
import re

# ============================
# 1) CARICAMENTO DATASET
# ============================

with open("final_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)


# ============================
# 2) FILTRO ANTI-CODICE
# ============================

code_terms = [
    "python", "java", "javascript", "c++", "c#", "sql", "html", "css",
    "print(", "function", "def ", "class ", "return", "loop",
    "variable", "array", "string", "compile", "debug", "run", "execute",
    "system.out", "algorithm", "source code"
]

code_snippet_patterns = [
    r"```.*?```",
    r"\bclass\s+\w+",
    r"\bdef\s+\w+",
    r"\bfunction\b",
    r"{.*}",
    r"\bvar\b|\blet\b|\bconst\b"
]


def requires_code(prompt: str) -> bool:
    """Rileva prompt che CONTENGONO codice."""
    p = prompt.lower()

    # keyword di linguaggi
    if any(term in p for term in code_terms):
        return True

    # regex blocchi o strutture tipiche
    for pat in code_snippet_patterns:
        if re.search(pat, prompt, flags=re.DOTALL):
            return True

    return False


def is_code_like_prompt(prompt: str) -> bool:
    """
    Rileva prompt che SONO codice anche senza keyword:
    - molte righe indentate
    - simboli tecnici in alta percentuale
    - nessuna parola discorsiva
    """
    p = prompt.strip()
    lines = p.split("\n")

    # 1) molte righe indentate → quasi sicuramente codice
    indent_count = sum(1 for l in lines if l.startswith((" ", "\t")))
    if indent_count >= 2:
        return True

    # 2) alta densità di simboli tecnici
    symbol_set = set("=(){}[]<>:+-*/%.,")
    symbol_ratio = sum(ch in symbol_set for ch in p) / max(len(p), 1)
    if symbol_ratio > 0.25:
        return True

    # 3) poche parole naturali → probabile codice
    common_words = [
        "the", "this", "that", "how", "what", "why", "please", "explain",
        "describe", "example", "can", "should", "could"
    ]

    if not any(w in p.lower() for w in common_words):
        if symbol_ratio > 0.15:
            return True

    return False


# ============================
# 3) GESTIONE FEW-SHOT
# ============================

false_example_context = [
    "example code", "example output", "example files", "example data",
    "example script", "example usage", "example function", "example project"
]

def is_false_example_context(prompt):
    p = prompt.lower()
    return any(x in p for x in false_example_context)

valid_example_words = [
    "example", "examples", "sample", "samples",
    "esempio", "esempi",
    "ejemplo", "ejemplos",
    "exemple", "exemples",
    "beispiel", "beispiele"
]

def refers_to_examples(prompt):
    p = prompt.lower()
    return any(w in p for w in valid_example_words)

non_fewshot_phrases = [
    "for example", "for instance",
    "per esempio", "ad esempio",
    "por ejemplo", "par exemple",
    "zum beispiel"
]

def contains_discursive_example(prompt):
    return any(x in prompt.lower() for x in non_fewshot_phrases)


# ============================
# 4) LISTE DI KEYWORDS PER I PATTERN
# ============================

instructional_keywords = [
    "explain", "describe", "list", "define", "summarize", "outline",
    "spiega", "descrivi", "elenca", "definisci", "riassumi",
    "explica", "describe", "define",
    "explique", "décris",
    "erkläre", "beschreibe"
]

personas_keywords = [
    "you are a", "you are an",
    "act as a", "act as an",
    "pretend you are",
    "your role is",
    "sei un", "sei una", "nel ruolo di",
    "eres un", "eres una",
    "tu es un", "tu es une",
    "du bist ein", "du bist eine"
]

cot_keywords = [
    "think step by step", "step by step",
    "explain your reasoning", "show your reasoning",
    "let's think step by step",
    "ragiona passo passo", "mostra i passaggi",
    "piensa paso a paso",
    "réfléchis étape par étape",
    "denke schritt für schritt"
]

few_shot_keywords = [
    "another example", "give me another example", "one more example",
    "more examples", "show me another", "next example",
    "another one", "another sample", "more samples",
    "previous example", "in the first example", "in the 1st example",
    "update your examples", "follow-up example",
    "altro esempio", "un altro esempio",
    "otro ejemplo", "más ejemplos",
    "un autre exemple", "plus d'exemples",
    "ein weiteres beispiel", "mehr beispiele"
]


# ============================
# 5) FUNZIONE DI DETECTION FINALE
# ============================

def detect_pattern(prompt):
    p = prompt.lower()

    # — FILTRO ANTI-CODICE MIGLIORATO —
    if requires_code(prompt) or is_code_like_prompt(prompt):
        return None

    # Chain-of-Thought
    if any(k in p for k in cot_keywords):
        return "Chain-of-Thought"

    # Few-shot
    if any(k in p for k in few_shot_keywords):
        if (refers_to_examples(prompt)
            and not contains_discursive_example(prompt)
            and not is_false_example_context(prompt)):
            return "Few-shot"

    # Personas
    if any(k in p for k in personas_keywords):
        return "Personas"

    # Instructional
    if any(k in p for k in instructional_keywords):
        return "Instructional"

    # Default → Zero-shot
    return "Zero-shot"


# ============================
# 6) ESTRAZIONE PROMPT CLASSIFICATI
# ============================

matched_prompts = []

for item in dataset:
    for sharing in item.get("ChatgptSharing", []):
        for conv in sharing.get("Conversations", []):
            prompt = conv.get("Prompt", "").strip()

            if len(prompt) < 10:
                continue

            pattern = detect_pattern(prompt)

            if pattern:  # scarta prompt di solo codice
                matched_prompts.append({
                    "prompt": prompt,
                    "pattern": pattern
                })

df = pd.DataFrame(matched_prompts).drop_duplicates(subset=["prompt"])
df.to_csv("prompt_fairness_full.csv", index=False, encoding="utf-8")

print("\n=== STATISTICHE FINALI ===")
print("Prompt totali:", len(df))
print(df["pattern"].value_counts())



