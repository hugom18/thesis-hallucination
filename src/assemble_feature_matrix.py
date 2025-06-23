# assemble_feature_matrix.py

import json
import pandas as pd

# 1) Load enriched features (bert_f1, cos_sim, prompt-level, labels)
with open("data/outputs_enriched.json", "r") as f:
    enriched = pd.DataFrame(json.load(f))

# 2) Load expanded signals (nli_contradiction only)
with open("data/outputs_expanded.json", "r") as f:
    expanded = pd.DataFrame(json.load(f))
# Keep only the new NLI field
expanded = expanded[["prompt_id", "model", "nli_contradiction"]]

# 3) Merge on prompt_id & model
df = pd.merge(enriched, expanded, on=["prompt_id", "model"], how="left")

# 4) Define final feature set
features = [
    "prompt_id", "model", "hallucination",
    "bert_f1", "cos_sim", "nli_contradiction",
    "token_len", "entity_count", "question_type", "ambiguity_flag"
]
df_final = df[features]

# 5) Save to CSV
df_final.to_csv("data/feature_matrix.csv", index=False)
print("✅ Feature matrix saved to data/feature_matrix.csv")
