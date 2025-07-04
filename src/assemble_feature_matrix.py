import json
import pandas as pd
import os

# --- Paths ---
enriched_path = "data/intermediate/outputs_enriched.json"
expanded_path = "data/intermediate/outputs_expanded.json"
output_path   = "data/final/feature_matrix.csv"

# 1) Load enriched features (bert_f1, cos_sim, prompt-level, labels)
with open(enriched_path, "r") as f:
    enriched = pd.DataFrame(json.load(f))

# 2) Load expanded signals (nli_contradiction only)
with open(expanded_path, "r") as f:
    expanded = pd.DataFrame(json.load(f))
# expanded may contain extra fields; ensure we merge all and keep them

# 3) Merge enriched and expanded on prompt_id & model
merged = pd.merge(
    enriched,
    expanded,
    on=["prompt_id", "model"],
    how="left",
    suffixes=("", "_exp")
)

# 4) Define final feature set (selected columns)
features = [
    "prompt_id", "model", "hallucination",
    # similarity metrics
    "bert_f1", "cos_sim", "nli_contradiction",
    # prompt-level features
    "token_len", "char_len", "entity_count", "pronoun_ratio", 
    "imperative_flag", "readability_score", "punct_density", 
    # other metadata
    "question_type", "ambiguity_flag"
]
# Ensure all features exist in the merged DataFrame
for feat in features:
    if feat not in merged.columns:
        merged[feat] = None

df_final = merged[features]

# 5) Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Save to CSV
df_final.to_csv(output_path, index=False)
print(f"Feature matrix saved to {output_path}")
