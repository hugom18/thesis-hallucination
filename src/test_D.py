import json
import statistics

# Path to your expanded outputs JSON
INPUT_PATH = "data/intermediate/test_outputs_expanded.json"

# Load data
with open(INPUT_PATH, "r") as f:
    records = json.load(f)

# Filter Category D entries
cat_d = [r for r in records]

# Extract metrics
bert_f1_vals    = [r["bert_f1"] for r in cat_d]
cos_sim_vals    = [r["cos_sim"] for r in cat_d]
nli_contra_vals = [r["nli_contradiction"] for r in cat_d]

# Compute and print averages
print(f"Category D count: {len(cat_d)}")
print(f"Avg BERT-F1:            {statistics.mean(bert_f1_vals):.4f}")
print(f"Avg SBERT cosine:       {statistics.mean(cos_sim_vals):.4f}")
print(f"Avg NLI contradiction: {statistics.mean(nli_contra_vals):.4f}")


