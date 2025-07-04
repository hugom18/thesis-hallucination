# ─────────────────────────────────────────────────────────────
# run_models.py – fetch LLM responses for test10_prompts.csv 
# ─────────────────────────────────────────────────────────────
import os
import csv
import json
import pathlib
from collections import OrderedDict
from dotenv import load_dotenv
from openai import OpenAI

# 1. Config paths ─────────────────────────────────────────────
CSV_IN   = "data/raw/10_prompts.csv"
JSON_OUT = "data/test/1_outputs.json"

# 2. Desired JSON field order ─────────────────────────────────
OUTPUT_ORDER = [
    "prompt_id", "category", "prompt", "model", "response",
    "factual_accuracy", "completeness", "support_verifiability",
    "reasoning", "tone_hedging", "hallucination", "label_notes",
    "reference_text", "bert_f1", "cos_sim", "token_len", "char_len",
    "entity_count", "question_type", "ambiguity_flag",
    "pronoun_ratio", "imperative_flag", "readability_score", "punct_density"
]


# 3. Default values for all eval fields ──────────────────────
EXTRA_COLS = {
    # your six manual‐check columns come prefilled:
    "factual_accuracy":      1,
    "completeness":          1,
    "support_verifiability": 1,
    "reasoning":             1,
    "tone_hedging":          1,
    "hallucination":         0,
    "label_notes":           "",

    # the rest remain blank/null
    "bert_f1":               None,
    "cos_sim":               None,
    "token_len":             None,
    "char_len":              None,
    "entity_count":          None,
    "question_type":         "",
    "ambiguity_flag":        None,
    "pronoun_ratio":         None,
    "imperative_flag":       None,
    "readability_score":     None,
    "punct_density":         None
}

# 4. Models to test ───────────────────────────────────────────
MODELS = ["gpt-3.5-turbo", "gpt-4o"]

# 5. OpenAI client ────────────────────────────────────────────
load_dotenv()  # expects OPENAI_API_KEY in .env
client = OpenAI()

# 6. Helpers ──────────────────────────────────────────────────
def load_prompts(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def call_model(prompt_text: str, model_name: str) -> str:
    resp = client.chat.completions.create(
        model       = model_name,
        messages    = [{"role": "user", "content": prompt_text}],
        temperature = 0,
        max_tokens  = 300
    )
    return resp.choices[0].message.content.strip()

# 7. Main ─────────────────────────────────────────────────────
def main():
    prompts = load_prompts(CSV_IN)
    results = []

    for row in prompts:
        prompt_text   = row["prompt"]
        reference_txt = row.get("reference_text", "")

        for model_name in MODELS:
            output = call_model(prompt_text, model_name)

            # build record with reference_text + prefilled evals
            record = {
                "prompt_id":      row.get("prompt_id",""),
                "category":       row.get("category",""),
                "prompt":         prompt_text,
                "model":          model_name,
                "response":       output,
                "reference_text": reference_txt
            }
            record.update(EXTRA_COLS)

            # reorder to your schema
            ordered = OrderedDict((k, record[k]) for k in OUTPUT_ORDER)
            results.append(ordered)
            print(f"{record['prompt_id']} – {model_name}: done")

    pathlib.Path(JSON_OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved {len(results)} records to {JSON_OUT}")

if __name__ == "__main__":
    main()