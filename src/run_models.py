import os
import csv
import json
from dotenv import load_dotenv
from openai import OpenAI

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
CSV_IN   = "data/10_prompts.csv"   # <- change to your actual file
JSON_OUT = "data/full_output.json"

# Extra columns to add (blank / None until later stages)
EXTRA_COLS = {
    "reference_text": "",
    "bert_f1": None,
    "cos_sim": None,
    "token_len": None,
    "entity_count": None,
    "question_type": "",
    "ambiguity_flag": None,
    "factual_accuracy": None,
    "completeness": None,
    "support_verifiability": None,
    "reasoning": None,
    "tone_hedging": None,
    "hallucination": None,
    "label_notes": ""
}

# ─────────────────────────────────────────────────────────────
# Setup OpenAI client
# ─────────────────────────────────────────────────────────────
load_dotenv()             # expects OPENAI_API_KEY in .env
client = OpenAI()

# ─────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────
def load_prompts(path):
    """Return list of dicts from a CSV file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    with open(path, newline='', encoding="utf-8") as f:
        return list(csv.DictReader(f))

def call_model(prompt, model_name):
    """Query OpenAI chat model (temperature 0) and return string."""
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=300
    )
    return resp.choices[0].message.content.strip()

# ─────────────────────────────────────────────────────────────
# Main routine
# ─────────────────────────────────────────────────────────────
def main():
    prompts = load_prompts(CSV_IN)
    results = []

    for row in prompts:
        for model in ["gpt-3.5-turbo", "gpt-4o"]:
            output = call_model(row["prompt"], model)

            # Start with existing CSV columns
            record = {
                "prompt_id": row.get("prompt_id", ""),
                "category":  row.get("category", ""),
                "prompt":    row.get("prompt", ""),
                "model":     model,
                "response":  output,
            }

            # Add any extra columns not present yet
            for col, default in EXTRA_COLS.items():
                record.setdefault(col, default)

            results.append(record)

    os.makedirs(os.path.dirname(JSON_OUT), exist_ok=True)
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
