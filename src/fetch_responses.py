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
from google import genai
from google.genai.types import HttpOptions
import requests
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from cohere import Client as CohereClient

# 1. Config paths ───────────────────────────────────────────── Config paths ─────────────────────────────────────────────
CSV_IN   = "data/raw/prompts.csv"
JSON_OUT = "data/intermediate/outputs.json"

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
    "factual_accuracy":      1,
    "completeness":          1,
    "support_verifiability": 1,
    "reasoning":             1,
    "tone_hedging":          1,
    "hallucination":         0,
    "label_notes":           "",
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

# 4. Models to test ─────────────────────────────────────────────
MODELS = [
    "gpt-3.5-turbo", 
    "gpt-4o",
    "gemini-2.5-pro",
    "claude-sonnet-4-0",
    "cohere-command"
]

# 5. Initialize clients ───────────────────────────────────────
load_dotenv()
# OpenAI
openai_client = OpenAI()
# Google Gemini (requires GOOGLE_API_KEY)
genai_client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY"),
    http_options=HttpOptions(api_version="v1beta")
)
# Anthropic Claude (requires ANTHROPIC_API_KEY)
anthropic_client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)
# Cohere
cohere_client = CohereClient(os.getenv("COHERE_API_KEY"))

# 6. Helpers ────────────────────────────────────────────────── Helpers ──────────────────────────────────────────────────
def load_prompts(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def call_openai(prompt_text: str, model_name: str) -> str:
    resp = openai_client.chat.completions.create(
        model       = model_name,
        messages    = [{"role": "user", "content": prompt_text}],
        temperature = 0,
        max_tokens  = 300
    )
    return resp.choices[0].message.content.strip()


def call_gemini(prompt_text: str, model_name: str) -> str:
    resp = genai_client.models.generate_content(
        model= model_name,
        contents=prompt_text
    )
    return resp.text.strip()


def call_claude(prompt_text: str, model_name: str) -> str:
    """
    Call Claude via HTTP to the Anthropic Messages API endpoint.
    """
    url = "https://api.anthropic.com/v1/chat/completions"
    headers = {
        "x-api-key": os.getenv("ANTHROPIC_API_KEY"),
        "Content-Type": "application/json"
    }
    body = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens_to_sample": 300,
        "temperature": 0
    }
    response = requests.post(url, headers=headers, json=body)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def call_cohere(prompt_text: str, model_name: str) -> str:
    # model_name like 'cohere-command'
    co_model = model_name.replace("cohere-", "")
    response = cohere_client.generate(
        model=co_model,
        prompt=prompt_text,
        max_tokens=300,
        temperature=0
    )
    return response.generations[0].text.strip()


# 7. Main ─────────────────────────────────────────────────────
def main():
    prompts = load_prompts(CSV_IN)
    results = []

    for row in prompts:
        prompt_text   = row["prompt"]
        reference_txt = row.get("reference_text", "")

        for model_name in MODELS:
            if model_name.startswith("gemini"):
                output = call_gemini(prompt_text, model_name)
            elif model_name.startswith("claude"):
                output = call_claude(prompt_text, model_name)
            elif model_name.startswith("cohere"):
                output = call_cohere(prompt_text, model_name)
            else:
                output = call_openai(prompt_text, model_name)

            record = {
                "prompt_id":      row.get("prompt_id",""),
                "category":       row.get("category",""),
                "prompt":         prompt_text,
                "model":          model_name,
                "response":       output,
                "reference_text": reference_txt
            }
            record.update(EXTRA_COLS)

            ordered = OrderedDict((k, record[k]) for k in OUTPUT_ORDER)
            results.append(ordered)
            print(f"{record['prompt_id']} – {model_name}: done")


    pathlib.Path(JSON_OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved {len(results)} records to {JSON_OUT}")

if __name__ == "__main__":
    main()