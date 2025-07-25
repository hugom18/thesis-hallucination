#compute_factuality_signals.py
# ─────────────────────────────────────────────────────────────
# compute_factuality_signals.py – compute NLI contradiction scores for LLM responses
# ─────────────────────────────────────────────────────────────
import os
import json
from dotenv import load_dotenv
import openai
import torch

# Disable tokenizers parallelism & limit threads
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- LOAD ENV VARS ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- CONFIGURATION ---
NLI_MODEL = "roberta-large-mnli"

# Load NLI model/tokenizer (public MNLI)
tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
model     = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).eval()

def compute_nli_contradiction(response: str, reference: str) -> float:
    """Return the contradiction probability from a RoBERTa-MNLI model."""
    inputs = tokenizer(response, reference, return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        probs = model(**inputs).logits.softmax(dim=-1)[0]
    return probs[2].item()  # index 2 = CONTRADICTION

# Load enriched data from intermediate folder
with open("data/intermediate/outputs_enriched.json", "r") as f:
    data = json.load(f)

# Compute only NLI contradiction
for entry in data:
    resp = entry["response"]
    ref  = entry["reference_text"]
    entry["nli_contradiction"] = compute_nli_contradiction(resp, ref)

# Ensure intermediate folder exists
os.makedirs("data/intermediate", exist_ok=True)

# Save expanded output to intermediate folder
with open("data/intermediate/outputs_expanded.json", "w") as f:
    json.dump(data, f, indent=2)

print("data/intermediate/outputs_expanded.json created with nli_contradiction only.")
