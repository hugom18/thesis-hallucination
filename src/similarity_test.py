import json
import spacy
import re
from transformers import AutoTokenizer
from bert_score import score
from sentence_transformers import SentenceTransformer, util

# Load spaCy, tokenizers & models
nlp       = spacy.load("en_core_web_sm")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
sbert     = SentenceTransformer('all-MiniLM-L6-v2')

# Load data
with open("data/outputs.json", "r") as f:
    data = json.load(f)

# Prepare texts
references = [e["reference_text"] for e in data]
responses  = [e["response"]       for e in data]

# 1) BERTScore F1
_, _, F1 = score(responses, references, lang="en", verbose=True)

# 2) SBERT cosine similarity
refs_emb = sbert.encode(references, convert_to_tensor=True)
res_emb  = sbert.encode(responses,  convert_to_tensor=True)
cos_scores = util.cos_sim(res_emb, refs_emb).diag().tolist()

# Enrich entries
for i, entry in enumerate(data):
    prompt = entry["prompt"]
    doc    = nlp(prompt)

    entry["bert_f1"]      = round(F1[i].item(), 4)
    entry["cos_sim"]      = round(cos_scores[i],   4)
    entry["token_len"]    = len(tokenizer.tokenize(prompt))
    entry["entity_count"] = len(doc.ents)

    m = re.match(r"\s*(who|what|when|where|why|how)\b", prompt.lower())
    entry["question_type"]  = m.group(1) if m else None
    entry["ambiguity_flag"] = None

# Save enriched output
with open("data/outputs_enriched.json", "w") as f:
    json.dump(data, f, indent=2)

print("outputs_enriched.json updated with bert_f1, cos_sim & prompt features.")
