import json
import spacy
import re
import string
from transformers import AutoTokenizer
from bert_score import score
from sentence_transformers import SentenceTransformer, util
from textstat import flesch_reading_ease

# Load spaCy, tokenizers & models
nlp       = spacy.load("en_core_web_sm")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
sbert     = SentenceTransformer('all-MiniLM-L6-v2')

# Load data
with open("data/intermediate/outputs.json", "r") as f:
    data = json.load(f)

# Prepare texts
references = [e.get("reference_text", "") for e in data]
responses  = [e.get("response",       "") for e in data]

# 1) BERTScore F1
_, _, F1 = score(responses, references, lang="en", verbose=False)

# 2) SBERT cosine similarity
refs_emb    = sbert.encode(references, convert_to_tensor=True)
res_emb     = sbert.encode(responses,  convert_to_tensor=True)
cos_scores  = util.cos_sim(res_emb, refs_emb).diag().tolist()

# Enrich entries
for i, entry in enumerate(data):
    prompt = entry.get("prompt", "")
    doc    = nlp(prompt)

    # Similarity metrics
    entry["bert_f1"] = round(F1[i].item(), 4)
    entry["cos_sim"] = round(cos_scores[i],   4)

    # Prompt-level features
    entry["token_len"]     = len(tokenizer.tokenize(prompt))
    entry["char_len"]      = len(prompt)
    entry["entity_count"]  = len(doc.ents)

    # Question type
    m = re.match(r"\s*(who|what|when|where|why|how)\b", prompt.lower())
    entry["question_type"] = m.group(1) if m else None

    # Ambiguity flag: propagate existing label
    entry["ambiguity_flag"] = entry.get("ambiguity_flag")

    # Pronoun ratio: pronouns / alphabetic tokens
    alpha_tokens = [t for t in doc if t.is_alpha]
    pronouns     = [t for t in doc if t.pos_ == 'PRON']
    entry["pronoun_ratio"] = round(len(pronouns) / len(alpha_tokens), 4) if alpha_tokens else 0.0

    # Imperative flag: first token is a base-form verb
    entry["imperative_flag"] = bool(doc[0].tag_ == 'VB') if doc else False

    # Readability score (Flesch Reading Ease)
    entry["readability_score"] = round(flesch_reading_ease(prompt), 2)

    # Punctuation density: punctuation chars / total chars
    puncts = [c for c in prompt if c in string.punctuation]
    entry["punct_density"] = round(len(puncts) / len(prompt), 4) if prompt else 0.0

# Save enriched output
with open("data/intermediate/outputs_enriched.json", "w") as f:
    json.dump(data, f, indent=2)

print("outputs_enriched.json updated with bert_f1, cos_sim & full set of prompt features.")
