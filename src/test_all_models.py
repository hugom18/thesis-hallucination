import os
from dotenv import load_dotenv
from transformers import pipeline
from openai import OpenAI

# Silence tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load keys
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
hf_key = os.getenv("HUGGINGFACE_API_KEY")

# Prompt
prompt = "What is the capital of France and why is it important in European history?"

# Hugging Face (gpt2)
pipe = pipeline("text-generation", model="gpt2", token=hf_key)
hf_response = pipe(prompt, max_new_tokens=50)[0]['generated_text']

# OpenAI (GPT-3.5)
client = OpenAI(api_key=openai_key)
openai_response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}]
).choices[0].message.content

# Results
print("\n🔵 Hugging Face:\n", hf_response)
print("\n🟢 OpenAI:\n", openai_response)
