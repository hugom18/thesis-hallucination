import google.generativeai as genai

# Replace with your real key
genai.configure(api_key="AIzaSyDQgJ_lVUgYuXKsKw2FzJ0ijrk_q3alWgw")

# Check available models
print("Available models:")
for model in genai.list_models():
    print("-", model.name)

# Try generating content from the correct model
model = genai.GenerativeModel(model_name="models/gemini-pro")

response = model.generate_content("Who was the president of Brazil in 1990?")
print("\nGemini Response:\n", response.text)


