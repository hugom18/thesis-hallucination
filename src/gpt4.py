import openai

# Set your API key directly (or use dotenv for safety)
openai.api_key = "my key"  # Replace with your actual OpenAI API key

client = openai.OpenAI()

# Call GPT-4
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "Who was the president of Brazil in 1990?"}
    ],
    temperature=0.7,
    max_tokens=300
)

# Print result
print("GPT-4 Response:\n", response.choices[0].message.content)
