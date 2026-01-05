from google import genai

# 1. Create the client (it will auto-detect 'GEMINI_API_KEY' from your environment)
client = genai.Client(api_key="YOUR_API_KEY")

# 2. Generate content (Note: 'models.generate_content' is the new syntax)
response = client.models.generate_content(
    model="gemini-2.0-flash", contents="Tell me a haiku about coding."
)

print(response.text)
