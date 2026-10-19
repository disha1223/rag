
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

for model in ["gemini-3.6-flash", "gemini-2.5-flash"]:
    print(f"\nTesting {model}...")

    try:
        response = client.models.generate_content(
            model=model,
            contents="Reply with just: PaperPilot works!"
        )
        print(response.text)
        break

    except Exception as error:
        print(f"Failed: {error}")
