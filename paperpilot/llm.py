
import os
from dotenv import load_dotenv
from google import genai

from paperpilot.search import search_paper

# Load the API key from the root .env file
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found. Check your .env file."
    )

client = genai.Client(api_key=api_key)


def answer_question(question):
    # Find relevant passages from the research paper
    results = search_paper(question, top_k=3)

    if not results:
        return "No relevant passages were found."

    # Build context with PDF page numbers
    context = "\n\n".join(
        f"[PDF page {item['page']}]\n{item['text']}"
        for item in results
    )

    prompt = f"""
You are PaperPilot, a research paper assistant.

Answer using ONLY the context below.
Do not invent facts.
Cite supporting pages using [PDF page N].
If the context is insufficient, say so.

CONTEXT:
{context}

QUESTION:
{question}

Give a clear, concise answer with page citations.
"""

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt
    )

    return response.text or "The model returned no text."
