
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq

from paperpilot.embeddings import create_embeddings
from paperpilot.database import collection

load_dotenv()

groq_api_key = os.getenv("GEMINI_API_KEY")

if not groq_api_key:
    raise RuntimeError("GEMINI_API_KEY is missing from .env")

client = Groq(api_key=groq_api_key)

app = FastAPI(
    title="PaperPilot API",
    description="Ask questions about indexed research papers",
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str
    paper_id: str


@app.get("/")
def home():
    return {
        "message": "PaperPilot API is running!"
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question.strip()
    paper_id = request.paper_id.strip()

    if not question or not paper_id:
        raise HTTPException(
            status_code=400,
            detail="Question and paper_id are required.",
        )

    try:
        # 1. Convert the question into an embedding.
        embedding = create_embeddings([question])[0]

        # 2. Retrieve relevant chunks from the selected paper.
        results = collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=8,
            where={"paper_id": paper_id},
            include=["documents", "metadatas"],
        )

        documents = results.get("documents") or [[]]
        metadatas = results.get("metadatas") or [[]]

        chunks = documents[0] if documents else []
        metadata_list = metadatas[0] if metadatas else []

        if not chunks:
            raise HTTPException(
                status_code=404,
                detail=(
                    "No indexed chunks found for this paper. "
                    "Check the paper_id and index the PDF first."
                ),
            )

        # 3. Build context with page numbers.
        context_parts = []
        sources = []

        for text, metadata in zip(chunks, metadata_list):
            page = metadata.get("page", "?")
            context_parts.append(f"[Page {page}]\n{text}")
            sources.append({
                "page": page,
                "text": text
            })

        context = "\n\n".join(context_parts)

        # 4. Prepare the RAG prompt.
        prompt = f"""
You are PaperPilot, a careful research-paper assistant.

Answer the user's question using ONLY the supplied research-paper excerpts.

Rules:
1. Do not use outside knowledge or invent facts.
2. Cite each factual claim using the exact source marker
   provided in the context, such as [Page 5].
3. Put the citation immediately after the claim it supports.
4. Do not cite a page unless the supplied excerpt supports the claim.
5. If the excerpts do not provide enough evidence, clearly say so.
6. Distinguish what the paper explicitly reports from your interpretation.
7. Do not guess unsupported statistics, relationships, or conclusions.
8. Keep the answer clear and concise.
9. Answer the exact question asked. Do not substitute a related
   quantity for the requested quantity.
10. Distinguish the number of institutions, participants,
    observations, and samples. They are not interchangeable.
11. When asked for a sample size, provide it only if the retrieved
    excerpts explicitly establish what was counted. Otherwise,
    state that the sample size cannot be verified from the excerpts.
12. Do not present a paraphrase as a verbatim quotation.
    Use quotation marks only for text copied exactly from the excerpts.

{context}

USER QUESTION:
{question}
"""

        # 5. Generate an answer using Groq.
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )

        answer = response.choices[0].message.content or ""
        import re
        retrieved_pages = {
            str(metadata.get("page"))
            for metadata in metadata_list
            if metadata.get("page") is not None
        }
        cited_pages = set(
            re.findall(r"\[Page\s+(\d+)\]", answer, re.IGNORECASE)
        
        )
        invalid_pages = cited_pages - retrieved_pages
        if invalid_pages:
             answer += (
        "\n\n**Citation warning:** The answer cites page(s) "
        + ", ".join(sorted(invalid_pages, key=int))
        + " that were not among the retrieved source pages. "
        "Please verify these citations against the PDF."
    )






        
        


        return {
            "question": question,

            "answer": answer,
            "pages": sorted({
                metadata.get("page")
                for metadata in metadata_list
                if metadata.get("page") is not None
            }),
            "sources": sources
        }

    except HTTPException:
        raise

    except Exception:
        import traceback
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                "The question could not be processed. "
                "Check the server terminal for the error."
            ),
        )
