
import os
import hashlib
import time

import pymupdf
import streamlit as st
from dotenv import load_dotenv
from google import genai

from paperpilot.embeddings import create_embeddings
from paperpilot.database import collection


# ---------- CONFIGURATION ----------

load_dotenv()

st.set_page_config(
    page_title="PaperPilot",
    page_icon="📚",
    layout="wide"
)

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY is missing. Check your .env file.")
    st.stop()

client = genai.Client(api_key=api_key)

st.title("📚 PaperPilot")
st.caption("Your AI research assistant — ask questions about your papers.")


# ---------- PDF PROCESSING ----------

def process_pdf(pdf_bytes, filename):
    paper_id = hashlib.sha256(pdf_bytes).hexdigest()

    existing = collection.get(
        where={"paper_id": paper_id},
        limit=1
    )

    if existing["ids"]:
        return paper_id, "This paper is already indexed."

    pdf = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    chunks = []

    try:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if not text:
                continue

            words = text.split()
            chunk_size = 220
            overlap = 40
            start = 0

            while start < len(words):
                chunk_text = " ".join(
                    words[start:start + chunk_size]
                )

                if chunk_text.strip():
                    chunks.append({
                        "text": chunk_text,
                        "page": page_number
                    })

                if start + chunk_size >= len(words):
                    break

                start += chunk_size - overlap
    finally:
        pdf.close()

    if not chunks:
        raise ValueError(
            "No readable text found. This PDF may need OCR."
        )

    texts = [chunk["text"] for chunk in chunks]
    embeddings = create_embeddings(texts)

    ids = [
        f"{paper_id}_chunk_{i}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {
            "paper_id": paper_id,
            "page": chunk["page"],
            "filename": filename
        }
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    return paper_id, f"Indexed {len(chunks)} text chunks."


# ---------- PERSISTENT PAPER LIBRARY ----------

def get_paper_library():
    """List papers already indexed in ChromaDB."""
    library = {}

    result = collection.get(include=["metadatas"])

    for metadata in result.get("metadatas") or []:
        if not metadata:
            continue

        paper_id = metadata.get("paper_id")
        if not paper_id:
            continue

        if paper_id not in library:
            library[paper_id] = {
                "filename": metadata.get("filename", "Untitled paper"),
                "pages": set(),
                "chunks": 0
            }

        library[paper_id]["chunks"] += 1

        page = metadata.get("page")
        if page is not None:
            library[paper_id]["pages"].add(page)

    return library

# ---------- RETRIEVAL ----------

def prepare_question(question, paper_id):
    total_start = time.perf_counter()

    start = time.perf_counter()
    query_embedding = create_embeddings([question])
    embedding_time = time.perf_counter() - start

    start = time.perf_counter()

    # Restrict retrieval to the selected paper
    paper_count = collection.get(
        where={"paper_id": paper_id},
        limit=1
    )

    if not paper_count["ids"]:
        return None, {
            "embedding": embedding_time,
            "search": 0.0,
            "total": time.perf_counter() - total_start
        }

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=3,
        where={"paper_id": paper_id},
        include=["documents", "metadatas"]
    )

    search_time = time.perf_counter() - start

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not documents:
        return None, {
            "embedding": embedding_time,
            "search": search_time,
            "total": time.perf_counter() - total_start
        }

    context_parts = []

    for text, metadata in zip(documents, metadatas):
        context_parts.append(
            f"[PDF page {metadata['page']}]\n{text}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are PaperPilot, a research paper assistant.

Answer using ONLY the provided paper excerpts.
Do not invent facts.
Cite supporting pages using [PDF page N].
Keep the answer concise, under 120 words.
If the excerpts do not contain the answer, say so clearly.

PAPER EXCERPTS:
{context}

QUESTION:
{question}
"""

    return prompt, {
        "embedding": embedding_time,
        "search": search_time,
        "total": time.perf_counter() - total_start
    }


# ---------- GEMINI STREAMING ----------

def stream_answer(prompt, timings):
    start = time.perf_counter()
    first_text_time = None

    try:
        stream = client.models.generate_content_stream(
            model="gemini-3.6-flash",
            contents=prompt
        )

        for chunk in stream:
            if chunk.text:
                if first_text_time is None:
                    first_text_time = time.perf_counter() - start

                yield chunk.text

    finally:
        gemini_time = time.perf_counter() - start

        print("\n--- PaperPilot performance ---")
        print(f"Question embedding: {timings['embedding']:.2f} seconds")
        print(f"ChromaDB search:    {timings['search']:.2f} seconds")
        print(
            f"First text received: {first_text_time:.2f} seconds"
            if first_text_time is not None
            else "First text received: No text"
        )
        print(f"Gemini generation:  {gemini_time:.2f} seconds")
        print(
            f"Total: {timings['total'] + gemini_time:.2f} seconds"
        )
        print("------------------------------\n")


# ---------- USER INTERFACE ----------

# ---------- USER INTERFACE ----------

st.subheader("📚 My Research Library")
st.caption("Your indexed research papers are saved in ChromaDB.")

uploaded_files = st.file_uploader(
    "Upload research papers (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

# Index uploaded papers
if uploaded_files:
    for uploaded_file in uploaded_files:
        pdf_bytes = uploaded_file.getvalue()
        paper_id = hashlib.sha256(pdf_bytes).hexdigest()

        existing = collection.get(
            where={"paper_id": paper_id},
            limit=1
        )

        if existing["ids"]:
            st.info(f"{uploaded_file.name} is already indexed.")
        else:
            with st.spinner(f"Indexing {uploaded_file.name}..."):
                try:
                    _, message = process_pdf(
                        pdf_bytes,
                        uploaded_file.name
                    )
                    st.success(f"{uploaded_file.name}: {message}")

                except Exception as error:
                    st.error(
                        f"Could not process {uploaded_file.name}: {error}"
                    )

# Load all papers from the persistent database
library = get_paper_library()

if not library:
    st.info("Your library is empty. Upload a PDF to get started.")

else:
    st.write(f"**{len(library)} paper(s) in your library**")

    paper_ids = list(library.keys())

    selected_id = st.selectbox(
        "Select a paper to chat with",
        options=paper_ids,
        format_func=lambda pid: library[pid]["filename"],
        key="selected_paper_id"
    )

    # Display saved papers and their indexed details
    with st.expander("View all papers in your library", expanded=True):
        for pid, details in library.items():
            st.markdown(f"**📄 {details['filename']}**")
            st.caption(
                f"{len(details['pages'])} pages detected · "
                f"{details['chunks']} indexed chunks"
            )

    st.divider()
    selected_name = library[selected_id]["filename"]
    st.subheader(f"💬 Chat with {selected_name}")

    # Keep a separate conversation for each paper
    if "chat_histories" not in st.session_state:
        st.session_state["chat_histories"] = {}

    if selected_id not in st.session_state["chat_histories"]:
        st.session_state["chat_histories"][selected_id] = []

    history = st.session_state["chat_histories"][selected_id]

    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask something about this research paper...",
        key=f"question_{selected_id}"
    )

    if question:
        history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Finding relevant passages..."):
                    prompt, timings = prepare_question(
                        question,
                        selected_id
                    )

                if prompt is None:
                    answer = (
                        "I couldn't find relevant passages in this paper."
                    )
                    st.markdown(answer)

                else:
                    answer = st.write_stream(
                        stream_answer(prompt, timings)
                    )

                    if not answer:
                        answer = "No answer was returned."
                        st.markdown(answer)

                history.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as error:
                st.error(f"Something went wrong: {error}")

