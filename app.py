
import os
import hashlib
import time
import requests

import pymupdf
import streamlit as st
from dotenv import load_dotenv

from paperpilot.embeddings import create_embeddings
from paperpilot.database import collection


# ---------- CONFIGURATION ----------

load_dotenv()

st.set_page_config(
    page_title="PaperPilot",
    page_icon="📚",
    layout="wide"
)
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
                with st.spinner("Finding relevant passages and generating answer..."):
                    response = requests.post(
                        "http://127.0.0.1:8000/ask",
                        json={
                            "question": question,
                            "paper_id": selected_id
                        },
                        timeout=180
                    )

                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer") or "No answer was returned."
                    st.markdown(answer)

                   
                  

                    
                else:
                    try:
                        detail = response.json().get(
                            "detail", response.text
                        )
                    except ValueError:
                        detail = response.text

                    answer = f"API error ({response.status_code}): {detail}"
                    st.error(answer)

                history.append({
                    "role": "assistant",
                    "content": answer
                })

            except requests.RequestException as error:
                st.error(
                    "Could not connect to FastAPI at "
                    "http://127.0.0.1:8000. "
                    "Make sure the API server is running. "
                    f"Details: {error}"
                )


