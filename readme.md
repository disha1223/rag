# PaperPilot — AI Research Paper Assistant

PaperPilot is an AI-powered research assistant that helps users explore academic research papers by asking questions about their contents. It uses Retrieval-Augmented Generation (RAG) to retrieve relevant passages from uploaded PDFs and generate context-aware answers.

## Features

- **PDF ingestion:** Extract text from academic PDFs using PyMuPDF.
- **Text chunking:** Split documents into overlapping text chunks.
- **Semantic embeddings:** Generate embeddings using Sentence Transformers and `BAAI/bge-small-en-v1.5`.
- **Vector search:** Store and retrieve document chunks using ChromaDB.
- **AI-powered answers:** Use the Google Gemini API to generate answers from retrieved context.
- **Research library:** Upload and select multiple indexed research papers.
- **Duplicate detection:** Use SHA-256 hashing to identify identical PDF content.
- **Paper-specific retrieval:** Restrict searches to the selected research paper.
- **Streaming responses:** Display generated answers progressively in the chat interface.

## Tech Stack

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- BAAI/bge-small-en-v1.5
- ChromaDB
- Google Gemini API
- python-dotenv

## How It Works

1. Upload a research paper in PDF format.
2. Extract text from the PDF.
3. Split the text into overlapping chunks.
4. Generate embeddings for the chunks.
5. Store the text, embeddings, and metadata in ChromaDB.
6. Enter a question about the selected paper.
7. Retrieve the three most relevant chunks using semantic similarity search.
8. Send the retrieved context and question to Gemini.
9. Display the generated response in the Streamlit chat interface.

## Project Structure

```text
paperPilot/
├── app.py
├── paperpilot/
│   ├── __init__.py
│   ├── embeddings.py
│   ├── database.py
│   ├── ingest.py
│   ├── chunking.py
│   └── search.py
├── data/
│   └── papers/
├── .env.example
├── .gitignore
├── requirements.txt
├── test_gemini.py
└── test_ingest.py
```

## Setup and Installation

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd paperPilot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root using `.env.example` as a template.

Add your own Google Gemini API key:

```dotenv
GEMINI_API_KEY=your_actual_api_key
```

A Hugging Face token is optional for public models that can be downloaded without authentication.

**Never commit `.env` or publish API keys.**

### 5. Run PaperPilot

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

## Architecture

**PDF → Text Extraction → Chunking → Embeddings → ChromaDB → Retrieval → Gemini → Answer**

The embedding model runs locally once downloaded. Answer generation uses the Gemini API, so the application requires internet access for that stage.

## Current Limitations

- Answer quality depends on the relevance of retrieved chunks and the quality of the language model's response.
- Scanned PDFs may require OCR.
- Original uploaded PDF files are not currently stored as a permanent document archive by the application.
- Chat histories are maintained in the active Streamlit session.
- Gemini API usage is subject to the configured model's availability and account quotas.

## Future Improvements

- Evaluate retrieval quality and answer grounding.
- Improve source citations and unsupported-answer handling.
- Add safe document deletion and persistent chat history.
- Support OCR and improved extraction of tables.
- Add automated tests for retrieval and answer quality.

## Security

Keep API credentials in environment variables. Do not commit `.env`, virtual environments, local databases, or private research documents.

## License

No license has been specified yet. Add an appropriate license before inviting others to reuse or redistribute the project.