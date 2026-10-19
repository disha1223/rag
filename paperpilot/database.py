
import chromadb

# Create a database stored on your laptop
client = chromadb.PersistentClient(path="data/chroma_db")

# Create or open the collection
collection = client.get_or_create_collection(
    name="research_papers",
    metadata={"hnsw:space": "cosine"}
)


def save_chunks(paper_id, chunks, embeddings):
    """Save text chunks and their embeddings."""

    ids = [
        f"{paper_id}_chunk_{i}"
        for i in range(len(chunks))
    ]

    documents = [chunk["text"] for chunk in chunks]

    metadatas = [
        {
            "paper_id": paper_id,
            "page": chunk["page"]
        }
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    print(f"Saved {len(chunks)} chunks to ChromaDB!")
