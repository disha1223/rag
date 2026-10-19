
from paperpilot.embeddings import create_embeddings
from paperpilot.database import collection


def search_paper(question, top_k=3):
    """Find paper chunks most relevant to a question."""

    query_embedding = create_embeddings([question])

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    matches = []

    for i, text in enumerate(results["documents"][0]):
        matches.append({
            "text": text,
            "page": results["metadatas"][0][i]["page"],
            "distance": results["distances"][0][i]
        })

    return matches
