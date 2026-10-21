
from paperpilot.database import collection

result = collection.get(include=["metadatas"])

seen = {}

for metadata in result.get("metadatas") or []:
    if not metadata:
        continue

    paper_id = metadata.get("paper_id")
    filename = metadata.get("filename", "Unknown filename")

    if paper_id:
        seen[paper_id] = filename

if not seen:
    print("No indexed papers found in this ChromaDB database.")
else:
    for paper_id, filename in seen.items():
        print(f"Filename: {filename}")
        print(f"Paper ID: {paper_id}")
        print("-" * 40)
