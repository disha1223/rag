
def chunk_paragraphs(paragraphs, chunk_size=220, overlap=40):
    """
    Combine extracted paragraphs into word-based chunks.
    Keep page numbers for citations.
    """
    chunks = []
    current_words = []
    current_page = None

    for paragraph in paragraphs:
        words = paragraph["text"].split()
        page = paragraph["page"]

        if not words:
            continue

        if not current_words:
            current_page = page

        current_words.extend(words)

        while len(current_words) >= chunk_size:
            chunk_words = current_words[:chunk_size]

            chunks.append({
                "text": " ".join(chunk_words),
                "page": current_page
            })

            current_words = current_words[chunk_size - overlap:]
            current_page = page

    if current_words and len(current_words) >= 30:
        chunks.append({
            "text": " ".join(current_words),
            "page": current_page
        })

    return chunks
