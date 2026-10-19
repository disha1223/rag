
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5",
    token=hf_token
)

def create_embeddings(texts):
    return model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )
