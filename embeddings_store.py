import os
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from chunk_store import build_chunk_records

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-mpnet-base-v2")
# Where Chroma keeps its files. Configurable so tests and experiments can use their own folder.
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_data")

# Loaded once, reused for every call — loading the model is slow, embedding text is fast
_model = None


def get_model():
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL} (first run only, may take a moment)")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    model = get_model()
    embeddings = model.encode(chunks, show_progress_bar=False)
    return embeddings.tolist()


# Chroma client — persists to disk, survives script restarts
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="documents")


def store_chunks(doc_id: str, doc_name: str, chunks: list[str]):
    embeddings = embed_chunks(chunks)

    # The labels come from chunk_store, the same module that deletion uses to find them again
    ids, metadatas = build_chunk_records(doc_id, doc_name, chunks)

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    print(f"Stored {len(chunks)} chunks for '{doc_name}' in Chroma.")


if __name__ == "__main__":
    from extraction import extract_text
    from chunking import chunk_text

    text = extract_text("data/documents/vacation_policy.docx")
    chunks = chunk_text(text)
    store_chunks(doc_id="doc_001", doc_name="vacation_policy.docx", chunks=chunks)

    print(f"\nTotal items now in Chroma collection: {collection.count()}")