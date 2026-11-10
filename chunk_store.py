"""
Small, dependency-free helpers for the link between a document and its chunks in Chroma.

Both the code that STORES chunks and the code that DELETES them use these functions, so the
way a chunk is labelled and the way it is looked up can never drift apart. That drift is how
orphaned vectors (chunks of a deleted document that keep showing up in answers) happen.

Nothing here imports chromadb or the embedding model: the collection is passed in.
"""


def build_chunk_records(doc_id, doc_name: str, chunks: list[str]):
    """Return (ids, metadatas) for the chunks of one document.

    The document id is always stored as TEXT, because Postgres ids are integers but
    Chroma metadata is looked up exactly: the int 7 and the text "7" are different values.
    """
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {"doc_id": str(doc_id), "doc_name": doc_name, "chunk_index": i}
        for i in range(len(chunks))
    ]
    return ids, metadatas


def count_chunks_for_document(collection, doc_id) -> int:
    return len(collection.get(where={"doc_id": str(doc_id)})["ids"])


def delete_chunks_for_document(collection, doc_id) -> int:
    """Delete every chunk of one document. Returns how many were removed (0 if none)."""
    existing = collection.get(where={"doc_id": str(doc_id)})
    ids = existing["ids"]
    if ids:
        collection.delete(ids=ids)
    return len(ids)