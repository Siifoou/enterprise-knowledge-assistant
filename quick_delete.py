from embeddings_store import collection
collection.delete(where={"doc_id": "doc_001"})
print("Remaining:", collection.count())