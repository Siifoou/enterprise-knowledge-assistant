from collections import Counter
from embeddings_store import collection

data = collection.get(include=["metadatas"])
print("Total chunks:", len(data["ids"]))

counts = Counter((m["doc_id"], m["doc_name"]) for m in data["metadatas"])
for (doc_id, doc_name), n in sorted(counts.items()):
    print(f"  doc_id={doc_id}  {doc_name}: {n} chunks")