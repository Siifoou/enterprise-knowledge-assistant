import os
from dotenv import load_dotenv
from embeddings_store import get_model, collection

load_dotenv()

TOP_K = int(os.getenv("TOP_K", 3))


def retrieve_relevant_chunks(question: str, top_k: int = None) -> list[dict]:
    top_k = top_k or TOP_K
    model = get_model()

    question_embedding = model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k,
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "doc_name": results["metadatas"][0][i]["doc_name"],
            "distance": results["distances"][0][i],
        })

    return chunks


if __name__ == "__main__":
    question = "How many vacation days do I get after 3 years?"
    results = retrieve_relevant_chunks(question)

    print(f"Question: {question}\n")
    for i, r in enumerate(results):
        print(f"--- Result {i} (distance: {r['distance']:.4f}, from {r['doc_name']}) ---")
        print(r['text'])
        print()