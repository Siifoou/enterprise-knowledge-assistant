import os
import time
from dotenv import load_dotenv
from retrieval import retrieve_relevant_chunks
from llm_provider import generate_answer_with_usage

load_dotenv()

# Provisional placeholder, to be recalibrated against the golden dataset results
DISTANCE_THRESHOLD = float(os.getenv("DISTANCE_THRESHOLD", 12.0))

NO_ANSWER = "I don't know based on the available documents."

PROMPT_TEMPLATE = """You are an assistant answering questions using ONLY the context provided below.

Context:
{context}

Question: {question}

Instructions:
- Answer using only the information in the context above.
- Do not use outside knowledge or make anything up.
- If the context does not contain enough information to answer, say so clearly.
- Be concise.

Answer:"""


def ask(question: str, use_cache: bool = True) -> dict:
    start = time.time()
    chunks = retrieve_relevant_chunks(question)
    retrieval_time = time.time() - start

    # "text" is the full chunk text: the RAGAS evaluation needs it to judge the answer
    retrieved = [
        {"doc_name": c["doc_name"], "distance": c["distance"], "text": c["text"]}
        for c in chunks
    ]

    if not chunks or chunks[0]["distance"] > DISTANCE_THRESHOLD:
        return {
            "answer": NO_ANSWER,
            "sources": [],
            "retrieved": retrieved,
            "latency": {"retrieval": retrieval_time, "generation": 0.0},
            "usage": {"prompt_tokens": None, "completion_tokens": None, "cached": False},
            "refused_by_threshold": True,
        }

    context = "\n\n".join(c["text"] for c in chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    start = time.time()
    generated = generate_answer_with_usage(prompt, use_cache=use_cache)
    generation_time = time.time() - start

    sources = [{"doc_name": c["doc_name"], "excerpt": c["text"][:150]} for c in chunks]

    return {
        "answer": generated["text"],
        "sources": sources,
        "retrieved": retrieved,
        "latency": {"retrieval": retrieval_time, "generation": generation_time},
        "usage": {
            "prompt_tokens": generated["prompt_tokens"],
            "completion_tokens": generated["completion_tokens"],
            "cached": generated["cached"],
        },
        "refused_by_threshold": False,
    }


if __name__ == "__main__":
    result = ask("How many vacation days do I get after 3 years?")
    print("ANSWER:", result["answer"])
    print("TOKENS:", result["usage"])
    print("\nSOURCES:")
    for s in result["sources"]:
        print(f"  - {s['doc_name']}: {s['excerpt']}...")