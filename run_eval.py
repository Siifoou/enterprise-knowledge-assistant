import csv
import hashlib
import json
import os
import re
import time
from datetime import datetime

from generation import ask, DISTANCE_THRESHOLD, NO_ANSWER
from retrieval import TOP_K, retrieve_relevant_chunks
from chunking import CHUNK_SIZE, CHUNK_OVERLAP
from embeddings_store import EMBEDDING_MODEL
from llm_provider import LLM_MODEL

DATASET_PATH = "golden_dataset.json"
os.makedirs("eval_results", exist_ok=True)

with open(DATASET_PATH, "rb") as f:
    raw = f.read()
dataset_version = hashlib.sha256(raw).hexdigest()[:8]
dataset = json.loads(raw.decode("utf-8"))


def numbers_in(text: str) -> set:
    # crude hint only: "10:00" -> {10}, zeros ignored
    return {int(n) for n in re.findall(r"\d+", text)} - {0}


def rate(rows, key):
    vals = [r[key] for r in rows if r[key] is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


retrieve_relevant_chunks("warm-up")  # loads the model first so it doesn't distort timings

rows = []
contexts_by_id = {}  # full chunk texts, saved in the JSON only (too long for the CSV)
for item in dataset:
    result = ask(item["question"], use_cache=False)  # fresh Groq calls, never cached, for evaluation
    retrieved = result["retrieved"]
    contexts_by_id[item["id"]] = [r["text"] for r in retrieved]
    acceptable = item["acceptable_sources"]
    answerable = item["answerability"] == "answerable"
    refused = result["answer"] == NO_ANSWER

    row = {
        "id": item["id"],
        "category": item["category"],
        "split": item.get("split", "calibration"),
        "question": item["question"],
        "expected_answer": item["expected_answer"],
        "system_answer": result["answer"],
        "top_distance": round(retrieved[0]["distance"], 3) if retrieved else None,
        "retrieved_docs": " | ".join(r["doc_name"] for r in retrieved),
        "retrieval_time_s": round(result["latency"]["retrieval"], 3),
        "generation_time_s": round(result["latency"]["generation"], 3),
    }

    if answerable:
        expected_nums = numbers_in(item["expected_answer"])
        row["retrieval_hit"] = any(r["doc_name"] in acceptable for r in retrieved)
        row["top1_hit"] = bool(retrieved) and retrieved[0]["doc_name"] in acceptable
        row["false_refusal"] = refused
        row["number_match"] = (expected_nums <= numbers_in(result["answer"])) if expected_nums else None
        row["unanswerable_result"] = None
    else:
        row["retrieval_hit"] = None
        row["top1_hit"] = None
        row["false_refusal"] = None
        row["number_match"] = None
        row["unanswerable_result"] = "refused_by_threshold" if refused else "REVIEW_MANUALLY"

    rows.append(row)
    print(f"{item['id']:>4} {item['category']:<13} top1={row['top1_hit']} refused={refused} dist={row['top_distance']}")
    time.sleep(1)  # be gentle with Groq rate limits

answered = [r for r in rows if r["retrieval_hit"] is not None]
unanswered = [r for r in rows if r["unanswerable_result"] is not None]

by_category = {}
for cat in sorted({r["category"] for r in answered}):
    cat_rows = [r for r in answered if r["category"] == cat]
    by_category[cat] = {
        "n": len(cat_rows),
        "retrieval_hit_rate": rate(cat_rows, "retrieval_hit"),
        "top1_hit_rate": rate(cat_rows, "top1_hit"),
    }

by_split = {}
for sp in sorted({r["split"] for r in answered}):
    sp_rows = [r for r in answered if r["split"] == sp]
    by_split[sp] = {
        "n": len(sp_rows),
        "retrieval_hit_rate": rate(sp_rows, "retrieval_hit"),
        "top1_hit_rate": rate(sp_rows, "top1_hit"),
    }

gen_times = [r["generation_time_s"] for r in rows if r["generation_time_s"] > 0]

summary = {
    "answerable_questions": len(answered),
    "retrieval_hit_rate": rate(answered, "retrieval_hit"),
    "top1_hit_rate": rate(answered, "top1_hit"),
    "false_refusals": sum(r["false_refusal"] for r in answered),
    "number_match_rate (crude hint)": rate(answered, "number_match"),
    "by_category": by_category,
    "by_split": by_split,
    "unanswerable_questions": len(unanswered),
    "unanswerable_refused_by_threshold": sum(r["unanswerable_result"] == "refused_by_threshold" for r in unanswered),
    "unanswerable_needing_manual_review": sum(r["unanswerable_result"] == "REVIEW_MANUALLY" for r in unanswered),
    "avg_retrieval_time_s": round(sum(r["retrieval_time_s"] for r in rows) / len(rows), 3),
    "avg_generation_time_s": round(sum(gen_times) / len(gen_times), 3) if gen_times else None,
}

config = {
    "date": datetime.now().isoformat(timespec="seconds"),
    "dataset_version": dataset_version,
    "dataset_size": len(dataset),
    "embedding_model": EMBEDDING_MODEL,
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "top_k": TOP_K,
    "llm_model": LLM_MODEL,
    "distance_threshold": DISTANCE_THRESHOLD,
}

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
json_rows = [{**r, "retrieved_contexts": contexts_by_id[r["id"]]} for r in rows]
with open(f"eval_results/eval_{stamp}.json", "w", encoding="utf-8") as f:
    json.dump({"config": config, "summary": summary, "rows": json_rows}, f, ensure_ascii=False, indent=2)

with open(f"eval_results/eval_{stamp}.csv", "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("\n=== CONFIG ===")
print(json.dumps(config, indent=2))
print("\n=== SUMMARY ===")
print(json.dumps(summary, indent=2))
print(f"\nSaved to eval_results/eval_{stamp}.json and .csv")