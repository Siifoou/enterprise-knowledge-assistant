"""
Turns the query log into numbers for the report.
    python summarize_logs.py                 (reads logs/queries.jsonl, or QUERY_LOG_PATH)
    python summarize_logs.py path/to/queries.jsonl
"""
import json
import math
import statistics
import sys
from collections import Counter

from query_log import read_log


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(p / 100 * len(ordered)) - 1)
    return ordered[index]


def describe(values: list[float]):
    if not values:
        return None
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "p95": round(percentile(values, 95), 3),
        "max": round(max(values), 3),
    }


def _numbers(records, key):
    return [r[key] for r in records if isinstance(r.get(key), (int, float))]


def summarize(records: list[dict]) -> dict:
    n = len(records)
    refused = sum(1 for r in records if r.get("refused_by_threshold"))
    generated = [r for r in records if not r.get("refused_by_threshold")]  # these actually called the LLM
    top_docs = Counter(r["retrieved"][0]["doc_name"] for r in records if r.get("retrieved"))
    return {
        "queries": n,
        "refused_by_threshold": refused,
        "refused_rate": round(refused / n, 3) if n else None,
        "answered_from_cache": sum(1 for r in records if r.get("cached")),
        "retrieval_time_s": describe(_numbers(records, "retrieval_time_s")),
        "generation_time_s": describe(_numbers(generated, "generation_time_s")),
        "total_time_s": describe(_numbers(records, "total_time_s")),
        "prompt_tokens_total": sum(_numbers(records, "prompt_tokens")),
        "completion_tokens_total": sum(_numbers(records, "completion_tokens")),
        "top_retrieved_documents": top_docs.most_common(5),
    }


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    records = read_log(path)
    if not records:
        print("No queries logged yet.")
    else:
        print(json.dumps(summarize(records), indent=2, ensure_ascii=False))