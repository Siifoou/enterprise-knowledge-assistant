"""
Scores a saved evaluation run with RAGAS (LLM-as-judge).
Reads eval_results/eval_*.json (made by run_eval.py), so it never touches Chroma or the pipeline.

What is scored:
  - answerable questions:   faithfulness, context_recall, answer_correct
  - unanswerable questions: answer_correct only (does the system correctly decline?)

Usage (inside the venv-eval environment):
    python run_ragas.py --limit 2        # smoke test: 2 answerable + 2 unanswerable questions
    python run_ragas.py                  # everything in the newest run
    python run_ragas.py --file eval_results/eval_20261108_162116.json
"""
import argparse
import glob
import json
import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AspectCritic, Faithfulness, LLMContextRecall
from ragas.run_config import RunConfig

load_dotenv()

# Judge model: defaults to the same model that writes the answers. A judge that is also the
# author can be slightly lenient on its own style, so note it as a limitation in the report,
# or set JUDGE_MODEL in .env to a different model from Groq's model list.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", os.getenv("LLM_MODEL", "openai/gpt-oss-120b"))

# Everything RAGAS adds besides the input columns is a metric score.
NON_METRIC_COLUMNS = {"id", "category", "split", "user_input", "retrieved_contexts", "response", "reference"}

# Validated on 10 hand-made cases with known verdicts (diagnose_aspect.py: 10 of 10 correct).
CORRECTNESS_DEFINITION = (
    "Answer 1 (Yes) only if the response states the same key fact(s) as the reference answer "
    "and does not contradict it. The response may be longer or worded differently, numbers may "
    "be written as digits or as words, and the response may be in a different language than the "
    "reference. Answer 0 (No) if the response gives a different key fact or number than the "
    "reference, or leaves it out. If the reference says the information is not available in the "
    "documents, answer 1 only if the response also says the question cannot be answered from the "
    "available documents. Compare only to the reference answer, not to your own knowledge."
)

parser = argparse.ArgumentParser()
parser.add_argument("--file", help="eval JSON to score (default: newest eval_*.json)")
parser.add_argument("--limit", type=int, help="smoke test: score only the first N questions of each group")
args = parser.parse_args()

files = sorted(glob.glob("eval_results/eval_*.json"))
if not args.file and not files:
    raise SystemExit("No eval_results/eval_*.json found. Run run_eval.py first.")
path = args.file or files[-1]

with open(path, encoding="utf-8") as f:
    data = json.load(f)

answerable = [r for r in data["rows"] if r.get("retrieval_hit") is not None]
unanswerable = [r for r in data["rows"] if r.get("retrieval_hit") is None]
if args.limit:
    answerable, unanswerable = answerable[: args.limit], unanswerable[: args.limit]
if not answerable:
    raise SystemExit("No answerable rows found in that file.")
if "retrieved_contexts" not in answerable[0]:
    raise SystemExit(
        "This results file has no chunk texts (it was made before they were saved). "
        "Re-run run_eval.py in the main environment, then try again."
    )

print(f"Scoring {len(answerable)} answerable + {len(unanswerable)} unanswerable questions from {path}")
print(f"Judge model: {JUDGE_MODEL}")

evaluator_llm = LangchainLLMWrapper(ChatGroq(model=JUDGE_MODEL, temperature=0))
# Few parallel workers + generous retries: Groq's free tier rate-limits quickly
run_config = RunConfig(max_workers=2, timeout=180, max_retries=6, max_wait=60)


def correctness_metric():
    return AspectCritic(name="answer_correct", definition=CORRECTNESS_DEFINITION)


def score(rows, metrics):
    dataset = EvaluationDataset.from_list([
        {
            "user_input": r["question"],
            "retrieved_contexts": r["retrieved_contexts"],
            "response": r["system_answer"],
            "reference": r["expected_answer"],
        }
        for r in rows
    ])
    result = evaluate(dataset=dataset, metrics=metrics, llm=evaluator_llm, run_config=run_config)
    df = result.to_pandas()
    df.insert(0, "id", [r["id"] for r in rows])
    df.insert(1, "category", [r["category"] for r in rows])
    df.insert(2, "split", [r.get("split", "calibration") for r in rows])
    return df


print("\n--- scoring answerable questions ---")
df_a = score(answerable, [Faithfulness(), LLMContextRecall(), correctness_metric()])
df_u = pd.DataFrame()
if unanswerable:
    print("\n--- scoring unanswerable questions ---")
    df_u = score(unanswerable, [correctness_metric()])

# blank scores = the judge failed (rate limit or unparseable output), counted per group
failed = {c: int(df_a[c].isna().sum()) for c in df_a.columns if c not in NON_METRIC_COLUMNS}
if len(df_u):
    failed["answer_correct (unanswerable)"] = int(df_u["answer_correct"].isna().sum())

df = pd.concat([df_a, df_u], ignore_index=True)
cols = [c for c in df.columns if c not in NON_METRIC_COLUMNS]

print("\n=== PER QUESTION (blank = not applicable for that question type, or judge failed) ===")
print(df[["id", "category", "split"] + cols].round(2).to_string(index=False))

print("\n=== MEANS, ANSWERABLE QUESTIONS ===")
print(df_a[[c for c in df_a.columns if c not in NON_METRIC_COLUMNS]].mean().round(3).to_string())
if len(df_u):
    print("\n=== answer_correct, UNANSWERABLE QUESTIONS (1 = correctly declined) ===")
    print(round(float(df_u["answer_correct"].mean()), 3))

print("\n=== FAILED JUDGMENTS (rate limit or parse error) ===")
print(pd.Series(failed).to_string())

print("\n=== BY CATEGORY ===")
print(df.groupby("category")[cols].mean().round(3).to_string())

print("\n=== BY SPLIT ===")
print(df.groupby("split")[cols].mean().round(3).to_string())

stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + ("_smoke" if args.limit else "")
df.drop(columns=["retrieved_contexts"], errors="ignore").to_csv(
    f"eval_results/ragas_{stamp}.csv", index=False, encoding="utf-8-sig"
)


def mean_or_none(series):
    m = series.mean()
    return None if pd.isna(m) else round(float(m), 3)


summary = {
    "judge_model": JUDGE_MODEL,
    "source_eval_file": path,
    "source_config": data.get("config"),
    "answerable_scored": len(answerable),
    "unanswerable_scored": len(unanswerable),
    "means_answerable": {c: mean_or_none(df_a[c]) for c in df_a.columns if c not in NON_METRIC_COLUMNS},
    "answer_correct_unanswerable": mean_or_none(df_u["answer_correct"]) if len(df_u) else None,
    "failed_judgments": failed,
}
with open(f"eval_results/ragas_{stamp}.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"\nSaved to eval_results/ragas_{stamp}.csv and .json")