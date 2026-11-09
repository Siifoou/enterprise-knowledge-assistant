"""
Tiny experiment: does FactualCorrectness give 0 because the reference answer is a short fragment?
Same answer each time, only the reference changes. Run inside venv-eval:  python diagnose_factual.py
"""
import os

import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import FactualCorrectness
from ragas.run_config import RunConfig

load_dotenv()
JUDGE_MODEL = os.getenv("JUDGE_MODEL", os.getenv("LLM_MODEL", "openai/gpt-oss-120b"))

FR_ANSWER = "Après 3 ans d'ancienneté, vous avez droit à **22 jours de congé** par an."
EN_ANSWER = "Someone with 6 years of service gets **27 vacation days per year**."

cases = [
    ("A  FR  short reference (what the dataset has now)", FR_ANSWER, "22 jours"),
    ("B  FR  full-sentence reference", FR_ANSWER,
     "Après 3 ans d'ancienneté, un salarié a droit à 22 jours de congé par an."),
    ("C  FR  full sentence, WRONG number (should score low)", FR_ANSWER,
     "Après 3 ans d'ancienneté, un salarié a droit à 18 jours de congé par an."),
    ("D  EN  short reference", EN_ANSWER, "27 days"),
    ("E  EN  full-sentence reference", EN_ANSWER,
     "An employee with 6 years of service gets 27 vacation days per year."),
]

dataset = EvaluationDataset.from_list([
    {"user_input": "question", "retrieved_contexts": ["context"], "response": resp, "reference": ref}
    for _, resp, ref in cases
])

result = evaluate(
    dataset=dataset,
    metrics=[FactualCorrectness()],
    llm=LangchainLLMWrapper(ChatGroq(model=JUDGE_MODEL, temperature=0)),
    run_config=RunConfig(max_workers=2, timeout=180, max_retries=6, max_wait=60),
)

df = result.to_pandas()
score_col = [c for c in df.columns if c.startswith("factual_correctness")][0]
print("\n=== FACTUAL CORRECTNESS, same answer, different reference ===")
for (label, _, _), score in zip(cases, df[score_col]):
    print(f"{label:<58} -> {'blank' if pd.isna(score) else round(float(score), 2)}")