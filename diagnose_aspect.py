"""
Tests a binary yes/no judge (RAGAS AspectCritic) on cases where the right verdict is known.
Run inside venv-eval:  python diagnose_aspect.py
"""
import os

import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AspectCritic
from ragas.run_config import RunConfig

load_dotenv()
JUDGE_MODEL = os.getenv("JUDGE_MODEL", os.getenv("LLM_MODEL", "openai/gpt-oss-120b"))

DEFINITION = (
    "Answer 1 (Yes) only if the response states the same key fact(s) as the reference answer "
    "and does not contradict it. The response may be longer or worded differently, numbers may "
    "be written as digits or as words, and the response may be in a different language than the "
    "reference. Answer 0 (No) if the response gives a different key fact or number than the "
    "reference, or leaves it out. If the reference says the information is not available in the "
    "documents, answer 1 only if the response also says the question cannot be answered from the "
    "available documents. Compare only to the reference answer, not to your own knowledge."
)

FR = "Après 3 ans d'ancienneté, vous avez droit à **22 jours de congé** par an."
EN = "Someone with 6 years of service gets **27 vacation days per year**."
NO_ANSWER = "I don't know based on the available documents."

# (label, expected verdict, response, reference)
cases = [
    ("1 FR correct, short reference",            1, FR, "22 jours"),
    ("2 FR correct, full-sentence reference",    1, FR, "Après 3 ans d'ancienneté, un salarié a droit à 22 jours de congé par an."),
    ("3 FR vs WRONG full-sentence reference",    0, FR, "Après 3 ans d'ancienneté, un salarié a droit à 18 jours de congé par an."),
    ("4 FR vs WRONG short reference",            0, FR, "18 jours"),
    ("5 EN correct, short reference",            1, EN, "27 days"),
    ("6 EN vs WRONG short reference",            0, EN, "30 days"),
    ("7 FR correct but wordy (extra true info)", 1,
     "Après 3 ans d'ancienneté, vous avez droit à 22 jours de congé par an, et vous pouvez reporter jusqu'à 5 jours non utilisés.",
     "22 jours"),
    ("8 'trois jours' vs '3 jours'",             1,
     "Vous pouvez télétravailler jusqu'à **trois jours par semaine**.", "3 jours par semaine"),
    ("9 refusal vs refusal reference",           1,
     "The provided context does not contain information about sick-day entitlement.", NO_ANSWER),
    ("10 confident invented answer vs refusal",  0,
     "You are entitled to 10 sick days per year.", NO_ANSWER),
]

dataset = EvaluationDataset.from_list([
    {"user_input": "question", "retrieved_contexts": ["context"], "response": resp, "reference": ref}
    for _, _, resp, ref in cases
])

result = evaluate(
    dataset=dataset,
    metrics=[AspectCritic(name="answer_correct", definition=DEFINITION)],
    llm=LangchainLLMWrapper(ChatGroq(model=JUDGE_MODEL, temperature=0)),
    run_config=RunConfig(max_workers=2, timeout=180, max_retries=6, max_wait=60),
)

df = result.to_pandas()
score_col = "answer_correct"
print("\n=== BINARY JUDGE: verdict vs what it should be ===")
wrong = 0
for (label, expected, _, _), score in zip(cases, df[score_col]):
    if pd.isna(score):
        shown, flag = "blank", "  <-- no verdict"
        wrong += 1
    else:
        shown = int(score)
        flag = "" if shown == expected else "  <-- WRONG"
        wrong += shown != expected
    print(f"{label:<44} should be {expected}, got {shown}{flag}")
print(f"\n{len(cases) - wrong} of {len(cases)} verdicts as expected")