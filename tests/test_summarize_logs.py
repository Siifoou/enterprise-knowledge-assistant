"""Tests for summarize_logs.py: the numbers that go into the report must be computed correctly."""
from summarize_logs import describe, percentile, summarize

RECORDS = [
    {"refused_by_threshold": False, "retrieval_time_s": 0.1, "generation_time_s": 1.0, "total_time_s": 1.1,
     "prompt_tokens": 100, "completion_tokens": 10, "cached": False, "retrieved": [{"doc_name": "a.docx"}]},
    {"refused_by_threshold": False, "retrieval_time_s": 0.2, "generation_time_s": 2.0, "total_time_s": 2.2,
     "prompt_tokens": 200, "completion_tokens": 20, "cached": False, "retrieved": [{"doc_name": "a.docx"}]},
    {"refused_by_threshold": False, "retrieval_time_s": 0.3, "generation_time_s": 3.0, "total_time_s": 3.3,
     "prompt_tokens": None, "completion_tokens": None, "cached": True, "retrieved": [{"doc_name": "b.pdf"}]},
    {"refused_by_threshold": True, "retrieval_time_s": 0.1, "generation_time_s": 0.0, "total_time_s": 0.1,
     "prompt_tokens": None, "completion_tokens": None, "cached": False, "retrieved": [{"doc_name": "b.pdf"}]},
]


def test_counts_and_refusal_rate():
    s = summarize(RECORDS)
    assert s["queries"] == 4
    assert s["refused_by_threshold"] == 1
    assert s["refused_rate"] == 0.25
    assert s["answered_from_cache"] == 1


def test_generation_time_ignores_questions_that_never_called_the_model():
    """A refused question has generation time 0: counting it would make the model look faster than it is."""
    s = summarize(RECORDS)
    assert s["generation_time_s"]["n"] == 3
    assert s["generation_time_s"]["mean"] == 2.0
    assert s["generation_time_s"]["median"] == 2.0


def test_token_totals_skip_missing_values():
    s = summarize(RECORDS)
    assert s["prompt_tokens_total"] == 300
    assert s["completion_tokens_total"] == 30


def test_top_retrieved_documents_are_counted():
    s = summarize(RECORDS)
    assert s["top_retrieved_documents"][0] == ("a.docx", 2) or s["top_retrieved_documents"][0] == ("b.pdf", 2)
    assert dict(s["top_retrieved_documents"]) == {"a.docx": 2, "b.pdf": 2}


def test_an_empty_log_does_not_crash():
    s = summarize([])
    assert s["queries"] == 0
    assert s["refused_rate"] is None
    assert s["total_time_s"] is None


def test_percentile_uses_the_nearest_rank_method():
    assert percentile(list(range(1, 101)), 95) == 95
    assert percentile([5.0], 95) == 5.0
    assert percentile([1, 2, 3], 95) == 3


def test_describe_of_nothing_is_none():
    assert describe([]) is None
