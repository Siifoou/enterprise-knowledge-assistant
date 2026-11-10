"""
Tests for the decision logic in generation.py: when to answer, when to refuse, what is passed on.
retrieval and llm_provider are replaced by stand-ins, so no model, no Chroma and no API are needed.
"""
import sys
import types

import pytest


@pytest.fixture
def generation(monkeypatch):
    state = types.SimpleNamespace(chunks=[], llm_calls=[])

    fake_retrieval = types.ModuleType("retrieval")
    fake_retrieval.retrieve_relevant_chunks = lambda question: state.chunks

    fake_llm = types.ModuleType("llm_provider")

    def fake_generate(prompt, max_retries=3, use_cache=True):
        state.llm_calls.append({"prompt": prompt, "use_cache": use_cache})
        return {"text": "22 days", "prompt_tokens": 120, "completion_tokens": 9, "cached": False}

    fake_llm.generate_answer_with_usage = fake_generate

    monkeypatch.setitem(sys.modules, "retrieval", fake_retrieval)
    monkeypatch.setitem(sys.modules, "llm_provider", fake_llm)
    monkeypatch.delitem(sys.modules, "generation", raising=False)
    import generation as module

    state.module = module
    yield state
    sys.modules.pop("generation", None)


def chunk(distance, doc="vacation_policy.docx", text="Employees with 2-5 years get 22 days."):
    return {"text": text, "doc_name": doc, "distance": distance}


def test_a_close_match_is_answered_with_sources_and_usage(generation):
    g = generation.module
    generation.chunks = [chunk(g.DISTANCE_THRESHOLD - 1)]

    result = g.ask("How many days after 3 years?")

    assert result["answer"] == "22 days"
    assert result["refused_by_threshold"] is False
    assert [s["doc_name"] for s in result["sources"]] == ["vacation_policy.docx"]
    assert result["usage"] == {"prompt_tokens": 120, "completion_tokens": 9, "cached": False}
    assert result["retrieved"][0]["text"].startswith("Employees")  # full text kept for the evaluation
    assert set(result["latency"]) == {"retrieval", "generation"}
    assert "How many days after 3 years?" in generation.llm_calls[0]["prompt"]
    assert "Employees with 2-5 years" in generation.llm_calls[0]["prompt"]


def test_a_distant_match_is_refused_without_calling_the_model(generation):
    g = generation.module
    generation.chunks = [chunk(g.DISTANCE_THRESHOLD + 1)]

    result = g.ask("What is the weather?")

    assert result["answer"] == g.NO_ANSWER
    assert result["refused_by_threshold"] is True
    assert result["sources"] == []
    assert result["usage"]["prompt_tokens"] is None
    assert generation.llm_calls == []


def test_no_retrieved_chunks_is_also_a_refusal(generation):
    g = generation.module
    generation.chunks = []
    result = g.ask("anything")
    assert result["refused_by_threshold"] is True and generation.llm_calls == []


def test_a_distance_exactly_at_the_threshold_is_still_answered(generation):
    """Only a distance strictly ABOVE the threshold is refused."""
    g = generation.module
    generation.chunks = [chunk(g.DISTANCE_THRESHOLD)]
    assert g.ask("q")["refused_by_threshold"] is False


def test_the_cache_setting_is_passed_to_the_model(generation):
    g = generation.module
    generation.chunks = [chunk(1.0)]
    g.ask("q", use_cache=False)
    g.ask("q")
    assert [call["use_cache"] for call in generation.llm_calls] == [False, True]
