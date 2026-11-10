"""Tests for llm_provider.py with a fake Groq client: usage reporting, the cache, and rate-limit retries."""
import json
import types

import pytest

import llm_provider


def response(text="the answer", prompt_tokens=11, completion_tokens=7):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text))],
        usage=types.SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class FakeClient:
    def __init__(self, *outcomes):
        self.calls = 0
        self._outcomes = list(outcomes)
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def setup(tmp_path, monkeypatch):
    cache_file = tmp_path / "cache.json"
    monkeypatch.setenv("LLM_CACHE_FILE", str(cache_file))
    sleeps = []
    monkeypatch.setattr(llm_provider.time, "sleep", lambda seconds: sleeps.append(seconds))

    def use(client):
        monkeypatch.setattr(llm_provider, "get_client", lambda: client)
        return client

    return types.SimpleNamespace(use=use, cache_file=cache_file, sleeps=sleeps)


def test_the_answer_comes_back_with_token_usage(setup):
    setup.use(FakeClient(response("22 days", 120, 9)))
    result = llm_provider.generate_answer_with_usage("prompt", use_cache=False)
    assert result == {"text": "22 days", "prompt_tokens": 120, "completion_tokens": 9, "cached": False}


def test_generate_answer_returns_plain_text(setup):
    setup.use(FakeClient(response("22 days")))
    assert llm_provider.generate_answer("prompt", use_cache=False) == "22 days"


def test_the_second_identical_prompt_is_served_from_the_cache(setup):
    client = setup.use(FakeClient(response("first")))
    first = llm_provider.generate_answer_with_usage("same prompt")
    second = llm_provider.generate_answer_with_usage("same prompt")

    assert client.calls == 1
    assert first["cached"] is False
    assert second == {"text": "first", "prompt_tokens": None, "completion_tokens": None, "cached": True}


def test_evaluation_mode_never_reads_or_writes_the_cache(setup):
    client = setup.use(FakeClient(response("one"), response("two")))
    llm_provider.generate_answer_with_usage("prompt", use_cache=False)
    llm_provider.generate_answer_with_usage("prompt", use_cache=False)

    assert client.calls == 2
    assert not setup.cache_file.exists()


def test_a_rate_limit_is_retried_with_waiting(setup):
    client = setup.use(FakeClient(Exception("Error 429 rate_limit_exceeded"), response("ok")))
    result = llm_provider.generate_answer_with_usage("prompt", use_cache=False)

    assert result["text"] == "ok"
    assert client.calls == 2
    assert setup.sleeps == [1]


def test_it_gives_up_after_max_retries(setup):
    client = setup.use(FakeClient(*[Exception("rate_limit")] * 3))
    with pytest.raises(Exception, match="rate_limit"):
        llm_provider.generate_answer_with_usage("prompt", max_retries=3, use_cache=False)
    assert client.calls == 3
    assert setup.sleeps == [1, 2]


def test_any_other_error_is_raised_immediately(setup):
    client = setup.use(FakeClient(ValueError("bad request")))
    with pytest.raises(ValueError):
        llm_provider.generate_answer_with_usage("prompt", use_cache=False)
    assert client.calls == 1
    assert setup.sleeps == []


def test_the_cache_file_keeps_the_old_format(setup):
    """Plain prompt-hash -> answer text, so the cache built earlier in the project still works."""
    setup.use(FakeClient(response("22 days")))
    llm_provider.generate_answer_with_usage("prompt")
    stored = json.loads(setup.cache_file.read_text(encoding="utf-8"))
    assert list(stored.values()) == ["22 days"]
