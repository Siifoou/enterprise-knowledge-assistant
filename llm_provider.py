import hashlib
import json
import os
import time

from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

_client = None


def _cache_path() -> str:
    return os.getenv("LLM_CACHE_FILE", "llm_cache.json")


def get_client():
    global _client
    if _client is None:
        from groq import Groq  # imported here, so the automated tests do not need the groq package
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


def _load_cache() -> dict:
    path = _cache_path()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict):
    with open(_cache_path(), "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _cache_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def generate_answer_with_usage(prompt: str, max_retries: int = 3, use_cache: bool = True) -> dict:
    """
    The ONLY place in the project that calls Groq directly.
    Returns {"text", "prompt_tokens", "completion_tokens", "cached"}.
    Token counts are None for a cached answer (no API call was made).
    use_cache=True during development to avoid repeat calls on identical prompts;
    use_cache=False for real evaluation runs, where freshness matters.
    """
    cache = _load_cache() if use_cache else {}
    key = _cache_key(prompt)

    if use_cache and key in cache:
        print("(using cached response: no API call made)")
        return {"text": cache[key], "prompt_tokens": None, "completion_tokens": None, "cached": True}

    client = get_client()

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content
            usage = getattr(response, "usage", None)

            if use_cache:
                cache[key] = text
                _save_cache(cache)

            return {
                "text": text,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "cached": False,
            }

        except Exception as e:
            if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                print(f"Rate limited, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise


def generate_answer(prompt: str, max_retries: int = 3, use_cache: bool = True) -> str:
    """Same as generate_answer_with_usage, but returns only the text."""
    return generate_answer_with_usage(prompt, max_retries, use_cache)["text"]