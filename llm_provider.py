import os
import time
import json
import hashlib
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
CACHE_FILE = "llm_cache.json"

_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        _client = Groq(api_key=api_key)
    return _client


def _load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _cache_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def generate_answer(prompt: str, max_retries: int = 3, use_cache: bool = True) -> str:
    """
    The ONLY place in the project that calls Groq directly.
    use_cache=True during dev/testing to avoid repeat calls on identical prompts.
    Set use_cache=False for real evaluation runs where freshness matters.
    """
    cache = _load_cache() if use_cache else {}
    key = _cache_key(prompt)

    if use_cache and key in cache:
        print("(using cached response — no API call made)")
        return cache[key]

    client = get_client()

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = response.choices[0].message.content

            if use_cache:
                cache[key] = answer
                _save_cache(cache)

            return answer

        except Exception as e:
            if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Rate limited, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise