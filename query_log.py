"""
A tiny structured query log: one JSON object per line (JSON Lines) in logs/queries.jsonl.
The path comes from QUERY_LOG_PATH and is read at call time, so tests and Docker can change it.
"""
import json
import os
import threading
from datetime import datetime, timezone

DEFAULT_PATH = "logs/queries.jsonl"
_lock = threading.Lock()


def log_path() -> str:
    return os.getenv("QUERY_LOG_PATH", DEFAULT_PATH)


def log_query(entry: dict, path: str | None = None) -> None:
    """Append one record (plus a UTC timestamp) to the log, creating the folder if needed."""
    path = path or log_path()
    record = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), **entry}
    line = json.dumps(record, ensure_ascii=False)

    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def read_log(path: str | None = None) -> list[dict]:
    """Read every record back. Missing file -> []; a corrupt line is skipped, not fatal."""
    path = path or log_path()
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records