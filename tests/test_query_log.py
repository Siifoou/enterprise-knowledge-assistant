"""Tests for query_log.py: one JSON line per question, readable back, never fatal."""
import json

from query_log import log_query, read_log


def test_log_query_writes_one_valid_json_line_with_a_timestamp(tmp_path):
    path = tmp_path / "queries.jsonl"
    log_query({"question": "How many days?", "total_time_s": 1.2}, path=str(path))

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["question"] == "How many days?"
    assert record["total_time_s"] == 1.2
    assert "T" in record["timestamp"]  # ISO format, e.g. 2026-11-10T08:00:00+00:00


def test_missing_folder_is_created(tmp_path):
    path = tmp_path / "deep" / "er" / "queries.jsonl"
    log_query({"question": "x"}, path=str(path))
    assert path.exists()


def test_records_are_appended_not_overwritten(tmp_path):
    path = str(tmp_path / "queries.jsonl")
    for i in range(3):
        log_query({"question": f"q{i}"}, path=path)
    assert [r["question"] for r in read_log(path)] == ["q0", "q1", "q2"]


def test_accents_are_kept_readable(tmp_path):
    path = tmp_path / "queries.jsonl"
    log_query({"question": "Combien de jours de congé ?"}, path=str(path))
    assert "congé" in path.read_text(encoding="utf-8")


def test_a_corrupt_line_is_skipped_not_fatal(tmp_path):
    path = tmp_path / "queries.jsonl"
    path.write_text('{"question": "ok"}\nthis is not json\n\n{"question": "ok too"}\n', encoding="utf-8")
    assert [r["question"] for r in read_log(str(path))] == ["ok", "ok too"]


def test_reading_a_missing_log_gives_an_empty_list(tmp_path):
    assert read_log(str(tmp_path / "nothing.jsonl")) == []


def test_the_path_comes_from_the_environment_at_call_time(tmp_path, monkeypatch):
    path = tmp_path / "from_env.jsonl"
    monkeypatch.setenv("QUERY_LOG_PATH", str(path))
    log_query({"question": "x"})
    assert len(read_log()) == 1 and path.exists()
