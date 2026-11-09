"""
Tests for chunking.py.
Each test checks one behaviour we want to keep, and the docstring says why it matters.
Chunk sizes are passed explicitly so the results never depend on your .env file.
"""
from chunking import chunk_text

ROWS = [
    "Years of Service | Vacation Days per Year",
    "0-2 years | 18 days",
    "2-5 years | 22 days",
    "5+ years | 27 days",
]
# This is the exact shape extraction.py produces for a Word table
TABLE = "[[TABLE_START]]\n" + "\n".join(ROWS) + "\n[[TABLE_END]]"
INTRO = (
    "FakeCorp provides paid vacation leave to all full-time employees. "
    "The number of vacation days depends on length of service, as outlined below."
)
OUTRO = (
    "Unused vacation days may be carried over to the following year, up to a maximum "
    "of 5 days. Requests must be submitted two weeks in advance."
)
DOC = "\n".join([INTRO, TABLE, OUTRO])


def test_table_is_never_split():
    """A table cut in half loses its meaning, so all rows must land in ONE chunk."""
    chunks = chunk_text(DOC, chunk_size=60, overlap=10)
    table_chunks = [c for c in chunks if "2-5 years" in c]
    assert len(table_chunks) == 1
    assert all(row in table_chunks[0] for row in ROWS)


def test_table_chunk_may_be_longer_than_chunk_size():
    """The table is kept whole even when it is longer than the size limit."""
    chunks = chunk_text(DOC, chunk_size=60, overlap=10)
    table_chunk = next(c for c in chunks if "2-5 years" in c)
    assert len(table_chunk) > 60


def test_table_markers_do_not_leak_into_chunks():
    """[[TABLE_START]] / [[TABLE_END]] are internal signals, never stored or shown."""
    chunks = chunk_text(DOC, chunk_size=60, overlap=10)
    assert all("[[TABLE" not in c for c in chunks)


def test_text_around_the_table_is_still_split():
    """Only the table is exempt from splitting: normal paragraphs still get chunked."""
    chunks = chunk_text(DOC, chunk_size=60, overlap=10)
    assert len(chunks) > 3


def test_consecutive_chunks_overlap():
    """The overlap keeps a sentence cut at a boundary visible in both neighbouring chunks."""
    text = " ".join(f"w{i:03d}" for i in range(80))
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 3
    for before, after in zip(chunks, chunks[1:]):
        assert before[-5:] in after


def test_plain_chunks_respect_the_size_limit():
    text = " ".join(f"w{i:03d}" for i in range(80))
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert all(len(c) <= 100 for c in chunks)


def test_tiny_trailing_fragment_is_dropped():
    """Regression: an earlier version produced a 1-character last chunk (pure noise)."""
    text = "abcdefghij" * 8 + "k"  # 81 characters: the last window would hold just "k"
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) == 2
    assert all(len(c) > 10 for c in chunks)


def test_empty_or_blank_text_gives_no_chunks():
    assert chunk_text("", chunk_size=60, overlap=10) == []
    assert chunk_text("   \n  ", chunk_size=60, overlap=10) == []


def test_short_text_stays_in_one_chunk():
    text = "A short policy sentence that fits easily."
    assert chunk_text(text, chunk_size=500, overlap=50) == [text]