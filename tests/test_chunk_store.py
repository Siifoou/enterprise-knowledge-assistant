"""
Tests for chunk_store.py: deleting a document must leave NO orphaned chunks in Chroma.
Uses an in-memory Chroma collection and hand-made 3-number vectors, so it needs no
embedding model, no Postgres and no files: it runs in a fraction of a second.
"""
import uuid

import chromadb
import pytest

from chunk_store import (
    build_chunk_records,
    count_chunks_for_document,
    delete_chunks_for_document,
)

DOC_A_VECTORS = [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.8, 0.2, 0.0]]
DOC_B_VECTORS = [[0.0, 1.0, 0.0], [0.0, 0.9, 0.1]]


@pytest.fixture
def collection():
    client = chromadb.EphemeralClient()
    return client.create_collection(name=f"test_{uuid.uuid4().hex}")


def add_document(collection, doc_id, name, vectors):
    """Store a document the same way the app does: labels come from build_chunk_records."""
    chunks = [f"{name} chunk {i}" for i in range(len(vectors))]
    ids, metadatas = build_chunk_records(str(doc_id), name, chunks)
    collection.add(ids=ids, embeddings=vectors, documents=chunks, metadatas=metadatas)


def test_deleting_a_document_removes_all_of_its_chunks(collection):
    add_document(collection, "1", "a.docx", DOC_A_VECTORS)
    assert count_chunks_for_document(collection, "1") == 3  # stored chunks are found

    removed = delete_chunks_for_document(collection, "1")

    assert removed == 3
    assert count_chunks_for_document(collection, "1") == 0


def test_other_documents_are_left_untouched(collection):
    add_document(collection, "1", "a.docx", DOC_A_VECTORS)
    add_document(collection, "2", "b.pdf", DOC_B_VECTORS)

    delete_chunks_for_document(collection, "1")

    assert count_chunks_for_document(collection, "2") == 2
    assert collection.count() == 2


def test_deleted_document_no_longer_appears_in_search(collection):
    """The real symptom of an orphan: a deleted document still shows up in answers."""
    add_document(collection, "1", "a.docx", DOC_A_VECTORS)
    add_document(collection, "2", "b.pdf", DOC_B_VECTORS)
    delete_chunks_for_document(collection, "1")

    # search with a vector identical to document 1's first chunk
    result = collection.query(query_embeddings=[[1.0, 0.0, 0.0]], n_results=2)

    assert all(m["doc_id"] == "2" for m in result["metadatas"][0])


def test_works_with_the_integer_ids_that_postgres_uses(collection):
    """Postgres hands out integer ids, Chroma stores text: the two must still match."""
    add_document(collection, 7, "a.docx", DOC_A_VECTORS)

    removed = delete_chunks_for_document(collection, 7)  # integer, as in main.py

    assert removed == 3
    assert collection.count() == 0


def test_deleting_an_unknown_document_is_harmless(collection):
    add_document(collection, "1", "a.docx", DOC_A_VECTORS)

    assert delete_chunks_for_document(collection, "999") == 0
    assert collection.count() == 3