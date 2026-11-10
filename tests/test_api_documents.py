"""
Tests for the document endpoints in main.py (upload, list, delete), run through FastAPI's TestClient.

The real main.py, extraction, chunking and chunk_store code runs. Only the two heavy parts are
swapped for light stand-ins, so these tests need no Docker, no Postgres and no embedding model:
  - models           -> an in-memory SQLite database instead of PostgreSQL
  - embeddings_store -> an in-memory Chroma collection with fake vectors instead of the real model
  - generation       -> a stub (answering questions is not tested here)
Uploaded files go to a temporary folder, so your real data/documents is never touched.
What they do NOT test: real PostgreSQL and real embeddings (those were checked by hand).
"""
import io
import sys
import types
import uuid
from datetime import datetime
from types import SimpleNamespace

import chromadb
import pytest
from docx import Document as DocxDocument
from fastapi.testclient import TestClient
from sqlalchemy import Column, DateTime, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool

from chunk_store import build_chunk_records, count_chunks_for_document


def docx_bytes(text):
    doc = DocxDocument()
    doc.add_paragraph(text)
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


V1 = docx_bytes("Version ONE of the telework policy. It is comfortably longer than fifty characters.")
V2 = docx_bytes("Version TWO of the telework policy. It is comfortably longer than fifty characters.")
TOO_SHORT = docx_bytes("Hi")  # almost no text: the app must reject it


@pytest.fixture
def api(tmp_path, monkeypatch):
    docs_dir = tmp_path / "docs"
    monkeypatch.setenv("DOCUMENTS_DIR", str(docs_dir))

    # stand-in for models.py: same table, but SQLite in memory
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base = declarative_base()

    class Document(Base):
        __tablename__ = "documents"
        id = Column(Integer, primary_key=True, autoincrement=True)
        name = Column(String, nullable=False)
        category = Column(String, nullable=True)
        file_path = Column(String, nullable=False)
        file_type = Column(String, nullable=False)
        language = Column(String, nullable=True)
        indexing_status = Column(String, default="pending")
        uploaded_at = Column(DateTime, default=datetime.utcnow)

    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)

    fake_models = types.ModuleType("models")
    fake_models.SessionLocal = SessionLocal
    fake_models.Document = Document

    # stand-in for embeddings_store.py: real chunk labelling, fake vectors, in-memory Chroma
    collection = chromadb.EphemeralClient().create_collection(name=f"test_{uuid.uuid4().hex}")

    def store_chunks(doc_id, doc_name, chunks):
        ids, metadatas = build_chunk_records(doc_id, doc_name, chunks)
        collection.add(ids=ids, embeddings=[[1.0, 0.0, 0.0]] * len(chunks), documents=chunks, metadatas=metadatas)

    fake_store = types.ModuleType("embeddings_store")
    fake_store.collection = collection
    fake_store.store_chunks = store_chunks

    fake_generation = types.ModuleType("generation")
    fake_generation.ask = lambda question, use_cache=True: {"answer": "stub", "sources": []}

    # monkeypatch puts the real modules back after each test
    monkeypatch.setitem(sys.modules, "models", fake_models)
    monkeypatch.setitem(sys.modules, "embeddings_store", fake_store)
    monkeypatch.setitem(sys.modules, "generation", fake_generation)
    monkeypatch.delitem(sys.modules, "main", raising=False)

    import main  # imported only now, so it picks up the stand-ins

    yield SimpleNamespace(
        client=TestClient(main.app),
        collection=collection,
        docs_dir=docs_dir,
        SessionLocal=SessionLocal,
        Document=Document,
    )
    sys.modules.pop("main", None)


def upload(api, name, content):
    return api.client.post("/documents/upload", files={"file": (name, content)}, data={"category": "HR"})


def rows(api):
    db = api.SessionLocal()
    result = [(d.id, d.name, d.indexing_status) for d in db.query(api.Document).all()]
    db.close()
    return result


def test_upload_stores_the_file_the_row_and_the_chunks(api):
    response = upload(api, "policy.docx", V1)

    assert response.status_code == 200
    doc_id = response.json()["id"]
    assert (api.docs_dir / "policy.docx").exists()                    # store 1: file
    assert rows(api) == [(doc_id, "policy.docx", "indexed")]          # store 2: database row
    assert count_chunks_for_document(api.collection, doc_id) > 0      # store 3: Chroma chunks


def test_delete_cleans_all_three_stores(api):
    doc_id = upload(api, "policy.docx", V1).json()["id"]

    assert api.client.delete(f"/documents/{doc_id}").status_code == 200

    assert not (api.docs_dir / "policy.docx").exists()
    assert rows(api) == []
    assert count_chunks_for_document(api.collection, doc_id) == 0


def test_uploading_the_same_filename_twice_is_refused(api):
    """Regression: the second upload used to overwrite the first file, and deleting either record
    then removed the file the other record still needed."""
    first = upload(api, "policy.docx", V1)
    second = upload(api, "policy.docx", V2)

    assert first.status_code == 200
    assert second.status_code == 409
    assert "already exists" in second.json()["detail"]
    assert (api.docs_dir / "policy.docx").read_bytes() == V1          # first file untouched
    assert len(rows(api)) == 1                                        # no second record
    assert count_chunks_for_document(api.collection, first.json()["id"]) > 0


def test_a_rejected_upload_is_kept_as_failed_and_blocks_the_name_until_removed(api):
    assert upload(api, "empty.docx", TOO_SHORT).status_code == 400
    failed = rows(api)
    assert len(failed) == 1 and failed[0][2] == "failed"

    assert upload(api, "empty.docx", V1).status_code == 409           # name still taken
    assert api.client.delete(f"/documents/{failed[0][0]}").status_code == 200
    assert upload(api, "empty.docx", V1).status_code == 200           # free again


def test_deleting_an_unknown_document_returns_404(api):
    assert api.client.delete("/documents/999").status_code == 404


def test_document_list_shows_uploaded_documents(api):
    upload(api, "a.docx", V1)
    upload(api, "b.docx", V2)

    names = sorted(d["name"] for d in api.client.get("/documents").json())

    assert names == ["a.docx", "b.docx"]


def test_the_filename_check_ignores_upper_and_lower_case(api):
    """On Windows 'Policy.docx' and 'policy.docx' are the same file, so they must count as a duplicate."""
    assert upload(api, "Policy.docx", V1).status_code == 200
    assert upload(api, "policy.docx", V2).status_code == 409
    assert len(rows(api)) == 1