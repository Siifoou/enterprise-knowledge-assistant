import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from models import SessionLocal, Document
from extraction import extract_text, EmptyDocumentError
from chunking import chunk_text
from embeddings_store import store_chunks, collection
from generation import ask as ask_pipeline

app = FastAPI(title="Enterprise Knowledge Assistant")

DOCUMENTS_DIR = "data/documents"
os.makedirs(DOCUMENTS_DIR, exist_ok=True)


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...), category: str = Form(None)):
    file_path = os.path.join(DOCUMENTS_DIR, file.filename)

    # Save the raw file to disk first (step 1 of the storage architecture)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    db = SessionLocal()
    doc = Document(
        name=file.filename,
        category=category,
        file_path=file_path,
        file_type=file.filename.split(".")[-1],
        indexing_status="pending",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)  # doc.id is now populated by Postgres

    # Step 2: extract, chunk, embed, store — using doc.id so deletion can find these chunks later
    try:
        text = extract_text(file_path)
        chunks = chunk_text(text)
        store_chunks(doc_id=str(doc.id), doc_name=doc.name, chunks=chunks)

        doc.indexing_status = "indexed"
        db.commit()

    except EmptyDocumentError as e:
        doc.indexing_status = "failed"
        db.commit()
        db.close()
        raise HTTPException(status_code=400, detail=str(e))

    doc_id, doc_name = doc.id, doc.name  # grab values BEFORE closing the session
    db.close()
    return {"id": doc.id, "name": doc.name, "status": "indexed"}


@app.get("/documents")
def list_documents():
    db = SessionLocal()
    docs = db.query(Document).all()
    result = [
        {"id": d.id, "name": d.name, "category": d.category, "status": d.indexing_status}
        for d in docs
    ]
    db.close()
    return result


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: int):
    db = SessionLocal()
    doc = db.query(Document).filter(Document.id == doc_id).first()

    if not doc:
        db.close()
        raise HTTPException(status_code=404, detail="Document not found")

    # Clean all 3 stores — this is the deletion requirement from the plan
    if os.path.exists(doc.file_path):
        os.remove(doc.file_path)  # 1. file

    existing = collection.get(where={"doc_id": str(doc_id)})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])  # 2. Chroma chunks

    db.delete(doc)  # 3. Postgres row
    db.commit()
    db.close()

    return {"deleted": doc_id}


@app.post("/ask")
def ask_question(question: str = Form(...)):
    result = ask_pipeline(question)
    return result