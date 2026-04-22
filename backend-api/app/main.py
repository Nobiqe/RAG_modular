import os
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.worker import process_pdf_task
from app.core.vector_db import init_db
from app.rag.retriever import search_documents
from app.rag.generator import generate_answer

UPLOAD_DIR = "/code/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- Server Starting Up: Initializing Database ---")
    init_db()
    yield
    print("--- Server Shutting Down ---")

# Bumped version to 0.2.0 because we are adding a major new feature!
app = FastAPI(title="Enhanced RAG API", version="1.2.0", lifespan=lifespan)

# Define what the incoming search request should look like
class SearchQuery(BaseModel):
    question: str

@app.get("/")
async def read_root(): 
    return {"status": "success", "message": "Welcome to the Enhanced RAG API!"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    task = process_pdf_task.delay(file_path)
    
    return {
        "message": "File received and saved. Processing in the background!",
        "filename": file.filename,
        "task_id": task.id
    }

# NEW: The search endpoint
@app.post("/search")
def search_knowledge_base(query: SearchQuery):
    results = search_documents(query.question)
    if not results:
        return {
            "question": query.question,
            "results": [],
            "answer": "No relevant documents found in the knowledge base."
        }
    answer = generate_answer(query.question, results)

    return {
        "question": query.question,
        "answer": answer,
        "sources": results
    }