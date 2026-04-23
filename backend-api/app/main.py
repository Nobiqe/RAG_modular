import os
import shutil
from typing import List # <--- NEW IMPORT
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.worker import process_document_task 
from app.core.vector_db import init_db
from app.rag.retriever import search_documents
from app.rag.generator import generate_answer

UPLOAD_DIR = "/code/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title="Enhanced RAG API", version="1.10.0", lifespan=lifespan)

class SearchQuery(BaseModel):
    question: str

@app.get("/")
async def read_root(): 
    return {"status": "success", "message": "Welcome to the Universal RAG API!"}

# NEW: The Multi-File Upload Endpoint
@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    tasks = []
    saved_files = []
    
    for file in files:
        ext = file.filename.split('.')[-1].lower()
        if ext not in ['pdf', 'txt', 'docx', 'csv']:
            continue # Skip unsupported files gracefully
            
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Launch a background worker for THIS specific file
        task = process_document_task.delay(file_path, file.filename)
        
        tasks.append(task.id)
        saved_files.append(file.filename)
    
    if not saved_files:
        raise HTTPException(status_code=400, detail="No supported files were uploaded.")

    return {
        "message": f"Successfully received {len(saved_files)} files. Processing in the background!",
        "files": saved_files,
        "task_ids": tasks
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
    answer = generate_answer(query.question, results, [])

    return {
        "question": query.question,
        "answer": answer,
        "sources": results
    }

# ---  View all uploaded documents ---
@app.get("/documents")
async def list_documents():
    if not os.path.exists(UPLOAD_DIR):
        return {"documents": [], "count": 0}
        
    files = os.listdir(UPLOAD_DIR)
    return {
        "documents": files,
        "count": len(files)
    }

# ---  Delete a specific document ---
@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    # 1. Delete the physical file from the server
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        raise HTTPException(status_code=404, detail="File not found on server.")

    # 2. Delete the associated vectors from the Qdrant database
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://vector_db:6333"))
    
    client.delete(
        collection_name="documents_collection",
        points_selector=models.Filter(
            must=[
                models.FieldCondition(
                    key="source_file",
                    match=models.MatchValue(value=filename)
                )
            ]
        )
    )
    
    return {
        "status": "success", 
        "message": f"Permanently deleted '{filename}' from the file system and vector database."
    }