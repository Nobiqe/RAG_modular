import os
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from app.worker import process_pdf_task
from app.core.vector_db import init_db

# Ensure the shared uploads folder exists when the server starts
UPLOAD_DIR = "/code/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- Server Starting Up: Initializing Database ---")
    init_db()
    yield
    print("--- Server Shutting Down ---")

app = FastAPI(title="Enhanced RAG API", version="0.1.1", lifespan=lifespan)

@app.get("/")
async def read_root(): 
    return {"status": "success", "message": "Welcome to the Enhanced RAG API!"}

# NEW: We use UploadFile to accept real binary files
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # Save the file to the shared folder
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Tell the worker exactly where the file is located
    task = process_pdf_task.delay(file_path)
    
    return {
        "message": "File received and saved. Processing in the background!",
        "filename": file.filename,
        "task_id": task.id
    }