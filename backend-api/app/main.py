from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
from  app.worker import process_pdf_task
from app.core.vector_db import init_db

# The lifespan context manager runs code before the server starts accepting requests
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- Server Starting Up: Initializing Database ---")
    init_db()
    yield
    print("--- Server Shutting Down ---")

app = FastAPI(title="Enhanced RAG API", version="0.1.1")

@app.get("/")
async def read_root(): 
    return {"status": "success", "message": "Welcome to the Enhanced RAG API!"}

@app.post("/upload")
def upload_document(filename: str):
    # The .delay() method is Celery's magic command to send the task to Redis
    task = process_pdf_task.delay(filename)
    
    return {
        "message": "File received. Processing in the background!",
        "filename": filename,
        "task_id": task.id
    }