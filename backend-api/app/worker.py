import os 
from celery import Celery
from app.rag.processor import extract_and_chunk_file, embed_and_store

REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379/0')

celery_app = Celery('ragworker', broker=REDIS_URL, backend=REDIS_URL)

# Renamed task for clarity
@celery_app.task
def process_document_task(file_path: str, filename: str):
    print(f"--- STARTING AI TASK: Processing {filename} ---")
    try:
        chunks = extract_and_chunk_file(file_path, filename)
        embed_and_store(chunks, filename)
        print(f"--- COMPLETED AI TASK: Processed {filename} ---")
        return {"status": "success", "file": filename, "chunks": len(chunks)}
    except Exception as e:
        print(f"--- ERROR: Failed to process {filename} ---")
        print(f"Error details: {str(e)}")
        return {"status": "error", "file": filename, "message": str(e)}