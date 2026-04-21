import os 
from celery import Celery

REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379/0')

# initialize Celery
celery_app = Celery(
    'ragworker', 
    broker=REDIS_URL,
    backend=REDIS_URL)

@celery_app.task
def process_pdf_task(file_name: str):
    # We will add LangGraph, text chunking, and Qdrant embedding here later!
    print(f"--- FAKE AI TASK: Processing {file_name} ---")
    return {"status": "success", "file": file_name, "message": "Document embedded into vector DB!"}