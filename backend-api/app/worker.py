import os 
from celery import Celery
from app.rag.processor import extract_and_chunk_pdf , embed_and_store

REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379/0')

# initialize Celery
celery_app = Celery(
    'ragworker', 
    broker=REDIS_URL,
    backend=REDIS_URL)

@celery_app.task
def process_pdf_task(file_name: str):
    print(f"--- STARTING AI TASK: Processing {file_name} ---")

    try:
        # Read and chunk the PDF
        chunks = extract_and_chunk_pdf(file_name)
        
        # embed the chunks and store in Qdrant
        embed_and_store(chunks)
        
        print(f"--- COMPLETED AI TASK: Processed {file_name} ---")
        return {
            "status": "success",
            "file": file_name,
            "chunks_created": len(chunks),
            "message": f"Successfully processed {file_name} and created {len(chunks)} chunks."
        }
    except Exception as e:
        print(f"--- ERROR in AI TASK: Failed to process {file_name} ---")
        print(f"Error details: {str(e)}")
        return {
            "status": "error",
            "file": file_name,
            "message": f"Failed to process {file_name}. Error: {str(e)}"
        }