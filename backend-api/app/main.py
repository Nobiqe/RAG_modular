from fastapi import FastAPI
from  app.worker import process_pdf_task
app = FastAPI(title="Enhanced RAG API", version="1.0.0")

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