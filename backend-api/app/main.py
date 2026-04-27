import os
import shutil
import json
import redis
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Import LangChain message types for memory
from langchain_core.messages import HumanMessage, AIMessage

from app.worker import process_document_task
from app.core.vector_db import init_db
from app.rag.retriever import search_documents
from app.rag.generator import generate_answer
from app.rag.router import route_question # <--- NEW IMPORT

UPLOAD_DIR = "/code/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Connect FastAPI to Redis for memory storage
redis_client = redis.Redis.from_url(os.getenv('REDIS_URL', 'redis://redis:6379/0'), decode_responses=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title="Agentic RAG API", version="2.0.0", lifespan=lifespan)

# --- Define the new Chat Request Schema ---
class ChatRequest(BaseModel):
    session_id: str  # e.g., "user_123" to track their specific memory
    message: str

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


@app.post("/chat")
async def chat_with_agent(request: ChatRequest):
    # 1. Fetch Memory for this specific session
    redis_key = f"memory:{request.session_id}"
    raw_memory = redis_client.get(redis_key)
    memory_data = json.loads(raw_memory) if raw_memory else []

    history = []
    for item in memory_data:
        if item["role"] == "human":
            history.append(HumanMessage(content=item["content"]))
        else:
            history.append(AIMessage(content=item["content"]))

    # 2. The Agentic Router
    category = route_question(request.message)
    print(f"🚦 API Router decided: {category}")

    # 3. Execute the appropriate logic
    if category == 'chat':
        # Skip database, just chat
        answer = generate_answer(request.message, [], history)
    
    elif category == 'web':
        # Placeholder for future web search tool
        answer = "I need to search the web for this, but my web tool is not currently active via the API."
    
    else: # category == 'rag'
        # Run the Corrective RAG pipeline
        results = search_documents(request.message)
        if not results:
            answer = "I couldn't find any relevant information in the uploaded documents."
        else:
            answer = generate_answer(request.message, results, history)

    # 4. Save Memory (Keep only the last 6 interactions to save context window)
    memory_data.append({"role": "human", "content": request.message})
    memory_data.append({"role": "ai", "content": answer})
    redis_client.set(redis_key, json.dumps(memory_data[-6:]))

    return {
        "status": "success",
        "route_taken": category,
        "answer": answer
    }    