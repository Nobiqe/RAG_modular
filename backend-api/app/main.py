from fastapi import FastAPI

app = FastAPI(title="Enhanced RAG API", version="1.0.0")

@app.get("/")
async def read_root(): 
    return {"status": "success", "message": "Welcome to the Enhanced RAG API!"}