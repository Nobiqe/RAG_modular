import os
from qdrant_client import QdrantClient
from fastembed import TextEmbedding

QDRANT_URL = os.getenv("QDRANT_URL", "http://vector_db:6333")
COLLECTION_NAME = "documents_collection"

def search_documents(query: str, limit: int = 3):
    print(f"Searching database for: '{query}'")
    client = QdrantClient(url=QDRANT_URL)
    
    # 1. Load the exact same AI model
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # 2. Convert the user's text question into a mathematical vector
    query_vector = list(embedding_model.embed([query]))[0].tolist()
    
    # 3. Ask Qdrant to find the closest matching paragraphs using the new API
    search_response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit
    )
    
    # 4. Extract the human-readable text and scores from the '.points' array
    results = []
    for hit in search_response.points:
        results.append({
            "relevance_score": round(hit.score, 4),
            "text": hit.payload.get("text"),
            "page_number": hit.payload.get("page")
        })
        
    return results