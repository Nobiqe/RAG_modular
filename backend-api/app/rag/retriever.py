import os
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from sentence_transformers import CrossEncoder # NEW: The Teacher AI

QDRANT_URL = os.getenv("QDRANT_URL", "http://vector_db:6333")
COLLECTION_NAME = "documents_collection"

def search_documents(query: str, broad_limit: int = 20, final_limit: int = 5):
    print(f"Searching database broadly for: '{query}'")
    client = QdrantClient(url=QDRANT_URL)
    
    # 1. Broad Search (Get top 20 candidates)
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    query_vector = list(embedding_model.embed([query]))[0].tolist()
    
    search_response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=broad_limit
    )
    
    candidate_payloads = [hit.payload for hit in search_response.points]
    
    if not candidate_payloads:
        return []
        
    # 2. The Teacher: Load the Cross-Encoder AI
    print(f"Re-ranking {len(candidate_payloads)} candidates...")
    reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # The Cross-Encoder expects pairs of [Question, Paragraph]
    pairs = [[query, payload.get("text")] for payload in candidate_payloads]
    
    # 3. Score the candidates
    scores = reranker_model.predict(pairs)
    
    # Combine the scores with the payloads and sort them highest to lowest
    scored_results = zip(scores, candidate_payloads)
    sorted_results = sorted(scored_results, key=lambda x: x[0], reverse=True)
    
    # 4. Keep only the Top 5 best answers
    top_5 = sorted_results[:final_limit]
    
    results = []
    for score, payload in top_5:
        results.append({
            "relevance_score": round(float(score), 4),
            "text": payload.get("text"),
            "source_file": payload.get("source_file", "unknown"),
            "page_number": payload.get("page")
        })
        
    return results