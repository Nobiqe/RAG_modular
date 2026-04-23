import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding, SparseTextEmbedding
from sentence_transformers import CrossEncoder

QDRANT_URL = os.getenv("QDRANT_URL", "http://vector_db:6333")
COLLECTION_NAME = "documents_collection"

def search_documents(query: str, broad_limit: int = 15, final_limit: int = 3):
    print(f"Searching Multi-Lingual Hybrid database for: '{query}'")
    client = QdrantClient(url=QDRANT_URL)
    
    # 1. Load Multi-Lingual Embeddings
    dense_model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    
    dense_query_vector = list(dense_model.embed([query]))[0].tolist()
    sparse_query_vector = list(sparse_model.embed([query]))[0]
    
    search_response = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=dense_query_vector, using="dense", limit=broad_limit),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_query_vector.indices.tolist(),
                    values=sparse_query_vector.values.tolist()
                ),
                using="sparse",
                limit=broad_limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=broad_limit
    )
    
    # 2. DEDUPLICATION LOGIC
    # We only want to keep unique parent paragraphs so we don't waste the AI's time
    unique_parents = {}
    for hit in search_response.points:
        p_id = hit.payload.get("parent_id")
        if p_id not in unique_parents:
            unique_parents[p_id] = hit.payload
            
    candidate_payloads = list(unique_parents.values())
    
    if not candidate_payloads:
        return []
        
    # 3. The Multi-Lingual Teacher
    print(f"Re-ranking {len(candidate_payloads)} UNIQUE parent candidates...")
    reranker_model = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    
    pairs = [[query, payload.get("text")] for payload in candidate_payloads]
    scores = reranker_model.predict(pairs)
    
    scored_results = zip(scores, candidate_payloads)
    sorted_results = sorted(scored_results, key=lambda x: x[0], reverse=True)
    
    top_results = sorted_results[:final_limit]
    
    results = []
    for score, payload in top_results:
        results.append({
            "relevance_score": round(float(score), 4),
            "text": payload.get("text"),
            "source_file": payload.get("source_file", "unknown"),
            "page_number": payload.get("page")
        })
        
    return results