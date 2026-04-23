import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding, SparseTextEmbedding
from sentence_transformers import CrossEncoder

QDRANT_URL = os.getenv("QDRANT_URL", "http://vector_db:6333")
COLLECTION_NAME = "documents_collection"

def search_documents(query: str, broad_limit: int = 10, final_limit: int = 3):
    print(f"Searching Hybrid database broadly for: '{query}'")
    client = QdrantClient(url=QDRANT_URL)
    
    # 1. Load BOTH AI Models
    dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    
    # 2. Embed the Query into BOTH formats
    dense_query_vector = list(dense_model.embed([query]))[0].tolist()
    sparse_query_vector = list(sparse_model.embed([query]))[0]
    
    # 3. Hybrid Search with Reciprocal Rank Fusion (RRF)
    # This automatically combines the keyword matches and the semantic meaning matches
    search_response = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=dense_query_vector,
                using="dense",
                limit=broad_limit,
            ),
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
    
    candidate_payloads = [hit.payload for hit in search_response.points]
    
    if not candidate_payloads:
        return []
        
    # 4. The Teacher: Load the Cross-Encoder AI
    print(f"Re-ranking {len(candidate_payloads)} candidates...")
    reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    pairs = [[query, payload.get("text")] for payload in candidate_payloads]
    scores = reranker_model.predict(pairs)
    
    scored_results = zip(scores, candidate_payloads)
    sorted_results = sorted(scored_results, key=lambda x: x[0], reverse=True)
    
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