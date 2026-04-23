import os
from qdrant_client import QdrantClient
from qdrant_client.http import models

QDRANT_URL = os.getenv("QDRANT_URL", "http://vector_db:6333")
COLLECTION_NAME = "documents_collection"

def init_db():
    print("--- Initializing Qdrant Database ---")
    client = QdrantClient(url=QDRANT_URL)
    
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"Creating Hybrid Search collection: {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            # 1. Define the Dense Vector (Math/Meaning)
            vectors_config={
                "dense": models.VectorParams(
                    size=384, # BGE-small size
                    distance=models.Distance.COSINE
                )
            },
            # 2. Define the Sparse Vector (Keywords/BM25)
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            }
        )
        print("Database initialized successfully!")
    else:
        print("Database already exists. Ready to go!")