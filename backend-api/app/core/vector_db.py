import os 
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

QDRANT_URL = os.getenv("QDRANT_URL", "http://vector_db:6333")
COLLECTION_NAME = "documents_collection"

def get_qdrant_client():
    return QdrantClient(url=QDRANT_URL)

def init_db():
    client = get_qdrant_client()
    if not client.collection_exists(collection_name=COLLECTION_NAME):
            print(f"Creating collection '{COLLECTION_NAME}' in Qdrant...")
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists in Qdrant.")