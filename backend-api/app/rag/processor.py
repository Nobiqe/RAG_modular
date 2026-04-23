import os
import uuid
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from fastembed import TextEmbedding, SparseTextEmbedding # <--- NEW IMPORT

QDRANT_URL = os.getenv("QDRANT_URL", "http://vector_db:6333")
COLLECTION_NAME = "documents_collection"

def extract_and_chunk_file(file_path: str, filename: str):
    # ... (Keep this exact function the same as before) ...
    print(f"Reading file: {filename}")
    ext = filename.split('.')[-1].lower()
    if ext == 'pdf':
        loader = PyPDFLoader(file_path)
    elif ext == 'txt':
        loader = TextLoader(file_path)
    elif ext == 'docx':
        loader = Docx2txtLoader(file_path)
    elif ext == 'csv':
        loader = CSVLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks

def embed_and_store(chunks, filename: str):
    print(f"Connecting to Qdrant to store {len(chunks)} hybrid chunks for {filename}...")
    client = QdrantClient(url=QDRANT_URL)
    
    # 1. Load BOTH AI Models
    print("Loading Dense and Sparse Models...")
    dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    
    documents = [chunk.page_content for chunk in chunks]
    
    metadata = [
        {"source_file": filename, "page": chunk.metadata.get("page", 0), "text": chunk.page_content} 
        for chunk in chunks
    ]
    
    # 2. Generate BOTH Vectors
    print("Generating mathematical vectors (Dense)...")
    dense_embeddings = list(dense_model.embed(documents))
    
    print("Generating keyword vectors (Sparse/BM25)...")
    sparse_embeddings = list(sparse_model.embed(documents))
    
    # 3. Structure the multi-vector data points
    print("Structuring data points...")
    points = []
    for i in range(len(chunks)):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                # Package both vectors under the names we defined in vector_db.py
                vector={
                    "dense": dense_embeddings[i].tolist(),
                    # FastEmbed outputs a specific object for sparse, we convert it to a dictionary
                    "sparse": {
                        "indices": sparse_embeddings[i].indices.tolist(),
                        "values": sparse_embeddings[i].values.tolist()
                    }
                },
                payload=metadata[i]
            )
        )
    
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Successfully embedded and stored {filename} in Hybrid Qdrant!")