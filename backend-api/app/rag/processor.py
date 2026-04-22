import os
import uuid
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from fastembed import TextEmbedding

QDRANT_URL = os.getenv("QDRANT_URL", "http://vector_db:6333")
COLLECTION_NAME = "documents_collection"

def extract_and_chunk_file(file_path: str, filename: str):
    print(f"Reading file: {filename}")
    
    # The Omni-Loader Logic
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
    print(f"Extracted and chunked {len(chunks)} chunks from {filename}.")
    
    return chunks

def embed_and_store(chunks, filename: str):
    print(f"Connecting to Qdrant to store {len(chunks)} chunks for {filename}...")
    client = QdrantClient(url=QDRANT_URL)
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    documents = [chunk.page_content for chunk in chunks]
    
    # Injecting Metadata: Now the AI knows exactly which file this came from!
    metadata = [
        {
            "source_file": filename, 
            "page": chunk.metadata.get("page", 0),
            "text": chunk.page_content  
        } 
        for chunk in chunks
    ]
    
    embeddings = list(embedding_model.embed(documents))
    
    points = []
    for i, embedding in enumerate(embeddings):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload=metadata[i]
            )
        )
    
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Successfully embedded and stored {filename} in Qdrant!")