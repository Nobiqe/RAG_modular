import os
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from fastembed import TextEmbedding

# Database connection settings
QDRANT_URL = os.getenv("QDRANT_URL", "http://vector_db:6333")
COLLECTION_NAME = "documents_collection"

def extract_and_chunk_pdf(file_path: str):
    print(f"Reading PDF from: {file_path}")
    
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    chunks = text_splitter.split_documents(pages)
    print(f"Extracted and chunked {len(chunks)} chunks from the PDF.")
    
    return chunks

def embed_and_store(chunks):
    print(f"Connecting to Qdrant to store {len(chunks)} chunks...")
    client = QdrantClient(url=QDRANT_URL)
    
    # 1. Initialize FastEmbed directly
    print("Loading FastEmbed Model...")
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    documents = [chunk.page_content for chunk in chunks]
    
    # We save the actual paragraph text inside the 'payload' so the AI can read it later!
    metadata = [
        {
            "source": chunk.metadata.get("source", "unknown"), 
            "page": chunk.metadata.get("page", 0),
            "text": chunk.page_content  
        } 
        for chunk in chunks
    ]
    
    # 2. Generate the mathematical vectors (converting English to arrays of 384 numbers)
    print("Generating mathematical vectors (this takes a moment)...")
    embeddings = list(embedding_model.embed(documents))
    
    # 3. Structure the data points for the database
    print("Structuring data points...")
    points = []
    for i, embedding in enumerate(embeddings):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),        # Generate a unique ID for each paragraph
                vector=embedding.tolist(),   # The 384 numbers
                payload=metadata[i]          # The original text and page number
            )
        )
    
    # 4. Upload using the modern, non-deprecated upsert method
    print("Uploading to Qdrant database...")
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print("Successfully embedded and stored in Qdrant!")