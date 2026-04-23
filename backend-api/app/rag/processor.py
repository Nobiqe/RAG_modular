import os
import uuid
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from fastembed import TextEmbedding, SparseTextEmbedding

QDRANT_URL = os.getenv("QDRANT_URL", "http://vector_db:6333")
COLLECTION_NAME = "documents_collection"

def extract_and_chunk_file(file_path: str, filename: str):
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
    
    # 1. Parent Splitter: Large blocks of text for context (e.g., full paragraphs)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    parent_chunks = parent_splitter.split_documents(documents)
    
    # 2. Child Splitter: Tiny chunks for highly accurate semantic search (e.g., single sentences)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    
    advanced_chunks = []
    
    for parent in parent_chunks:
        parent_id = str(uuid.uuid4())
        parent_text = parent.page_content
        
        # Chop this specific parent into children
        children = child_splitter.split_text(parent_text)
        
        for child_text in children:
            advanced_chunks.append({
                "parent_id": parent_id,
                "parent_text": parent_text, 
                "child_text": child_text,   
                "source_file": filename,
                "page": parent.metadata.get("page", 0)
            })
            
    print(f"Extracted {len(parent_chunks)} Parents and {len(advanced_chunks)} Children from {filename}.")
    return advanced_chunks

def embed_and_store(advanced_chunks, filename: str):
    print(f"Connecting to Qdrant to store {len(advanced_chunks)} chunks for {filename}...")
    client = QdrantClient(url=QDRANT_URL)
    
    # --- MULTI-LINGUAL UPGRADE ---
    print("Loading Multi-Lingual Dense and Sparse Models...")
    dense_model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    
    # We only embed the TINY child sentences to get pinpoint accuracy
    child_texts = [chunk["child_text"] for chunk in advanced_chunks]
    
    dense_embeddings = list(dense_model.embed(child_texts))
    sparse_embeddings = list(sparse_model.embed(child_texts))
    
    points = []
    for i in range(len(advanced_chunks)):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_embeddings[i].tolist(),
                    "sparse": {
                        "indices": sparse_embeddings[i].indices.tolist(),
                        "values": sparse_embeddings[i].values.tolist()
                    }
                },
                payload={
                    "source_file": filename,
                    "page": advanced_chunks[i]["page"],
                    "parent_id": advanced_chunks[i]["parent_id"],
                    "text": advanced_chunks[i]["parent_text"] 
                }
            )
        )
    
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Successfully embedded and stored {filename} using Parent-Child logic!")