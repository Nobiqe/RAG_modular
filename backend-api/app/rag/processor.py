import os
import uuid
import numpy as np
from pdf2image import convert_from_path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from fastembed import TextEmbedding, SparseTextEmbedding

# --- NEW: Import our OCR engine ---
from app.rag.ocr_engine import process_image_with_llm_ocr

QDRANT_URL = os.getenv("QDRANT_URL", "http://vector_db:6333")
COLLECTION_NAME = "documents_collection"

def extract_and_chunk_file(file_path: str, filename: str):
    print(f"Reading file: {filename}")
    ext = filename.split('.')[-1].lower()
    
    documents = []
    
    # 1. Pure Image Flow
    if ext in ['png', 'jpg', 'jpeg']:
        print(f"🖼️ Image detected. Bypassing standard loader, routing to OCR...")
        corrected_text = process_image_with_llm_ocr(file_path)
        if corrected_text:
            documents.append(Document(page_content=corrected_text, metadata={"page": 1}))
            
    # 2. Hybrid PDF Flow
    elif ext == 'pdf':
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        for i, page in enumerate(pages):
            text = page.page_content.strip()
            # If the page has fewer than 50 characters, it is likely a scanned image or empty
            if len(text) < 50:  
                print(f"📄 Page {i+1} appears blank or scanned. Triggering fallback OCR...")
                # Convert ONLY this specific page to an image
                images = convert_from_path(file_path, first_page=i+1, last_page=i+1)
                if images:
                    # Convert PIL image to numpy array for EasyOCR
                    img_array = np.array(images[0])
                    corrected_text = process_image_with_llm_ocr(img_array)
                    page.page_content = corrected_text
            
            if page.page_content.strip():
                documents.append(page)
                
    # 3. Standard Text Flows
    elif ext == 'txt':
        documents = TextLoader(file_path).load()
    elif ext == 'docx':
        documents = Docx2txtLoader(file_path).load()
    elif ext == 'csv':
        documents = CSVLoader(file_path).load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    if not documents:
        print(f"⚠️ No text could be extracted from {filename}")
        return []

    # --- Parent-Child Splitting Logic (Unchanged) ---
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    parent_chunks = parent_splitter.split_documents(documents)
    
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    advanced_chunks = []
    
    for parent in parent_chunks:
        parent_id = str(uuid.uuid4())
        parent_text = parent.page_content
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