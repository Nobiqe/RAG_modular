import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# ... rest of the file stays the same

def extract_and_chunk_pdf(file_path:str):
    # Load the PDF document
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # 2. Configure the Chunking Strategy
    # chunk_size: How many characters per paragraph
    # chunk_overlap: How many characters to share between chunks so we don't cut a sentence in half    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
        )

    # Split the documents into chunks
    chunks = text_splitter.split_documents(pages)
    print(f"Extracted and chunked {len(chunks)} chunks from the PDF.")
    return chunks