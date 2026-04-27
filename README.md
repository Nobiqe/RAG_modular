# 🤖 Agentic Multi-Lingual RAG API (v2.1.0)

A production-grade, headless AI microservice featuring **Self-Healing Corrective RAG (CRAG)**, **Multi-Lingual OCR**, and **Hybrid Search**. This system is designed to ingest, process, and understand complex documents in both **English and Persian (Farsi)** with high accuracy.

---

## 🚀 Key Features

* **Agentic Routing**: Automatically distinguishes between "small talk," "web search," and "document queries."
* **Multi-Lingual OCR**: Uses EasyOCR + Llama 3 Post-Processing to extract and fix text from scanned images/PDFs in Persian and English.
* **Corrective RAG (Self-Healing)**: If the first search fails, the system autonomously rewrites the query and tries again.
* **Hybrid Search**: Combines Dense (Semantic) and Sparse (Keyword) vectors using Qdrant's Reciprocal Rank Fusion (RRF).
* **Parent-Child Retrieval**: Tiny "Child" chunks are used for math matching, while massive "Parent" paragraphs are fed to the LLM for context.

---

## 📂 Project Structure & Logic

| Folder/File | Purpose |
| :--- | :--- |
| `app/main.py` | **The Gateway**: FastAPI server. Manages `/chat`, `/upload`, and `/stats`. |
| `app/rag/processor.py` | **The Ingestion Engine**: Handles file loading and Parent-Child chunking. |
| `app/rag/retriever.py` | **The Search Logic**: Handles Hybrid Search and "Teacher" AI (Cross-Encoder) scoring. |
| `app/rag/generator.py` | **The Final Output**: Combines context and memory to generate the final answer. |
| `app/rag/router.py` | **The Traffic Cop**: Categorizes user intent before running the heavy pipeline. |
| `app/rag/rewriter.py` | **The Self-Healer**: Rewrites poor queries during the Corrective RAG loop. |
| `app/rag/ocr_engine.py` | **The Vision Engine**: Handles EasyOCR and LLM-based text correction. |
| `app/worker.py` | **The Worker**: Celery background tasks for heavy file processing. |

---

## 🧠 How to Change AI Models

You can customize the intelligence of the system by swapping models in specific folders.

### 1. Changing the Embedding Models (The Math)
To change how text is converted to vectors, edit **`app/rag/processor.py`** and **`app/rag/retriever.py`**:
* **Dense (Semantic):** Look for `TextEmbedding(model_name=...)`. 
* **Sparse (Keyword):** Look for `SparseTextEmbedding(model_name=...)`.

### 2. Changing the Re-Ranker (The Teacher AI)
To change the model that "grades" the search results, edit **`app/rag/retriever.py`**:
* Look for `CrossEncoder('cross-encoder/mmarco-...')`. Using the `mmarco` series is recommended for high-accuracy Persian support.

### 3. Changing the LLM (The Brain)
To change the model used for Chat, Routing, and OCR Correction, edit **`app/rag/generator.py`**, **`app/rag/router.py`**, and **`app/rag/rewriter.py`**:
* Update the `model="meta-llama/llama-3-8b-instruct"` string to any model supported by your OpenRouter API (e.g., `gpt-4o`, `claude-3.5-sonnet`).

---

## 🌐 Server Deployment Guide

To run this on a **VPS or Cloud Server**, follow these steps:

### 1. Environment Variables
Create a `.env` file on your server to keep keys secure:
```env
OPENROUTER_API_KEY=your_key_here
REDIS_URL=redis://redis:6379/0
QDRANT_URL=http://vector_db:6333
