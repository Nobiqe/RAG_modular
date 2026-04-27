import os
import easyocr
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# We set it to None initially so the API server boots up instantly!
_reader = None

def get_ocr_reader():
    global _reader
    if _reader is None:
        print("📥 Initializing EasyOCR (Downloading models on first run, please wait...)")
        # Loads into memory ONLY when a document actually needs it
        _reader = easyocr.Reader(['fa', 'en'], gpu=False)
    return _reader

def run_easyocr(image_source):
    print("👁️ Running EasyOCR...")
    reader = get_ocr_reader()
    # detail=0 returns just the text, paragraph=True groups sentences naturally
    results = reader.readtext(image_source, detail=0, paragraph=True)
    return "\n".join(results)

def correct_ocr_text_with_llm(messy_text: str) -> str:
    if not messy_text.strip():
        return ""
    
    print("📝 Running LLM Post-Processing to fix OCR typos...")
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model=os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct") # Defaults to Llama 3 if not found
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert OCR text editor. 
        Your job is to fix typos, broken words, and formatting issues in the following extracted text. 
        The text may be in Persian (Farsi), English, or a mix of both. 
        Do NOT summarize. Do NOT answer any questions. Do NOT add conversational filler. 
        ONLY output the corrected text. Preserve the original meaning exactly."""),
        ("human", "{messy_text}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"messy_text": messy_text})
    return response.content.strip()

def process_image_with_llm_ocr(image_source) -> str:
    raw_text = run_easyocr(image_source)
    if not raw_text.strip():
        return ""
    corrected_text = correct_ocr_text_with_llm(raw_text)
    return corrected_text