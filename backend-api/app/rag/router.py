import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def route_question(question: str) -> str:
    # We use a fast, lightweight model for routing (like Llama 3 8B)
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="meta-llama/llama-3-8b-instruct" 
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intelligent routing agent.
        Analyze the user's message and classify it into EXACTLY ONE of these three categories:
        
        1. 'chat': The user is making small talk, saying hello, expressing emotions, or asking about your identity.
        2. 'web': The user is asking about current events, real-time data, or general world facts (e.g., weather, sports scores).
        3. 'rag': The user is asking a specific question that requires looking up information from uploaded documents, PDFs, or a specific knowledge base.
        
        Return ONLY the exact word 'chat', 'web', or 'rag'. Do not include punctuation or extra text."""),
        ("human", "{question}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"question": question})
    
    # Clean up the response just in case the AI adds spaces or punctuation
    category = response.content.strip().lower()
    
    # Safety fallback: if the AI gets confused, default to searching the database
    if category not in ['chat', 'web', 'rag']:
        return 'rag'
        
    return category