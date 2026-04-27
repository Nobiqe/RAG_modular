import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def rewrite_query(original_query: str) -> str:
    print(f"🔄 Corrective RAG: Rewriting poor query: '{original_query}'")
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model=os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct") # Defaults to Llama 3 if not found
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a search optimization expert. 
        The user provided a search query that failed to find good results in a vector database. 
        Your job is to rewrite the query to make it highly specific, clear, and keyword-rich.
        - If the original query is in Persian, write the new query in Persian.
        - If it is in English, write it in English.
        - Do NOT answer the question. ONLY output the rewritten query."""),
        ("human", "Original Query: {question}\n\nRewritten Query:")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"question": original_query})
    
    # Strip any extra quotes the AI might accidentally add
    new_query = response.content.strip().strip('"').strip("'")
    print(f"✨ New Optimized Query: '{new_query}'")
    return new_query