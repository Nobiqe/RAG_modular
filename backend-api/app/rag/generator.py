import os 
from langchain_openai  import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder

def generate_answer(question: str,context: list, chat_history: None) -> str:
    print("Initializing Llama 3 via OpenRouter...")
    #We use the OpenAI package but point it at OpenRouter!
    if chat_history is None:
        chat_history = []

    llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model=os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct") # Defaults to Llama 3 if not found 
        )
    
    # combine the retrived paraghraphs into a single string and create a prompt for the LLM
    context_text = "\n\n".join([chunk["text"] for chunk in context])
    # Create the strict instructions for the AI
    prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a highly intelligent and professional AI assistant. 
            You have access to a document's Context and the user's Chat History.
            
            CRITICAL RULES:
            1. LANGUAGE LOCK: You MUST answer in the EXACT SAME LANGUAGE as the user's prompt. If the user asks in Persian (Farsi), you MUST reply entirely in fluent Persian. Do NOT mix languages.
            2. If the user asks a question about the document, answer it using ONLY the provided Context.
            3. If the user is just chatting normally or referring to past messages, use the Chat History to answer them conversationally.
            
            Context:
            {context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

    chain = prompt | llm
    print("Generating answer...")
    response = chain.invoke({"context": context_text, "question": question, "chat_history": chat_history})
    
    return response.content

