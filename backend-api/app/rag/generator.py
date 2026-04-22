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
            model="nvidia/nemotron-nano-9b-v2:free", 
        )
    
    # combine the retrived paraghraphs into a single string and create a prompt for the LLM
    context_text = "\n\n".join([chunk["text"] for chunk in context])
    # Create the strict instructions for the AI
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a highly intelligent AI assistant. 
        Answer the user's question ONLY using the provided Context. 
        If the answer is not in the Context, say 'I cannot answer this based on the provided document.'
        
        Context:
        {context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    chain = prompt | llm
    print("Generating answer...")
    response = chain.invoke({"context": context_text, "question": question, "chat_history": chat_history})
    
    return response.content

