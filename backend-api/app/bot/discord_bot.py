import os
import asyncio
import discord
from app.rag.retriever import search_documents
from app.rag.generator import generate_answer
from langchain_core.messages import AIMessage, HumanMessage

# Initialize Discord Client and allow it to read message text
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# NEW: Store memory per user ID
user_memory = {}

@client.event
async def on_ready():
    print(f"--- Discord Bot Online! Logged in as {client.user} ---")

@client.event
async def on_message(message):
    # Ignore messages sent by the bot itself
    if message.author == client.user:
        return

    question = message.content[5:].strip()
        
    # Send a temporary message so the user knows it's thinking
    status_message = await message.channel.send("⏳ *Searching the database and thinking...*")
        
    try:
        # We use asyncio.to_thread because Discord is async, but our RAG functions are synchronous
        # 1. Retrieve the data
        results = await asyncio.to_thread(search_documents, question)
            
        if not results:
            await status_message.edit(content="No relevant documents found in the database.")
            return
        # -- New MEMORY LOGIC --
        # Store the question and retrieved results in memory for this user
        user_id = message.author.id
        if user_id not in user_memory:
            user_memory[user_id] = []
        history = user_memory[user_id]        
        # 2. Generate the answer
        answer = await asyncio.to_thread(generate_answer, question, results,history)
            
        # save the question and answer to memory list
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=answer))

        user_memory[user_id] = history[-5:]      # Update the memory for this user
        # 3. Edit the temporary message with the final AI response
        await status_message.edit(content=f"**Question:** {question}\n**Answer:** {answer}")
            
    except Exception as e:
        print(f"Error during AI processing: {e}")
        await status_message.edit(content="❌ *Sorry, my AI brain encountered an error.*")

if __name__ == "__main__":
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        print("ERROR: DISCORD_TOKEN is missing!")
    else:
        client.run(token)