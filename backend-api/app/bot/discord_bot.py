import json
import os
import asyncio
import discord
import redis 

from app.rag.retriever import search_documents
from app.rag.generator import generate_answer
from langchain_core.messages import AIMessage, HumanMessage

# Connect to Redis database 
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# Initialize Discord Client
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"--- Discord Bot Online! Logged in as {client.user} ---")

@client.event
async def on_message(message):
    # Ignore messages sent by the bot itself
    if message.author == client.user:
        return

    # FIXED: Read the exact message without chopping off 5 letters!
    question = message.content.strip()
        
    status_message = await message.channel.send("⏳ *Searching the database and thinking...*")
        
    try:
        # 1. Retrieve the data
        results = await asyncio.to_thread(search_documents, question)
            
        # -- MEMORY LOGIC --
        user_id = str(message.author.id)
        redis_key = f"memory:{user_id}"

        # Pull the existing memory for this user
        raw_memory = redis_client.get(redis_key)
        if raw_memory:
            memory_data = json.loads(raw_memory)
        else:
            memory_data = []

        history = []
        for item in memory_data:
            if item["role"] == "human":
                history.append(HumanMessage(content=item["content"]))
            else:
                history.append(AIMessage(content=item["content"]))
                
        # 2. Generate the answer
        answer = await asyncio.to_thread(generate_answer, question, results, history)

        # Update JSON list with the new question and answer
        memory_data.append({"role": "human", "content": question})
        memory_data.append({"role": "ai", "content": answer})

        # FIXED: Slice the list directly to keep the last 6 messages
        memory_data = memory_data[-6:]      

        # Save to Redis
        redis_client.set(redis_key, json.dumps(memory_data))
        
        # 3. Edit the temporary message
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