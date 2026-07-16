import chromadb

# Persistent client - creates a folder that survives restarts
client = chromadb.PersistentClient(path="./jarvis_memory")

# Create or get the memory collection
collection = client.get_or_create_collection("conversations")

def remember(text, role):
    collection.add(
        documents=[text],
        ids=[f"{role}_{collection.count()}"],
        metadatas=[{"role": role}]
    )

def recall(query, n=3):
    if collection.count() == 0:
        return ""
    results = collection.query(query_texts=[query], n_results=min(n, collection.count()))
    return "\n".join(results['documents'][0])

# Test it
remember("My GPU is an RTX 5070 with 12GB VRAM", "user")
remember("I'm building a YouTube series about local AI", "user")

print("Testing recall...")
memory = recall("what hardware do I have")
print(f"Recalled: {memory}")