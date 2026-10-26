"""
Query the Diabetes Knowledge Base using MedCPT.

Uses MedCPT Query Encoder to encode the user's question,
then searches ChromaDB for the most relevant research
summaries from the knowledge base.

MedCPT uses asymmetric encoding:
  - Articles were encoded with MedCPT-Article-Encoder
  - Queries are encoded with MedCPT-Query-Encoder
"""

import chromadb
from transformers import AutoTokenizer, AutoModel
import torch
import requests
import json
from dotenv import load_dotenv
import os
# ==========================================
# CONFIGURATION
# ==========================================
load_dotenv()
# Replace with your free Groq API key from https://console.groq.com/keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ==========================================
# STEP 1: Load MedCPT Query Encoder
# ==========================================

QUERY_MODEL_NAME = "ncbi/MedCPT-Query-Encoder"

print("Loading MedCPT Query Encoder...")

query_tokenizer = AutoTokenizer.from_pretrained(
    QUERY_MODEL_NAME
)

query_model = AutoModel.from_pretrained(
    QUERY_MODEL_NAME
)

query_model.eval()

print("Query Encoder loaded successfully!\n")


# ==========================================
# STEP 2: Connect to ChromaDB
# ==========================================

client = chromadb.PersistentClient(
    path="./vector_db"
)

collection = client.get_collection(
    name="research_abstracts"
)

print(
    f"Connected to collection: 'research_abstracts' "
    f"({collection.count()} documents)\n"
)


# ==========================================
# STEP 3: Define Query Function
# ==========================================

def encode_query(question):
    """
    Encode a question using MedCPT Query Encoder.

    MedCPT Query Encoder expects a simple list
    of query strings (unlike the Article Encoder
    which expects [title, abstract] pairs).

    Returns:
        A list of 768 floats (the query embedding).
    """

    encoded = query_tokenizer(
        [question],
        truncation=True,
        padding=True,
        return_tensors="pt",
        max_length=512,
    )

    with torch.no_grad():
        output = query_model(**encoded)

    # Extract CLS token embedding
    query_embedding = (
        output.last_hidden_state[:, 0, :]
        .squeeze()
        .numpy()
        .tolist()
    )

    return query_embedding


def search(question, top_k=3):
    """
    Search the knowledge base for documents
    relevant to the given question.

    Args:
        question: The user's question string.
        top_k:    Number of top results to return.

    Returns:
        ChromaDB query results dict with keys:
        ids, documents, metadatas, distances.
    """

    query_embedding = encode_query(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    return results

 

def display_results(question, results):
    """
    Pretty-print the search results.
    """

    Source_Files = []

    print("=" * 60)
    print(f"QUESTION: {question}")
    print("=" * 60)

    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for rank, (doc_id, doc, meta, dist) in enumerate(
        zip(ids, docs, metas, distances), start=1
    ):
        # Cosine distance: lower = more similar
        # Similarity = 1 - distance
        similarity = 1 - dist

        print(f"\n--- Result {rank} ---")
        print(f"  Source:     {meta.get('source', 'N/A')}")
        print(f"  Similarity: {similarity:.4f}")
        print(f"  Content:")
        print(f"    {doc[:300]}...")
        if len(doc) > 300:
            print(f"    [...{len(doc) - 300} more characters]")

    print("\n" + "=" * 60)
  
def display_source_files(results):
    return [
        meta.get("source", "N/A")
        for meta in results["metadatas"][0]
    ]

   


# ==========================================
# STEP 3.5: Generate LLM Response (RAG)
# ==========================================

def generate_rag_response(question, results, api_key):
    """
    Generate an answer to the question using Groq LLM,
    passing the retrieved documents as context.
    """
    if not api_key or api_key == "YOUR_GROQ_API_KEY_HERE":
        return "\n[!] Groq API key not set. Skipping LLM generation. Please set GROQ_API_KEY in the script."
        
    print("\nGenerating response with Groq (gpt-oss-20b)...")
    
    # Combine retrieved documents into a single context string
    docs = results["documents"][0]
    context = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(docs)])
    
    # Build the prompt
    system_prompt = (
        "You are a helpful medical assistant specializing in diabetes for children and adolescents. "
        "Use ONLY the provided context to answer the user's question. "
        "If the answer cannot be found in the context, say 'I cannot find the answer in the provided documents.'"
    )
    
    user_prompt = f"Context information is below.\n---------------------\n{context}\n---------------------\n\nQuestion: {question}\nAnswer:"
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        return f"Error communicating with Groq API: {e}\nDetails: {e.response.text}"
    except Exception as e:
        return f"Error communicating with Groq API: {e}"


# ==========================================
# STEP 4: Interactive Query Loop
# ==========================================

if __name__ == "__main__":

    print("-" * 60)
    print("Diabetes Knowledge Base — Ask a Question")
    print("Type 'quit' or 'exit' to stop.")
    print("-" * 60)

    while True:

        print()
        question = input("Your question: ").strip()

        if not question:
            print("  (Please enter a question)")
            continue

        if question.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break

        results = search(question, top_k=3)
        display_results(question, results)

        # Call LLM to generate the final RAG answer
        print("=" * 60)
        print("LLM ANSWER (RAG):")
        print("-" * 60)
        answer = generate_rag_response(question, results, GROQ_API_KEY)
        print(answer)
        print("=" * 60)
        print("Sorce_Files:")
       
        print(display_source_files(results))
