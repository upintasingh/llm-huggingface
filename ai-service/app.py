from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import requests
import faiss
import numpy as np

app = FastAPI()

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

OLLAMA_URL = "http://localhost:11434/api/generate"

dimension = 384
index = faiss.IndexFlatL2(dimension)

documents = []

# ----------- Request Models -----------

class TextRequest(BaseModel):
    text: str

class PromptRequest(BaseModel):
    prompt: str

class StoreRequest(BaseModel):
    texts: list[str]

class SearchRequest(BaseModel):
    query: str

class AskRequest(BaseModel):
    query: str

# ----------- Embedding API -----------

@app.post("/embed")
def embed(req: TextRequest):
    vector = embedding_model.encode(req.text).tolist()
    return {"embedding": vector}

# ----------- LLM API -----------

@app.post("/generate")
def generate(req: PromptRequest):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "llama3",
            "prompt": req.prompt,
            "stream": False
        }
    )
    return response.json()

# ----------- Store -----------

@app.post("/store")
def store(req: StoreRequest):
    if not req.texts:
        return {"message": "No texts provided"}

    vectors = embedding_model.encode(req.texts)

    index.add(np.array(vectors).astype("float32"))
    documents.extend(req.texts)

    return {"message": "stored successfully", "count": len(documents)}

# ----------- Search -----------

@app.post("/search")
def search(req: SearchRequest):
    if len(documents) == 0:
        return {"results": []}

    query_vector = embedding_model.encode([req.query])

    D, I = index.search(np.array(query_vector).astype("float32"), k=3)

    results = [documents[i] for i in I[0] if 0 <= i < len(documents)]

    return {"results": results}

# ----------- AGENTS -----------

def retrieval_agent(query: str, k=3):
    if len(documents) == 0:
        return []

    query_vector = embedding_model.encode([query])
    D, I = index.search(np.array(query_vector).astype("float32"), k=k)

    retrieved_docs = []

    for idx in I[0]:
        if 0 <= idx < len(documents):
            retrieved_docs.append(documents[idx])

    return retrieved_docs


def validation_agent(docs: list[str]):
    return len(docs) > 0


def generation_agent(query: str, docs: list[str]):
    context = "\n".join(docs)

    prompt = f"""
You are an AI assistant.

Use ONLY the context below.
If answer not found, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            }
        )

        return response.json().get("response")

    except Exception as e:
        return f"LLM Error: {str(e)}"


def retry_agent(query: str):
    # Retry with higher K
    docs = retrieval_agent(query, k=5)

    if not docs:
        return None, []

    answer = generation_agent(query, docs)
    return answer, docs

# ----------- ASK API -----------

@app.post("/ask")
def ask(req: AskRequest):

    # Step 1: Retrieval
    docs = retrieval_agent(req.query)

    # Step 2: Validation + Retry
    if not validation_agent(docs):
        answer, retry_docs = retry_agent(req.query)

        if not answer:
            return {
                "answer": "I don't know based on available data.",
                "sources": []
            }

        return {
            "answer": answer,
            "sources": retry_docs,
            "note": "answered using retry"
        }

    # Step 3: Normal flow
    answer = generation_agent(req.query, docs)

    return {
        "answer": answer,
        "sources": docs
    }