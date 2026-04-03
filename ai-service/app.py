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

@app.post("/store")
def store(req: StoreRequest):
    vectors = embedding_model.encode(req.texts)

    index.add(np.array(vectors).astype("float32"))
    documents.extend(req.texts)

    return {"message": "stored successfully"}

@app.post("/search")
def search(req: SearchRequest):
    query_vector = embedding_model.encode([req.query])

    D, I = index.search(np.array(query_vector).astype("float32"), k=3)

    results = [documents[i] for i in I[0]]

    return {
        "results": results
    }