from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import requests
import faiss
import numpy as np
import logging

# ----------- Logging Setup -----------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# ----------- App Init -----------

app = FastAPI()

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

# ----------- APIs -----------

@app.post("/embed")
def embed(req: TextRequest):
    logger.info(f"[EMBED] Text received")
    vector = embedding_model.encode(req.text).tolist()
    return {"embedding": vector}


@app.post("/generate")
def generate(req: PromptRequest):
    logger.info(f"[LLM] Generating response")

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
    logger.info(f"[STORE] Incoming texts: {len(req.texts)}")

    if not req.texts:
        logger.warning("[STORE] No texts provided")
        return {"message": "No texts provided"}

    vectors = embedding_model.encode(req.texts)

    index.add(np.array(vectors).astype("float32"))
    documents.extend(req.texts)

    logger.info(f"[STORE] Total documents: {len(documents)}")

    return {"message": "stored successfully", "count": len(documents)}


@app.post("/search")
def search(req: SearchRequest):
    logger.info(f"[SEARCH] Query: {req.query}")

    if len(documents) == 0:
        logger.warning("[SEARCH] No documents in index")
        return {"results": []}

    query_vector = embedding_model.encode([req.query])
    D, I = index.search(np.array(query_vector).astype("float32"), k=3)

    results = [documents[i] for i in I[0] if 0 <= i < len(documents)]

    logger.info(f"[SEARCH] Found {len(results)} results")

    return {"results": results}

# ----------- AGENTS -----------

def retrieval_agent(query: str, k=3):
    logger.info(f"[Retrieval] Query: {query}")

    if len(documents) == 0:
        logger.warning("[Retrieval] No documents available")
        return []

    query_vector = embedding_model.encode([query])
    D, I = index.search(np.array(query_vector).astype("float32"), k=k)

    retrieved_docs = []

    for idx in I[0]:
        if 0 <= idx < len(documents):
            retrieved_docs.append(documents[idx])

    logger.info(f"[Retrieval] Retrieved {len(retrieved_docs)} docs")

    return retrieved_docs


def validation_agent(docs: list[str]):
    valid = len(docs) > 0
    logger.info(f"[Validation] Docs valid: {valid}")
    return valid


def generation_agent(query: str, docs: list[str]):
    logger.info(f"[Generation] Query: {query}")
    logger.info(f"[Generation] Using {len(docs)} docs")

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

        answer = response.json().get("response")
        logger.info("[Generation] Success")

        return answer

    except Exception as e:
        logger.error(f"[Generation] Error: {str(e)}")
        return f"LLM Error: {str(e)}"


def retry_agent(query: str):
    logger.info("[Retry] Triggered")

    docs = retrieval_agent(query, k=5)

    if not docs:
        logger.warning("[Retry] No docs found")
        return None, []

    answer = generation_agent(query, docs)
    return answer, docs


def multi_query_agent(query: str):
    logger.info(f"[Multi-Query] Original: {query}")

    prompt = f"""
You are an AI assistant.

Generate 3 different rephrased versions of the user question
to improve document retrieval.

Original Question:
{query}

Return only the questions as a list.
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )

    text = response.json().get("response", "")

    queries = [q.strip("- ").strip() for q in text.split("\n") if q.strip()]
    queries = list(set(queries))[:3]

    logger.info(f"[Multi-Query] Generated: {queries}")

    return queries


def multi_retrieval_agent(queries: list[str], k=3):
    logger.info(f"[Multi-Retrieval] Running {len(queries)} queries")

    all_docs = set()

    for q in queries:
        docs = retrieval_agent(q, k)
        for d in docs:
            all_docs.add(d)

    logger.info(f"[Multi-Retrieval] Unique docs: {len(all_docs)}")

    return list(all_docs)

# ----------- ASK API -----------

@app.post("/ask")
def ask(req: AskRequest):

    logger.info(f"[ASK] Query: {req.query}")

    # Step 1: Normal retrieval
    docs = retrieval_agent(req.query)

    # Step 2: Multi-query retry
    if not validation_agent(docs):
        logger.warning("[ASK] No docs found → Multi-query triggered")

        queries = multi_query_agent(req.query)
        docs = multi_retrieval_agent(queries)

        if not docs:
            logger.error("[ASK] No docs found even after multi-query")

            return {
                "answer": "I don't know based on available data.",
                "sources": []
            }

        answer = generation_agent(req.query, docs)

        return {
            "answer": answer,
            "sources": docs,
            "note": "answered using multi-query retrieval"
        }

    # Step 3: Normal flow
    logger.info("[ASK] Using normal retrieval")

    answer = generation_agent(req.query, docs)

    return {
        "answer": answer,
        "sources": docs
    }