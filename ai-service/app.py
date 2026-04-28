from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
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
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

OLLAMA_URL = "http://localhost:11434/api/generate"

dimension = 384
index = faiss.IndexFlatL2(dimension)

documents = []
tokenized_corpus = []
bm25 = None

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
    logger.info("[EMBED] Text received")
    vector = embedding_model.encode(req.text).tolist()
    return {"embedding": vector}


@app.post("/generate")
def generate(req: PromptRequest):
    logger.info("[LLM] Generating response")

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": "llama3",
                "prompt": req.prompt,
                "stream": False
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    except Exception as e:
        logger.error(f"[LLM] Error: {str(e)}")
        return {"error": str(e)}


@app.post("/store")
def store(req: StoreRequest):
    global bm25, tokenized_corpus
    logger.info(f"[STORE] Incoming texts: {len(req.texts)}")

    if not req.texts:
        logger.warning("[STORE] No texts provided")
        return {"message": "No texts provided"}

    vectors = embedding_model.encode(req.texts)
    vectors = np.array(vectors).astype("float32")

    index.add(vectors)
    documents.extend(req.texts)

    tokenized_corpus = [doc.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)

    logger.info(f"[STORE] Total documents: {len(documents)}")

    return {"message": "stored successfully", "count": len(documents)}


@app.post("/search")
def search(req: SearchRequest):
    logger.info(f"[SEARCH] Query: {req.query}")

    if len(documents) == 0:
        logger.warning("[SEARCH] No documents in index")
        return {"results": []}

    query_vector = embedding_model.encode([req.query])
    query_vector = np.array(query_vector).astype("float32")

    D, I = index.search(query_vector, k=3)

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
    query_vector = np.array(query_vector).astype("float32")

    D, I = index.search(query_vector, k=k)

    retrieved_docs = [documents[i] for i in I[0] if 0 <= i < len(documents)]

    logger.info(f"[Retrieval] Retrieved {len(retrieved_docs)} docs")

    return retrieved_docs

def hybrid_retrieval_agent(query: str, k=5, alpha=0.5):
    """
    Hybrid = Vector (FAISS) + BM25
    alpha → weight for vector vs keyword
    """

    logger.info("[Hybrid] Running hybrid retrieval")

    if len(documents) == 0 or bm25 is None:
        logger.warning("[Hybrid] No documents available")
        return []

    # -------- VECTOR SEARCH --------
    query_vector = embedding_model.encode([query])
    query_vector = np.array(query_vector).astype("float32")

    D, I = index.search(query_vector, k=k)

    vector_scores = {documents[i]: float(D[0][idx]) for idx, i in enumerate(I[0]) if i < len(documents)}

    # -------- BM25 SEARCH --------
    tokenized_query = query.split()
    bm25_scores_raw = bm25.get_scores(tokenized_query)

    # Normalize BM25 scores
    bm25_scores = {}
    max_bm25 = max(bm25_scores_raw) if len(bm25_scores_raw) > 0 else 1

    for i, score in enumerate(bm25_scores_raw):
        bm25_scores[documents[i]] = score / max_bm25 if max_bm25 != 0 else 0

    # -------- COMBINE SCORES --------
    combined_scores = {}

    all_docs = set(vector_scores.keys()) | set(bm25_scores.keys())

    for doc in all_docs:
        v_score = vector_scores.get(doc, 0)
        b_score = bm25_scores.get(doc, 0)

        # NOTE: FAISS distance → lower is better → invert it
        v_score = 1 / (1 + v_score)

        combined_score = alpha * v_score + (1 - alpha) * b_score
        combined_scores[doc] = combined_score

    # -------- SORT --------
    ranked_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

    final_docs = [doc for doc, _ in ranked_docs[:k]]

    logger.info(f"[Hybrid] Final docs: {len(final_docs)}")

    return final_docs


def reranking_agent(query: str, docs: list[str], top_k=3, score_threshold=0.5):
    logger.info(f"[Re-Ranker] Input docs: {len(docs)}")

    if not docs:
        logger.warning("[Re-Ranker] No docs to rank")
        return []

    pairs = [(query, doc) for doc in docs]
    scores = reranker_model.predict(pairs)

    scored_docs = list(zip(docs, scores))

    for doc, score in scored_docs:
        logger.info(f"[Re-Ranker] Score: {score:.4f} | Doc: {doc[:80]}")

    filtered_docs = [(doc, score) for doc, score in scored_docs if score >= score_threshold]

    logger.info(f"[Re-Ranker] After threshold ({score_threshold}): {len(filtered_docs)} docs")

    if not filtered_docs:
        logger.warning("[Re-Ranker] No docs passed threshold → fallback to all")
        filtered_docs = scored_docs

    ranked_docs = sorted(filtered_docs, key=lambda x: x[1], reverse=True)

    top_docs = [doc for doc, _ in ranked_docs[:top_k]]

    logger.info(f"[Re-Ranker] Final selected docs: {len(top_docs)}")

    return top_docs


def compression_agent(query: str, docs: list[str], top_k=2, max_chars=300):
    """
    Query-aware compression using reranker.
    Keeps only most relevant docs and trims them.
    """

    logger.info("[Compression] Query-aware compression started")

    if not docs:
        return []

    # Step 1: Score docs again (fine filtering)
    pairs = [(query, doc) for doc in docs]
    scores = reranker_model.predict(pairs)

    scored_docs = list(zip(docs, scores))

    # Step 2: Sort by relevance
    ranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)

    # Step 3: Keep top_k most relevant docs
    top_docs = [doc for doc, _ in ranked_docs[:top_k]]

    logger.info(f"[Compression] Selected top {len(top_docs)} docs")

    # Step 4: Trim for token safety
    compressed_docs = []
    for doc in top_docs:
        if len(doc) > max_chars:
            compressed_docs.append(doc[:max_chars] + "...")
        else:
            compressed_docs.append(doc)

    logger.info(f"[Compression] Final compressed docs: {len(compressed_docs)}")

    return compressed_docs


def validation_agent(docs: list[str]):
    valid = len(docs) > 0
    logger.info(f"[Validation] Docs valid: {valid}")
    return valid


def generation_agent(query: str, docs: list[str]):
    logger.info(f"[Generation] Query: {query}")
    logger.info(f"[Generation] Using {len(docs)} docs")

    context = "\n".join(docs)

    prompt = f"""
You are a precise AI assistant.

Rules:
- Use ONLY the provided context
- If answer is not clearly present, say "I don't know"
- Do NOT hallucinate

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
            },
            timeout=30
        )
        response.raise_for_status()

        data = response.json()
        answer = data.get("response", "").strip()

        if not answer:
            logger.warning("[Generation] Empty response")
            return "I don't know."

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
Generate 3 different rephrased versions of the user question.

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
        },
        timeout=30
    )

    text = response.json().get("response", "")

    queries = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        line = line.lstrip("-1234567890. ").strip()
        queries.append(line)

    queries = list(dict.fromkeys(queries))[:3]

    logger.info(f"[Multi-Query] Generated: {queries}")

    return queries


def multi_retrieval_agent(queries: list[str], k=3):
    logger.info(f"[Multi-Retrieval] Running {len(queries)} queries")

    all_docs = set()

    for q in queries:
        docs = hybrid_retrieval_agent(q, k)
        all_docs.update(docs)

    logger.info(f"[Multi-Retrieval] Unique docs: {len(all_docs)}")

    return list(all_docs)

# ----------- ASK API -----------

@app.post("/ask")
def ask(req: AskRequest):

    logger.info(f"[ASK] Query: {req.query}")

    docs = hybrid_retrieval_agent(req.query, k=10)

    if not validation_agent(docs):
        logger.warning("[ASK] No docs → Multi-query triggered")

        queries = multi_query_agent(req.query)
        docs = multi_retrieval_agent(queries, k=5)

        if not docs:
            return {
                "answer": "I don't know based on available data.",
                "sources": []
            }

        docs = reranking_agent(req.query, docs, top_k=3)

        docs = compression_agent(req.query, docs)

        answer = generation_agent(req.query, docs)

        return {
            "answer": answer,
            "sources": docs,
            "note": "multi-query + reranking + compression"
        }

    docs = reranking_agent(req.query, docs, top_k=3)

    docs = compression_agent(req.query, docs)

    answer = generation_agent(req.query, docs)

    return {
        "answer": answer,
        "sources": docs
    }