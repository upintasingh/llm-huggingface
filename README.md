Enterprise AI Assistant (RAG System)



A production-inspired Retrieval-Augmented Generation (RAG) system built using FastAPI, FAISS, and a local LLM (Ollama + Llama3).



This project goes beyond a basic RAG implementation by introducing a multi-agent architecture with query refinement, validation, and retry mechanisms to improve answer quality and system robustness.



Project Repository







🧠 Key Features

✅ Vector Search using FAISS

✅ Local LLM (Ollama + Llama3) — no external API dependency

✅ Multi-Agent Architecture (Retrieval, Validation, Retry, Generation)

✅ Multi-Query Retrieval (LLM-powered query expansion)

✅ Failure Handling \& Retry Logic (resilient pipeline)

✅ Context-Aware Answer Generation (grounded responses)

✅ Structured Logging for observability and debugging

✅ RESTful APIs with FastAPI



Architecture Overview

User Query

&#x20;   │

&#x20;   ▼

Retrieval Agent (FAISS)

&#x20;   │

&#x20;   ├── ✅ Docs Found → Generation Agent → Final Answer

&#x20;   │

&#x20;   └── ❌ No Docs

&#x20;           ▼

&#x20;    Multi-Query Agent (Query Expansion)

&#x20;           ▼

&#x20;    Multi-Retrieval Agent

&#x20;           │

&#x20;           ├── ✅ Docs Found → Generation Agent → Final Answer

&#x20;           │

&#x20;           └── ❌ No Docs → Fallback ("I don't know")



Tech Stack

Backend: FastAPI

Vector Database: FAISS

Embeddings: SentenceTransformers (all-MiniLM-L6-v2)

LLM: Ollama (Llama3 - local)

Language: Python



APIs

1️⃣ Store Documents

POST /store



Stores documents after chunking + embedding.

2️⃣ Ask Question (RAG Pipeline)

POST /ask



Runs full multi-agent RAG pipeline:



Retrieval

Validation

Retry (if needed)

Generation

3️⃣ Search (Vector Retrieval)

POST /search



Returns top-k similar documents from FAISS.



4️⃣ Generate (LLM Only)

POST /generate



Direct LLM call without retrieval.



🔄 System Flow

User sends query to /ask

Retrieval Agent searches FAISS

Validation Agent checks relevance

If retrieval fails → Multi-Query Agent generates variations

Multi-Retrieval improves recall

Generation Agent produces grounded response

If no relevant data → fallback response

🧠 Key Learnings

RAG is not just embeddings — query understanding is critical

Multi-query retrieval improves recall but adds latency

LLM outputs require validation and refinement

Failure handling (retry/fallback) is essential in production systems

Observability (logging) is key for debugging AI pipelines

🚀 How to Run

1\. Start Ollama

ollama serve

2\. Run LLM Model

ollama run llama3

3\. Start FastAPI Server

uvicorn app:app --reload

4\. Open API Docs

http://127.0.0.1:8000/docs

🔥 Future Improvements

🔹 Re-ranking Agent (improve precision)

🔹 Hybrid Search (BM25 + Vector)

🔹 PostgreSQL + pgvector (production-ready storage)

🔹 Redis caching (performance optimization)

🔹 Streaming responses

🔹 Observability (tracing, metrics, alerts)

🤝 Contributions



Contributions, issues, and feature requests are welcome!

Feel free to fork and improve the project 🚀



📌 Author



Upinta Singh



⭐ Support



If you found this project useful, please consider giving it a ⭐ on GitHub!

