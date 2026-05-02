class RetrievalService:
    def __init__(self, embedding_service, faiss_store):
        self.embedding_service = embedding_service
        self.faiss_store = faiss_store

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        if not self.faiss_store.has_documents():
            return []

        query_vector = self.embedding_service.embed_query(query)
        _, indices = self.faiss_store.search(query_vector, k)

        return [
            self.faiss_store.documents[i]
            for i in indices[0]
            if 0 <= i < len(self.faiss_store.documents)
        ]

    def hybrid_retrieve(self, query: str, k: int = 5, alpha: float = 0.5) -> list[str]:
        if not self.faiss_store.has_documents() or self.faiss_store.bm25 is None:
            return []

        query_vector = self.embedding_service.embed_query(query)
        distances, indices = self.faiss_store.search(query_vector, k)

        vector_scores = {
            self.faiss_store.documents[i]: float(distances[0][idx])
            for idx, i in enumerate(indices[0]) if i < len(self.faiss_store.documents)
        }

        bm25_scores_raw = self.faiss_store.bm25.get_scores(query.split())
        max_bm25 = max(bm25_scores_raw) if len(bm25_scores_raw) > 0 else 1

        bm25_scores = {
            self.faiss_store.documents[i]: (score / max_bm25 if max_bm25 else 0)
            for i, score in enumerate(bm25_scores_raw)
        }

        combined = {}
        for doc in set(vector_scores) | set(bm25_scores):
            v = 1 / (1 + vector_scores.get(doc, 0))
            b = bm25_scores.get(doc, 0)
            combined[doc] = alpha * v + (1 - alpha) * b

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:k]]