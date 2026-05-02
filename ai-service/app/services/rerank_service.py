from sentence_transformers import CrossEncoder


class RerankService:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: list[str], top_k: int = 3, score_threshold: float = 0.5) -> list[str]:
        if not docs:
            return []

        pairs = [(query, doc) for doc in docs]
        scores = self.model.predict(pairs)

        scored_docs = list(zip(docs, scores))
        filtered = [(doc, score) for doc, score in scored_docs if score >= score_threshold]

        if not filtered:
            filtered = scored_docs

        ranked = sorted(filtered, key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]

    def compress(self, query: str, docs: list[str], top_k: int = 2, max_chars: int = 300) -> list[str]:
        top_docs = self.rerank(query, docs, top_k=top_k, score_threshold=0.0)

        compressed = []
        for doc in top_docs:
            compressed.append(doc[:max_chars] + "..." if len(doc) > max_chars else doc)

        return compressed