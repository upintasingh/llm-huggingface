class RagPipeline:
    def __init__(self, retrieval_service, rerank_service, generation_service):
        self.retrieval_service = retrieval_service
        self.rerank_service = rerank_service
        self.generation_service = generation_service

    def ask(self, query: str):
        docs = self.retrieval_service.hybrid_retrieve(query, k=10)

        if not docs:
            return {
                "answer": "I don't know based on available data.",
                "sources": []
            }

        docs = self.rerank_service.rerank(query, docs, top_k=3)
        docs = self.rerank_service.compress(query, docs)
        answer = self.generation_service.generate_answer(query, docs)

        return {
            "answer": answer,
            "sources": docs
        }