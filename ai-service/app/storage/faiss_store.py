import faiss
from rank_bm25 import BM25Okapi


class FaissStore:
    def __init__(self, dimension: int = 384):
        self.index = faiss.IndexFlatL2(dimension)
        self.documents: list[str] = []
        self.tokenized_corpus: list[list[str]] = []
        self.bm25 = None

    def add_documents(self, texts: list[str], vectors):
        self.index.add(vectors)
        self.documents.extend(texts)

        self.tokenized_corpus = [doc.split() for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query_vector, k: int = 3):
        return self.index.search(query_vector, k)

    def has_documents(self) -> bool:
        return len(self.documents) > 0