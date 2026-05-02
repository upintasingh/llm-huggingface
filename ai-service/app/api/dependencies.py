from app.main import rag_pipeline, retrieval_service, embedding_service, faiss_store


def get_rag_pipeline():
    return rag_pipeline


def get_retrieval_service():
    return retrieval_service


def get_embedding_service():
    return embedding_service


def get_faiss_store():
    return faiss_store