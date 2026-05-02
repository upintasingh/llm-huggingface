from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "llm-huggingface"
    env: str = "dev"
    log_level: str = "INFO"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "llama3"

    top_k: int = 5
    faiss_dimension: int = 384

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()