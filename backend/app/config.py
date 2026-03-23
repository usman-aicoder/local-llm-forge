from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    # MongoDB
    mongo_url: str = "mongodb://localhost:27017"
    mongo_db_name: str = "llmplatform"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"

    # Ollama
    ollama_url: str = "http://localhost:11434"

    # Storage paths
    storage_base: str = "./storage"
    datasets_raw_dir: str = "./storage/datasets/raw"
    datasets_cleaned_dir: str = "./storage/datasets/cleaned"
    datasets_formatted_dir: str = "./storage/datasets/formatted"
    datasets_tokenized_dir: str = "./storage/datasets/tokenized"
    models_hf_dir: str = "./storage/models/hf"
    checkpoints_dir: str = "./storage/checkpoints"
    merged_models_dir: str = "./storage/merged_models"
    gguf_exports_dir: str = "./storage/gguf_exports"
    rag_documents_dir: str = "./storage/rag_documents"

    # HuggingFace
    hf_home: str = "./storage/models/hf"
    hf_token: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    def abs(self, path: str) -> Path:
        """Resolve a storage path to absolute, relative to the backend directory."""
        p = Path(path)
        if p.is_absolute():
            return p
        return (Path(__file__).parent.parent / path).resolve()


settings = Settings()
