from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Agentic RAG API"
    openai_api_key: str | None = None
    openai_model: str = "gpt-5"
    openai_timeout_seconds: float = 30.0
    database_url: str | None = None
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "agentic_rag_docs"
    max_iterations: int = 4
    max_tool_calls: int = 12
    max_evidence_chunks: int = 12
    tool_timeout_seconds: float = 15.0
    max_retries: int = 2

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)


settings = Settings()
