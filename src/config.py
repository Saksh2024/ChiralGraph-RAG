"""
src/config.py
─────────────
Configuration management for ChiralGraph-RAG using Pydantic Settings.
Supports dynamic switching between in-memory/offline modes and production cloud infrastructure.
"""

import os
from pathlib import Path
from typing import Literal, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General
    PROJECT_NAME: str = "ChiralGraph-RAG"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000

    # Graph Backend: "networkx" (in-memory) or "neo4j"
    GRAPH_BACKEND: Literal["networkx", "neo4j"] = "networkx"
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    NEO4J_DATABASE: str = "neo4j"

    # Vector Backend: "memory" or "qdrant"
    VECTOR_BACKEND: Literal["memory", "qdrant"] = "memory"
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION: str = "chiral_sar_literature"

    # LLM Provider: "mock", "groq", or "openai"
    LLM_PROVIDER: Literal["mock", "groq", "openai"] = "mock"
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Cheminformatics Parameters
    ETKDG_CONFORMER_COUNT: int = 10
    FORCE_FIELD: str = "MMFF94"
    MAX_CONFORMER_STRAIN_KCAL: float = 4.0


settings = Settings()
