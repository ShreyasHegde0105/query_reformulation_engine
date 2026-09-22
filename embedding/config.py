from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclasses.dataclass
class IngestionConfig:
    """Configuration for chunk embedding generation and Supabase ingestion."""

    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384
    embedding_batch_size: int = 32
    db_batch_size: int = 50
    input_chunks_path: Path = Path("Data/chunks/chunks.jsonl")
    validated_docs_path: Path = Path("Data/validated/validated_documents.json")
    local_files_only: bool = True
    dry_run: bool = False
    timeout: float = 30.0

    @classmethod
    def from_env(cls, **overrides: Any) -> IngestionConfig:
        """Construct IngestionConfig from environment variables and optional overrides."""
        url = os.getenv("SUPABASE_URL")
        # Service-role key has priority for local/backend ingestion pipelines
        key = (
            os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            or os.getenv("SUPABASE_KEY")
            or os.getenv("SUPABASE_ANON_KEY")
        )

        model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        dim = int(os.getenv("EMBEDDING_DIMENSION", "384"))
        emb_batch = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
        db_batch = int(os.getenv("UPSERT_BATCH_SIZE", "50"))
        chunks_path = Path(os.getenv("CHUNKS_PATH", "Data/chunks/chunks.jsonl"))
        val_docs_path = Path(os.getenv("VALIDATED_DOCS_PATH", "Data/validated/validated_documents.json"))

        kwargs: Dict[str, Any] = {
            "supabase_url": url,
            "supabase_key": key,
            "embedding_model_name": model_name,
            "embedding_dimension": dim,
            "embedding_batch_size": emb_batch,
            "db_batch_size": db_batch,
            "input_chunks_path": chunks_path,
            "validated_docs_path": val_docs_path,
            "local_files_only": os.getenv("HF_LOCAL_FILES_ONLY", "1").lower() in ("1", "true", "yes"),
            "dry_run": os.getenv("DRY_RUN", "0").lower() in ("1", "true", "yes"),
        }

        kwargs.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "supabase_url": self.supabase_url,
            "supabase_key_set": bool(self.supabase_key),
            "embedding_model_name": self.embedding_model_name,
            "embedding_dimension": self.embedding_dimension,
            "embedding_batch_size": self.embedding_batch_size,
            "db_batch_size": self.db_batch_size,
            "input_chunks_path": str(self.input_chunks_path),
            "validated_docs_path": str(self.validated_docs_path),
            "local_files_only": self.local_files_only,
            "dry_run": self.dry_run,
            "timeout": self.timeout,
        }
