from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from embedding.client import SupabaseRestClient
from embedding.encoder import EmbeddingEncoder
from retrieval.models import RetrievedChunk

logger = logging.getLogger("retrieval.vector")


class VectorRetriever:
    """Production vector retriever for dense semantic search across enterprise chunks.
    
    Coordinates:
      User Query -> EmbeddingEncoder.embed_query() -> 384-d vector ->
      SupabaseRestClient.rpc('match_chunks_vector') -> List[RetrievedChunk]
    """

    def __init__(
        self,
        encoder: Optional[EmbeddingEncoder] = None,
        client: Optional[SupabaseRestClient] = None,
    ) -> None:
        """Initialize VectorRetriever with dependency injection support."""
        self.encoder = encoder or EmbeddingEncoder()

        if client is not None:
            self.client = client
        else:
            load_dotenv()
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
            if not url or not key:
                raise ValueError(
                    "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables are required."
                )
            self.client = SupabaseRestClient(url, key)

    def retrieve(
        self,
        query: str,
        match_count: int = 20,
        match_threshold: float = 0.0,
        filter_domain: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """Perform vector similarity search against the Supabase database.
        
        Args:
            query: Natural language query string.
            match_count: Maximum number of candidate chunks to return (1 to 100).
            match_threshold: Minimum cosine similarity score threshold (0.0 to 1.0).
            filter_domain: Optional domain string to pre-filter chunks (e.g. 'Academic Affairs').
            
        Returns:
            List of RetrievedChunk objects ordered by cosine similarity descending.
            
        Raises:
            ValueError: If query or parameters are invalid.
            RuntimeError: If RPC execution or response parsing fails.
        """
        # 1. Query validation
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        # 2. Parameter validation
        if isinstance(match_count, bool) or not isinstance(match_count, int) or match_count < 1 or match_count > 100:
            raise ValueError("match_count must be an integer between 1 and 100.")

        if isinstance(match_threshold, bool) or not isinstance(match_threshold, (int, float)) or match_threshold < 0.0 or match_threshold > 1.0:
            raise ValueError("match_threshold must be a float between 0.0 and 1.0.")

        clean_domain: Optional[str] = None
        if filter_domain is not None:
            if not isinstance(filter_domain, str):
                raise ValueError("filter_domain must be a string or None.")
            clean_domain = filter_domain.strip() if filter_domain.strip() else None

        # 3. Dense vector generation (Instruction prefixing handled internally by EmbeddingEncoder)
        query_embedding = self.encoder.embed_query(query)

        # 4. RPC payload construction
        payload = {
            "query_embedding": query_embedding,
            "match_threshold": float(match_threshold),
            "match_count": int(match_count),
            "filter_domain": clean_domain,
        }

        # 5. Execute PostgREST RPC call
        raw_results = self.client.rpc("match_chunks_vector", payload)

        if not isinstance(raw_results, list):
            raise RuntimeError(
                f"Unexpected RPC response format from 'match_chunks_vector': expected list, got {type(raw_results).__name__}"
            )

        # 6. Map response to typed RetrievedChunk objects
        retrieved_chunks: List[RetrievedChunk] = []
        required_fields = (
            "chunk_id",
            "document_id",
            "title",
            "domain",
            "chunk_index",
            "chunk_text",
            "similarity",
        )

        for i, row in enumerate(raw_results):
            if not isinstance(row, dict):
                raise RuntimeError(f"Invalid row at index {i}: expected dict, got {type(row).__name__}")

            missing = [f for f in required_fields if f not in row or row[f] is None]
            if missing:
                raise RuntimeError(
                    f"Malformed record at index {i}: missing required field(s) {missing} in vector RPC response."
                )

            chunk = RetrievedChunk(
                chunk_id=str(row["chunk_id"]),
                document_id=str(row["document_id"]),
                title=str(row["title"]),
                domain=str(row["domain"]),
                chunk_index=int(row["chunk_index"]),
                chunk_text=str(row["chunk_text"]),
                section=row.get("section"),
                token_count=row.get("token_count"),
                score=float(row["similarity"]),
                metadata=row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
                retriever_type="vector",
            )
            retrieved_chunks.append(chunk)

        return retrieved_chunks
