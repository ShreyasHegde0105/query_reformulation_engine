from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from embedding.client import SupabaseRestClient
from retrieval.models import RetrievedChunk

logger = logging.getLogger("retrieval.keyword")


class KeywordRetriever:
    """Production keyword retriever for full-text search across enterprise document chunks.
    
    Coordinates:
      User Query -> SupabaseRestClient.rpc('match_chunks_keyword') -> List[RetrievedChunk]
      
    Uses PostgreSQL native Full-Text Search (websearch_to_tsquery + ts_rank_cd).
    Does NOT use embedding models or vector operations.
    """

    def __init__(
        self,
        client: Optional[SupabaseRestClient] = None,
    ) -> None:
        """Initialize KeywordRetriever with dependency injection support."""
        if client is not None:
            self.client = client
        else:
            load_dotenv(override=True)
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
        filter_domain: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """Perform keyword full-text search against the Supabase database.
        
        Args:
            query: Natural language query, keyword terms, or boolean search string.
            match_count: Maximum number of candidate chunks to return (1 to 100).
            filter_domain: Optional domain string to pre-filter chunks (e.g. 'Academic Affairs').
            
        Returns:
            List of RetrievedChunk objects ordered by keyword_score descending.
            
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

        clean_domain: Optional[str] = None
        if filter_domain is not None:
            if not isinstance(filter_domain, str):
                raise ValueError("filter_domain must be a string or None.")
            clean_domain = filter_domain.strip() if filter_domain.strip() else None

        # 3. Construct RPC payload (preserves raw search syntax for websearch_to_tsquery)
        payload = {
            "query_text": query.strip(),
            "match_count": int(match_count),
            "filter_domain": clean_domain,
        }

        # 4. Execute PostgREST RPC call
        raw_results = self.client.rpc("match_chunks_keyword", payload)

        if not isinstance(raw_results, list):
            raise RuntimeError(
                f"Unexpected RPC response format from 'match_chunks_keyword': expected list, got {type(raw_results).__name__}"
            )

        # 5. Map response to typed RetrievedChunk objects
        retrieved_chunks: List[RetrievedChunk] = []
        required_fields = (
            "chunk_id",
            "document_id",
            "title",
            "domain",
            "chunk_index",
            "chunk_text",
            "keyword_score",
        )

        for i, row in enumerate(raw_results):
            if not isinstance(row, dict):
                raise RuntimeError(f"Invalid row at index {i}: expected dict, got {type(row).__name__}")

            missing = [f for f in required_fields if f not in row or row[f] is None]
            if missing:
                raise RuntimeError(
                    f"Malformed record at index {i}: missing required field(s) {missing} in keyword RPC response."
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
                score=float(row["keyword_score"]),
                metadata=row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
                retriever_type="keyword",
            )
            retrieved_chunks.append(chunk)

        return retrieved_chunks
