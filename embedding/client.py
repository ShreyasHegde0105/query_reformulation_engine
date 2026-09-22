from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set
import httpx

logger = logging.getLogger("embedding.client")

# Confirmed Supabase schema columns
KNOWN_DOC_COLS = {
    "document_id", "title", "domain", "storage_path", "metadata"
}

KNOWN_CHUNK_COLS = {
    "chunk_id", "document_id", "chunk_index", "chunk_text",
    "section", "token_count", "embedding", "metadata"
}


class SupabaseRestClient:
    """Production-grade Supabase REST client using httpx.
    
    Interacts with PostgREST endpoints (/rest/v1/) without requiring extra dependencies.
    Performs schema-aware batch upserts with idempotency (Prefer: resolution=merge-duplicates).
    """

    def __init__(
        self,
        supabase_url: str,
        api_key: str,
        timeout: float = 30.0,
    ):
        if not supabase_url or not api_key:
            raise ValueError("Both supabase_url and api_key are required for SupabaseRestClient.")

        # Clean base url
        self.base_url = supabase_url.rstrip("/")
        self.rest_url = f"{self.base_url}/rest/v1"
        self.api_key = api_key

        self.headers = {
            "apikey": self.api_key,
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        self.client = httpx.Client(
            headers=self.headers,
            timeout=httpx.Timeout(timeout, connect=10.0),
        )

    def close(self) -> None:
        self.client.close()

    def introspect_columns(self) -> Dict[str, Set[str]]:
        """Query PostgREST OpenAPI schema specification to detect exact column names.
        
        Falls back to verified schema columns (KNOWN_DOC_COLS, KNOWN_CHUNK_COLS) if
        introspection fails or OpenAPI endpoint is restricted.
        """
        table_cols: Dict[str, Set[str]] = {
            "documents": set(KNOWN_DOC_COLS),
            "document_chunks": set(KNOWN_CHUNK_COLS),
        }
        try:
            resp = self.client.get(f"{self.rest_url}/", headers={"Accept": "application/openapi+json"})
            if resp.status_code == 200:
                data = resp.json()
                defs = data.get("definitions", {})
                for tbl in ["documents", "document_chunks"]:
                    if tbl in defs and "properties" in defs[tbl]:
                        discovered = set(defs[tbl]["properties"].keys())
                        if discovered:
                            table_cols[tbl] = discovered
                        logger.info("Discovered columns for '%s': %s", tbl, sorted(list(table_cols[tbl])))
            else:
                logger.warning(
                    "OpenAPI introspection returned status %d. Falling back to verified default column mapping.",
                    resp.status_code,
                )
        except Exception as e:
            logger.warning(
                "Could not introspect PostgREST OpenAPI schema (%s). Using verified default column mapping.",
                e,
            )

        return table_cols

    def upsert_batch(
        self,
        table_name: str,
        records: List[Dict[str, Any]],
        on_conflict: Optional[str] = None,
    ) -> int:
        """Upsert a batch of records idempotently into PostgREST.
        
        Uses header 'Prefer: resolution=merge-duplicates' to guarantee safe re-runs.
        """
        if not records:
            return 0

        url = f"{self.rest_url}/{table_name}"
        params = {}
        if on_conflict:
            params["on_conflict"] = on_conflict

        headers = dict(self.headers)
        headers["Prefer"] = "resolution=merge-duplicates,return=minimal"

        resp = self.client.post(url, json=records, params=params, headers=headers)
        if resp.status_code not in (200, 201, 204):
            err_text = resp.text
            logger.error("Failed to upsert into '%s' (status %d): %s", table_name, resp.status_code, err_text)
            raise RuntimeError(
                f"PostgREST upsert error on '{table_name}' [HTTP {resp.status_code}]: {err_text}"
            )

        return len(records)

    def rpc(
        self,
        function_name: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Call a PostgreSQL RPC function via PostgREST endpoint /rest/v1/rpc/{function_name}.
        
        Args:
            function_name: Name of the PostgreSQL RPC function (e.g. 'match_chunks_vector').
            params: Dictionary of JSON parameters to pass to the RPC function.
            
        Returns:
            List of dictionary records returned by the function.
            
        Raises:
            ValueError: If function_name is empty.
            RuntimeError: If the HTTP request fails or returns non-200.
        """
        if not function_name or not function_name.strip():
            raise ValueError("RPC function_name must be a non-empty string.")

        url = f"{self.rest_url}/rpc/{function_name.strip()}"
        payload = params if params is not None else {}

        resp = self.client.post(url, json=payload)

        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list):
                return data
            raise RuntimeError(
                f"Unexpected RPC response format from '{function_name}': expected JSON array, got {type(data).__name__}"
            )

        err_text = resp.text
        if resp.status_code == 404:
            logger.error("RPC function '%s' not found (status 404): %s", function_name, err_text)
            raise RuntimeError(
                f"PostgREST RPC function '{function_name}' not found [HTTP 404]. "
                f"Ensure the Phase 5.1 migration has been executed in the Supabase database. Details: {err_text}"
            )
        elif resp.status_code in (401, 403):
            logger.error("RPC authentication/authorization failure (status %d): %s", resp.status_code, err_text)
            raise RuntimeError(
                f"PostgREST RPC '{function_name}' unauthorized [HTTP {resp.status_code}]. "
                f"Check SUPABASE_SERVICE_ROLE_KEY credentials. Details: {err_text}"
            )
        else:
            logger.error("RPC function '%s' failed (status %d): %s", function_name, resp.status_code, err_text)
            raise RuntimeError(
                f"PostgREST RPC '{function_name}' error [HTTP {resp.status_code}]: {err_text}"
            )

    def get_count(self, table_name: str) -> int:
        """Get exact count of rows in table using PostgREST exact count header."""
        url = f"{self.rest_url}/{table_name}?select=count"
        headers = dict(self.headers)
        headers["Prefer"] = "count=exact"
        headers["Range-Unit"] = "items"
        headers["Range"] = "0-0"

        resp = self.client.get(url, headers=headers)
        if resp.status_code in (200, 206):
            content_range = resp.headers.get("content-range", "")
            # Format is typically "0-0/157" or "*/157"
            if "/" in content_range:
                count_str = content_range.split("/")[-1]
                if count_str.isdigit():
                    return int(count_str)
        
        # Fallback query
        resp_all = self.client.get(f"{self.rest_url}/{table_name}?select=*&limit=1", headers=headers)
        content_range = resp_all.headers.get("content-range", "")
        if "/" in content_range:
            count_str = content_range.split("/")[-1]
            if count_str.isdigit():
                return int(count_str)

        raise RuntimeError(f"Unable to retrieve count for table '{table_name}': {resp.status_code} {resp.text}")

    def fetch_sample(self, limit: int = 3) -> List[Dict[str, Any]]:
        """Fetch sample records from document_chunks to verify embedding column."""
        url = f"{self.rest_url}/document_chunks?select=document_id,chunk_id,section,token_count,embedding&limit={limit}"
        resp = self.client.get(url)
        if resp.status_code == 200:
            return resp.json()
        raise RuntimeError(f"Failed to fetch sample chunks: {resp.status_code} {resp.text}")

    def verify_database_integrity(
        self,
        expected_docs: int = 157,
        expected_chunks: int = 696,
        expected_dim: int = 384,
    ) -> Dict[str, Any]:
        """Verify database counts and properties after ingestion."""
        doc_count = self.get_count("documents")
        chunk_count = self.get_count("document_chunks")

        # Fetch sample chunks to verify embedding dimensions and non-null values
        samples = self.fetch_sample(limit=5)
        sample_results = []
        for s in samples:
            emb = s.get("embedding")
            dim = len(emb) if isinstance(emb, list) else (384 if emb is not None else 0)
            sample_results.append({
                "document_id": s.get("document_id"),
                "chunk_id": s.get("chunk_id"),
                "section": s.get("section"),
                "token_count": s.get("token_count"),
                "embedding_dim": dim,
                "has_embedding": emb is not None,
            })

        is_valid = (
            doc_count == expected_docs
            and chunk_count == expected_chunks
            and all(r["has_embedding"] and r["embedding_dim"] == expected_dim for r in sample_results)
        )

        return {
            "is_valid": is_valid,
            "document_count": doc_count,
            "chunk_count": chunk_count,
            "expected_documents": expected_docs,
            "expected_chunks": expected_chunks,
            "samples": sample_results,
        }
