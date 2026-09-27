from __future__ import annotations

import logging
import math
import threading
from collections import OrderedDict
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("embedding.encoder")

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class EmbeddingEncoder:
    """Production wrapper for BAAI/bge-small-en-v1.5 embedding generation."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        expected_dimension: int = 384,
        batch_size: int = 32,
        local_files_only: bool = True,
        query_cache_size: int = 1024,
    ):
        self.model_name = model_name
        self.expected_dimension = expected_dimension
        self.batch_size = batch_size
        self.local_files_only = local_files_only
        self._model: Optional[SentenceTransformer] = None
        # Query embeddings are reused: the redundancy filter and vector search
        # embed the same query variants, and users repeat queries.
        self.query_cache_size = query_cache_size
        self._query_cache: "OrderedDict[str, List[float]]" = OrderedDict()
        self._cache_lock = threading.Lock()

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model '%s'...", self.model_name)
            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    local_files_only=self.local_files_only,
                )
            except Exception as e:
                if self.local_files_only:
                    logger.warning(
                        "Failed to load '%s' from local cache (%s). Attempting online download...",
                        self.model_name,
                        e,
                    )
                    self._model = SentenceTransformer(self.model_name, local_files_only=False)
                else:
                    raise

            # Verify model dimension
            test_emb = self._model.encode("test", normalize_embeddings=True)
            actual_dim = len(test_emb)
            if actual_dim != self.expected_dimension:
                raise ValueError(
                    f"Model dimension mismatch! Expected {self.expected_dimension}, "
                    f"got {actual_dim} from '{self.model_name}'"
                )
            logger.info("Embedding model loaded successfully (dimension: %d).", actual_dim)

        return self._model

    def encode_passages(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> List[List[float]]:
        """Encode raw chunk texts without instruction prefix.
        
        Applies L2 normalization so vectors are unit-norm for cosine similarity.
        """
        if not texts:
            return []

        model = self._get_model()
        bs = batch_size or self.batch_size

        logger.info("Encoding %d passages with batch size %d...", len(texts), bs)
        raw_embs = model.encode(
            texts,
            batch_size=bs,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )

        embeddings: List[List[float]] = []
        for i, vec in enumerate(raw_embs):
            vec_list = [float(x) for x in vec]
            self._validate_embedding_vector(vec_list, index=i)
            embeddings.append(vec_list)

        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """Encode user query or reformulated query with the BGE query instruction prefix."""
        if not query or not query.strip():
            raise ValueError("Query string must be non-empty.")

        key = query.strip()
        with self._cache_lock:
            cached = self._query_cache.get(key)
            if cached is not None:
                self._query_cache.move_to_end(key)
                return list(cached)

        model = self._get_model()
        prefixed_query = f"{BGE_QUERY_INSTRUCTION}{key}"
        raw_emb = model.encode(prefixed_query, normalize_embeddings=True)
        vec_list = [float(x) for x in raw_emb]
        self._validate_embedding_vector(vec_list, index=0)

        if self.query_cache_size > 0:
            with self._cache_lock:
                self._query_cache[key] = vec_list
                self._query_cache.move_to_end(key)
                while len(self._query_cache) > self.query_cache_size:
                    self._query_cache.popitem(last=False)
        return list(vec_list)

    def _validate_embedding_vector(self, vec: List[float], index: int) -> None:
        if len(vec) != self.expected_dimension:
            raise ValueError(
                f"Embedding vector at index {index} has invalid dimension: {len(vec)} "
                f"(expected {self.expected_dimension})"
            )

        norm_sq = 0.0
        for val in vec:
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Embedding vector at index {index} contains NaN or Infinite value.")
            norm_sq += val * val

        norm = math.sqrt(norm_sq)
        if not (0.98 <= norm <= 1.02):
            raise ValueError(
                f"Embedding vector at index {index} is not unit-normalized (norm = {norm:.4f})"
            )
