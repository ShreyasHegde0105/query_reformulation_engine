"""HTTP API for the query reformulation search engine.

Run with::

    uvicorn api.app:app --port 8000

Endpoints:
    GET  /health  – liveness plus which components are active
    POST /search  – full pipeline search; ``debug=true`` adds the pipeline trace,
                    ``answer=true`` adds an LLM answer grounded in the top results
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from reformulation.exceptions import QueryValidationError
from reformulation.search_pipeline import SearchPipeline
from reformulation.validator import MAX_QUERY_LENGTH

logger = logging.getLogger("api")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH, description="Raw user query.")
    top_k: int = Field(10, ge=1, le=50, description="Number of results to return.")
    domain: Optional[str] = Field(None, description="Restrict results to one department/domain.")
    max_variants: int = Field(5, ge=1, le=10, description="Maximum LLM reformulations.")
    debug: bool = Field(False, description="Include the pipeline trace in the response.")
    answer: bool = Field(False, description="Generate an answer from the top results (needs the LLM).")


class SearchHit(BaseModel):
    rank: int
    chunk_id: str
    document_id: str
    title: str
    section: Optional[str]
    domain: str
    score: float
    text: str


class AnswerPayload(BaseModel):
    text: str
    citations: List[str]
    answerable: bool


class SearchResponse(BaseModel):
    query: str
    results: List[SearchHit]
    degraded: bool = Field(False, description="True when part of retrieval failed; results may be incomplete.")
    answer: Optional[AnswerPayload] = None
    answer_error: Optional[str] = None
    trace: Optional[Dict[str, Any]] = None


class _NoReformulation:
    """Used when no LLM is configured: search runs on the original query only."""

    def generate_variants(self, query: str, max_variants: int = 5) -> List[str]:
        return []


def _public_answer_error(exc: Exception) -> str:
    from reformulation.llm.exceptions import LLMRateLimitError, LLMTimeoutError

    if isinstance(exc, LLMRateLimitError):
        return "Answer generation is temporarily rate-limited by the LLM provider; please retry shortly."
    if isinstance(exc, LLMTimeoutError):
        return "Answer generation timed out; please retry."
    return f"Answer generation failed ({type(exc).__name__})."


def build_pipeline() -> tuple[SearchPipeline, Optional[Any], Dict[str, Any]]:
    """Assemble the production pipeline (and answer generator, if an LLM is configured)."""
    from embedding.encoder import EmbeddingEncoder
    from reformulation.answer_generator import AnswerGenerator
    from reformulation.llm.client import LLMClient
    from reformulation.llm.config import LLMConfig
    from reformulation.reformulator import QueryReformulator
    from reformulation.semantic_similarity import SemanticSimilarity
    from retrieval.bm25_retriever import BM25Retriever
    from retrieval.hybrid_retriever import HybridRetriever
    from retrieval.vector_retriever import VectorRetriever

    info: Dict[str, Any] = {}
    answerer: Optional[AnswerGenerator] = None
    try:
        llm = LLMClient(LLMConfig.from_env())
        reformulator: Any = QueryReformulator(llm)
        answerer = AnswerGenerator(llm)
        info["reformulation"] = "llm"
        info["answers"] = "llm"
    except Exception as exc:
        logger.warning("LLM not configured (%s); searching without reformulation or answers.", exc)
        reformulator = _NoReformulation()
        info["reformulation"] = "disabled"
        info["answers"] = "disabled"

    encoder = EmbeddingEncoder()
    bm25 = BM25Retriever()
    pipeline = SearchPipeline(
        reformulator=reformulator,
        semantic_similarity=SemanticSimilarity(embed_fn=encoder.embed_query),
        hybrid_retriever=HybridRetriever(vector_retriever=VectorRetriever(encoder=encoder), keyword_retriever=bm25),
    )
    info["vector"] = "supabase"
    info["keyword"] = "bm25"
    return pipeline, answerer, info


def create_app(
    pipeline: Optional[SearchPipeline] = None,
    components: Optional[Dict[str, Any]] = None,
    answer_generator: Optional[Any] = None,
) -> FastAPI:
    """Create the API. Pass ``pipeline`` to inject one (tests); otherwise it is built at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if pipeline is None:
            app.state.pipeline, app.state.answerer, app.state.components = build_pipeline()
        else:
            app.state.pipeline, app.state.answerer = pipeline, answer_generator
            app.state.components = components or {}
        yield

    app = FastAPI(title="Query Reformulation Search API", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request) -> Dict[str, Any]:
        return {"status": "ok", "components": request.app.state.components}

    @app.post("/search", response_model=SearchResponse)
    def search(body: SearchRequest, request: Request) -> SearchResponse:
        query = body.query.strip()
        if not query:
            raise HTTPException(status_code=422, detail="query must not be blank")
        try:
            result = request.app.state.pipeline.search_detailed(
                query,
                max_variants=body.max_variants,
                filter_domain=body.domain,
                final_k=body.top_k,
            )
        except QueryValidationError as exc:
            # Only problems with the query itself are client errors; their messages are ours and safe.
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            # Anything else (including internal ValueErrors) is logged in full but not shown to the client.
            logger.exception("Search failed for %r", query)
            raise HTTPException(status_code=503, detail="Search backend unavailable") from exc

        hits = [
            SearchHit(
                rank=i,
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                title=c.title,
                section=c.section,
                domain=c.domain,
                score=c.score,
                text=c.chunk_text,
            )
            for i, c in enumerate(result.results, start=1)
        ]
        answer, answer_error = None, None
        if body.answer:
            answerer = request.app.state.answerer
            if answerer is None:
                answer_error = "Answer generation is not configured (no LLM)."
            else:
                try:
                    answer = AnswerPayload(**answerer.generate(query, result.results).to_dict())
                except Exception as exc:
                    # Full provider error goes to the log only; clients get a short, safe message.
                    logger.warning("Answer generation failed for %r: %s", query, exc)
                    answer_error = _public_answer_error(exc)

        trace = None
        if body.debug:
            trace = {key: value for key, value in result.to_dict().items() if key != "results"}
        return SearchResponse(query=query, results=hits, degraded=result.degraded, answer=answer,
                              answer_error=answer_error, trace=trace)

    return app


app = create_app()
