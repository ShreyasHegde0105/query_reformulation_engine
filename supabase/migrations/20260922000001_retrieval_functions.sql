-- ==============================================================================
-- Migration: 20260922000001_retrieval_functions.sql
-- Description: Database retrieval functions for vector similarity search and
--              keyword full-text search across enterprise document chunks.
-- Phase: 5.1 (Retrieval Database Layer)
-- ==============================================================================

-- Ensure vector extension is available in schema extensions
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;

-- ==============================================================================
-- Function 1: match_chunks_vector
-- Performs dense vector similarity search using cosine distance (<=>).
-- Leverages the existing HNSW cosine index on document_chunks.embedding.
-- ==============================================================================
CREATE OR REPLACE FUNCTION match_chunks_vector(
    query_embedding extensions.vector(384),
    match_threshold float DEFAULT 0.0,
    match_count integer DEFAULT 20,
    filter_domain text DEFAULT NULL
)
RETURNS TABLE (
    chunk_id text,
    document_id text,
    title text,
    domain text,
    chunk_index integer,
    chunk_text text,
    section text,
    token_count integer,
    similarity float,
    metadata jsonb
)
LANGUAGE sql
STABLE
AS $$
    SELECT
        c.chunk_id,
        c.document_id,
        d.title,
        d.domain,
        c.chunk_index,
        c.chunk_text,
        c.section,
        c.token_count,
        (1 - (c.embedding <=> query_embedding))::float AS similarity,
        c.metadata
    FROM document_chunks c
    JOIN documents d ON c.document_id = d.document_id
    WHERE (filter_domain IS NULL OR d.domain = filter_domain)
      AND (1 - (c.embedding <=> query_embedding)) >= match_threshold
    ORDER BY c.embedding <=> query_embedding ASC
    LIMIT match_count;
$$;

-- ==============================================================================
-- Function 2: match_chunks_keyword
-- Performs native PostgreSQL Full-Text Search (FTS) using websearch_to_tsquery
-- and ranks candidate chunks using ts_rank_cd.
-- ==============================================================================
CREATE OR REPLACE FUNCTION match_chunks_keyword(
    query_text text,
    match_count integer DEFAULT 20,
    filter_domain text DEFAULT NULL
)
RETURNS TABLE (
    chunk_id text,
    document_id text,
    title text,
    domain text,
    chunk_index integer,
    chunk_text text,
    section text,
    token_count integer,
    keyword_score float,
    metadata jsonb
)
LANGUAGE sql
STABLE
AS $$
    SELECT
        c.chunk_id,
        c.document_id,
        d.title,
        d.domain,
        c.chunk_index,
        c.chunk_text,
        c.section,
        c.token_count,
        ts_rank_cd(to_tsvector('english', c.chunk_text), websearch_to_tsquery('english', query_text))::float AS keyword_score,
        c.metadata
    FROM document_chunks c
    JOIN documents d ON c.document_id = d.document_id
    WHERE (filter_domain IS NULL OR d.domain = filter_domain)
      AND to_tsvector('english', c.chunk_text) @@ websearch_to_tsquery('english', query_text)
    ORDER BY keyword_score DESC
    LIMIT match_count;
$$;

-- ==============================================================================
-- Function Permissions
-- Grant execute permissions to backend and authenticated roles
-- ==============================================================================
GRANT EXECUTE ON FUNCTION match_chunks_vector(
    extensions.vector(384),
    float,
    integer,
    text
) TO service_role;

GRANT EXECUTE ON FUNCTION match_chunks_keyword(
    text,
    integer,
    text
) TO service_role;
