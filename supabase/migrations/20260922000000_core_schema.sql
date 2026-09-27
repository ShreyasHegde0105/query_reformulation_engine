-- ==============================================================================
-- Migration: 20260922000000_core_schema.sql
-- Description: Core tables for the enterprise document corpus and its chunks.
--
-- Reconstructed on 2026-09-27 from the live project's PostgREST schema
-- (column names, types, nullability, primary/foreign keys, defaults).
-- Every statement is idempotent, so this is safe to run against the existing
-- database; on a fresh project it recreates what ingestion and the retrieval
-- RPCs (20260922000001_retrieval_functions.sql) expect.
-- ==============================================================================

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;

-- ------------------------------------------------------------------------------
-- documents: one row per source document (157 in the current corpus)
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    document_id  text PRIMARY KEY,
    title        text NOT NULL,
    domain       text,
    file_name    text,
    storage_path text,
    metadata     jsonb NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------------------------
-- document_chunks: one row per retrieval chunk (696 in the current corpus).
-- `embedding` holds bge-small-en-v1.5 vectors of the contextual passage
-- (title / section / text); `chunk_text` holds the raw chunk text.
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id    text PRIMARY KEY,
    document_id text NOT NULL REFERENCES documents (document_id),
    chunk_text  text NOT NULL,
    section     text,
    chunk_index integer,
    token_count integer,
    metadata    jsonb NOT NULL,
    embedding   extensions.vector(384),
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS document_chunks_document_id_idx
    ON document_chunks (document_id);

CREATE INDEX IF NOT EXISTS documents_domain_idx
    ON documents (domain);

-- HNSW cosine index used by match_chunks_vector. The live project already has
-- one under an unknown name, so only create it when no HNSW index exists.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_indexes
        WHERE tablename = 'document_chunks'
          AND indexdef ILIKE '%USING hnsw%embedding%'
    ) THEN
        CREATE INDEX document_chunks_embedding_hnsw_idx
            ON document_chunks USING hnsw (embedding extensions.vector_cosine_ops);
    END IF;
END
$$;
