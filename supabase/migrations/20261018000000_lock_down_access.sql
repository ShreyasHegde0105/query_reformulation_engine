-- ==============================================================================
-- Migration: 20261018000000_lock_down_access.sql
-- Description: Restrict the corpus tables and retrieval RPCs to the backend.
--
-- Before this migration the tables had no Row Level Security and the retrieval
-- functions kept PostgreSQL's default PUBLIC EXECUTE grant, so anyone holding
-- the public anon key could read, insert or delete chunks through PostgREST.
--
-- The backend (ingestion, retrieval, API) connects with the service_role key,
-- which bypasses RLS, so it is unaffected. With RLS enabled and no policies,
-- the anon and authenticated roles see no rows. Add per-user policies here when
-- end users need direct access (e.g. document-level permissions).
--
-- Every statement is idempotent.
-- ==============================================================================

ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;

REVOKE ALL ON TABLE documents FROM anon, authenticated;
REVOKE ALL ON TABLE document_chunks FROM anon, authenticated;

REVOKE EXECUTE ON FUNCTION match_chunks_vector(
    extensions.vector(384),
    float,
    integer,
    text
) FROM PUBLIC, anon, authenticated;

REVOKE EXECUTE ON FUNCTION match_chunks_keyword(
    text,
    integer,
    text
) FROM PUBLIC, anon, authenticated;

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
