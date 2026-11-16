-- Cross-process L1 cache invalidation via LISTEN/NOTIFY
CREATE OR REPLACE FUNCTION rag_store.notify_cache_invalidate() RETURNS TRIGGER AS $$
DECLARE
    target TEXT;
BEGIN
    IF TG_TABLE_NAME = 'documents' THEN
        target := COALESCE(NEW.collection, OLD.collection, 'auto');
    ELSE
        target := 'graph';
    END IF;
    PERFORM pg_notify('rag_cache_invalidate', target);
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_documents_cache_invalidate ON rag_store.documents;
CREATE TRIGGER trg_documents_cache_invalidate
    AFTER INSERT OR UPDATE OR DELETE ON rag_store.documents
    FOR EACH ROW EXECUTE FUNCTION rag_store.notify_cache_invalidate();

DROP TRIGGER IF EXISTS trg_graph_edges_cache_invalidate ON rag_store.graph_edges;
CREATE TRIGGER trg_graph_edges_cache_invalidate
    AFTER INSERT OR UPDATE OR DELETE ON rag_store.graph_edges
    FOR EACH ROW EXECUTE FUNCTION rag_store.notify_cache_invalidate();

-- SFT / DPO trace context snapshots & feedback table
ALTER TABLE rag_store.query_traces
    ADD COLUMN IF NOT EXISTS retrieved_contexts JSONB NOT NULL DEFAULT '[]'::jsonb;

CREATE TABLE IF NOT EXISTS rag_store.trace_feedback (
    id BIGSERIAL PRIMARY KEY,
    trace_id BIGINT NOT NULL REFERENCES rag_store.query_traces(id) ON DELETE CASCADE,
    role TEXT NOT NULL DEFAULT 'default',
    chosen TEXT NOT NULL,
    rejected TEXT,
    reward DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_rag_trace_feedback_trace ON rag_store.trace_feedback(trace_id);
