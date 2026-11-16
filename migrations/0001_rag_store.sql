CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS rag_store;

CREATE TABLE IF NOT EXISTS rag_store.collections (
    name TEXT PRIMARY KEY,
    embedding_model TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS rag_store.documents (
    id TEXT PRIMARY KEY,
    collection TEXT NOT NULL,
    source_path TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('spanish', coalesce(content, ''))) STORED,
    embedding vector(768) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_rag_docs_collection ON rag_store.documents(collection);
CREATE INDEX IF NOT EXISTS idx_rag_docs_tsv ON rag_store.documents USING GIN (tsv);
CREATE INDEX IF NOT EXISTS idx_rag_docs_embedding ON rag_store.documents USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS rag_store.graph_nodes (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    node_type TEXT NOT NULL DEFAULT 'concept',
    source_repo TEXT NOT NULL DEFAULT 'default',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_rag_graph_nodes_label ON rag_store.graph_nodes(label);

CREATE TABLE IF NOT EXISTS rag_store.graph_edges (
    source_id TEXT NOT NULL REFERENCES rag_store.graph_nodes(id) ON DELETE CASCADE,
    target_id TEXT NOT NULL REFERENCES rag_store.graph_nodes(id) ON DELETE CASCADE,
    relation TEXT NOT NULL DEFAULT 'related',
    weight DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (source_id, target_id, relation)
);
CREATE INDEX IF NOT EXISTS idx_rag_graph_edges_target ON rag_store.graph_edges(target_id);

CREATE TABLE IF NOT EXISTS rag_store.query_traces (
    id BIGSERIAL PRIMARY KEY,
    collection TEXT NOT NULL,
    query_text TEXT NOT NULL,
    retrieved_ids TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    latency_ms DOUBLE PRECISION NOT NULL,
    cache_hit BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
