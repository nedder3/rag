<div align="center">

# `rag` — Hybrid Vector, Full-Text & Graph A* Engine in Rust

**Domain-agnostic Hybrid Retrieval Microservice & MCP Server powered by Rust, Axum, PostgreSQL 17 (`pgvector` HNSW + `tsvector` GIN), and Reciprocal Rank Fusion**

[![Rust 1.83+](https://img.shields.io/badge/Rust-1.83+-000000?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![Axum 0.8](https://img.shields.io/badge/Axum-0.8-4B32C3?style=for-the-badge)](https://github.com/tokio-rs/axum)
[![PostgreSQL 17](https://img.shields.io/badge/PostgreSQL-17_+_pgvector-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![MCP JSON-RPC 2.0](https://img.shields.io/badge/MCP-JSON--RPC_2.0-10B981?style=for-the-badge)](https://modelcontextprotocol.io/)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-FE5196?style=for-the-badge)](https://conventionalcommits.org)

</div>

---

## Executive Summary

**`rag`** is a standalone, zero-Python retrieval engine written in **Rust** following **Hexagonal Architecture (Ports & Adapters)** and **TDD**:

- **Single-Query Hybrid RRF:** Fuses `pgvector` HNSW cosine similarity (`<=>`) and PostgreSQL `tsvector` GIN (`ts_rank_cd`) via Reciprocal Rank Fusion (`1 / (60 + rank)`) inside one transactional SQL query.
- **Graph A\* Traversal (1–2 Hops):** Expands relevant context over `rag_store.graph_nodes` and `rag_store.graph_edges` using a `BinaryHeap` priority queue.
- **Pluggable `EmbeddingProvider`:** Uses **Ollama** (`/api/embed`, e.g., `nomic-embed-text` / `bge-m3`) when available, with automatic fallback to a deterministic 768-d unit-norm local embedder for hermetic CI and offline testing.
- **Native MCP Server (`rag-mcp`):** Exposes `rag_query` and `rag_ingest` over JSON-RPC 2.0 `stdio` for IDEs and autonomous agents.
- **Fine-Tuning Trace Log (`rag_store.query_traces`):** Persists query, retrieved IDs, latency, and cache hits for downstream SFT/DPO dataset curation.

---

## Visual Architecture

```mermaid
flowchart LR
    C["HTTP Client / IDE Agent"] -->|"REST :8010"| HTTP["rag-server (Axum)"]
    C -->|"JSON-RPC stdio"| MCP["rag-mcp (stdio)"]
    HTTP --> SVC["RagService (L1 LRU Cache)"]
    MCP --> SVC
    SVC -->|"EmbeddingProvider"| EMB["Ollama /api/embed<br/>(fallback: LocalHashEmbedder)"]
    SVC -->|"sqlx PgPool"| PG[("PostgreSQL 17 (rag_store)<br/>pgvector HNSW + tsvector GIN + Graph")]
```

```mermaid
sequenceDiagram
    participant Client
    participant Svc as RagService (Rust)
    participant Cache as L1 LRU Cache
    participant PG as PostgreSQL 17 (rag_store)

    Client->>Svc: POST /v1/query {query, collection, include_graph}
    Svc->>Cache: Lookup (collection, query, limit)
    alt L1 Cache Hit
        Cache-->>Svc: Cached RRF hits
    else Cache Miss
        Svc->>Svc: Embed query (Ollama or LocalHash)
        Svc->>PG: WITH vec_hits, fts_hits SELECT ... RRF
        Svc->>PG: A* 1-2 hop traversal on graph_edges
        PG-->>Svc: Ranked chunks + graph neighbors
        Svc->>Cache: Store in LruCache
    end
    Svc->>PG: INSERT INTO rag_store.query_traces
    Svc-->>Client: 200 OK {results, graph_neighbors, latency_ms}
```

---

## API & MCP Specifications

### REST Endpoints (`rag-server`)
- `GET /health` → `{"status":"ok","documents":0}`
- `POST /v1/ingest` → `{"collection":"default","document_id":"doc_1","source_path":"README.md","content":"...","metadata":{}}`
- `POST /v1/query` → `{"query":"pgvector HNSW","collection":"auto","limit":4,"include_graph":true}`
- `POST /v1/graph/ingest` → `{"nodes":[...],"edges":[...]}`
- `GET /v1/metrics` → `{"total_queries":10,"cache_hits":4,"cache_misses":6,"avg_latency_ms":1.24}`

### MCP Tools (`rag-mcp` over `stdio`)
- `rag_query`: Hybrid `pgvector` + `tsvector` RRF + Graph A* search
- `rag_ingest`: Upsert a document into `rag_store.documents`

---

## Reproducible Build, Test & Benchmark Guide

### Docker Compose
```bash
git clone https://github.com/nedder3/rag.git
cd rag
docker compose up -d --build
```

### Local Build, TDD Suite & Evaluation Benchmark
```bash
export RAG_DATABASE_URL="postgresql://rag_admin:rag_dev_password@localhost:5433/rag_db"
cargo test
cargo run --release --bin rag-eval
```
