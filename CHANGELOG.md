# Changelog

All notable changes to this project will be documented in this file.
The format follows [Conventional Commits](https://www.conventionalcommits.org/).

## [0.1.0] - 2026-11-16
### Added
- Hexagonal architecture in Rust (`domain`, `application`, `infrastructure`).
- Structural Markdown chunker, Reciprocal Rank Fusion (RRF), and `Hit@K` / `MRR` metrics.
- PostgreSQL 17 + `pgvector` HNSW + `tsvector` GIN hybrid search and A* (1–2 hop) graph traversal.
- Pluggable `EmbeddingProvider` with Ollama `/api/embed` and deterministic local fallback.
- Axum REST API (`rag-server`), MCP JSON-RPC 2.0 `stdio` server (`rag-mcp`), and evaluation benchmark (`rag-eval`).
- Multi-stage `Dockerfile` and `docker-compose.yml`.
