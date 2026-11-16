use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use std::sync::Arc;
use tower::ServiceExt;

use rag::application::ports::EmbeddingProvider;
use rag::application::service::RagService;
use rag::domain::chunking::{chunk_markdown, compute_mrr_and_hit_at_k, reciprocal_rank_fusion};
use rag::domain::models::{GraphEdge, GraphNode, IngestDocument, QueryOptions};
use rag::infrastructure::embeddings::{FallbackEmbedder, LocalHashEmbedder};
use rag::infrastructure::http::build_router;
use rag::infrastructure::mcp::handle_jsonrpc;
use rag::infrastructure::postgres::PostgresRagRepository;

#[test]
fn tdd_chunk_markdown_respects_headers_and_code_blocks() {
    let md = "# Intro\nShort intro.\n\n## Architecture\n```rust\nfn main() {}\n```\nDetails about pgvector and HNSW indexing.\n\n## Benchmarks\nLatency p50 is under 5ms.";
    let chunks = chunk_markdown(md, "README.md", 120);
    assert!(chunks.len() >= 2);
    assert!(chunks.iter().any(|c| c.content.contains("fn main() {}")));
    assert!(chunks.iter().all(|c| c.source_path == "README.md"));
}

#[test]
fn tdd_reciprocal_rank_fusion_and_eval_metrics() {
    let dense = vec![("doc_a".to_string(), 0.91), ("doc_b".to_string(), 0.85)];
    let sparse = vec![("doc_b".to_string(), 0.72), ("doc_c".to_string(), 0.40)];
    let fused = reciprocal_rank_fusion(&dense, &sparse, 60.0);
    // doc_b appears in both lists (rank 2 dense + rank 1 sparse) -> highest RRF score
    assert_eq!(fused[0].0, "doc_b");

    let queries = vec![
        (vec!["d1".to_string(), "d2".to_string(), "d3".to_string()], "d2".to_string()),
        (vec!["d9".to_string(), "d8".to_string(), "d7".to_string()], "d9".to_string()),
    ];
    let (hit3, mrr) = compute_mrr_and_hit_at_k(&queries, 3);
    assert!((hit3 - 1.0).abs() < 1e-9);
    // (0.5 + 1.0) / 2 = 0.75
    assert!((mrr - 0.75).abs() < 1e-9);
}

#[tokio::test]
async fn tdd_local_and_fallback_embedder_produce_unit_vectors() {
    let local = LocalHashEmbedder::new(768);
    let v1 = local.embed("pgvector HNSW hybrid search").await.unwrap();
    let v2 = local.embed("pgvector HNSW hybrid search").await.unwrap();
    let v3 = local.embed("completely unrelated topic").await.unwrap();
    assert_eq!(v1.len(), 768);
    assert_eq!(v1, v2);
    assert_ne!(v1, v3);
    let norm: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4);

    // Unreachable Ollama endpoint must cleanly fall back to LocalHashEmbedder
    let fb = FallbackEmbedder::new("http://127.0.0.1:59999".into(), "nomic-embed-text".into(), 768);
    let vf = fb.embed("test fallback").await.unwrap();
    assert_eq!(vf.len(), 768);
}

fn test_db_url() -> String {
    std::env::var("RAG_DATABASE_URL")
        .unwrap_or_else(|_| "postgresql://rag_admin:rag_dev_password@localhost:5433/rag_db".to_string())
}

#[tokio::test]
async fn tdd_postgres_hybrid_graph_cache_http_and_mcp_e2e() {
    let dsn = test_db_url();
    let repo = PostgresRagRepository::connect(&dsn).await.expect("Postgres must be reachable");
    repo.run_migrations().await.expect("Migrations must succeed");

    let embedder: Arc<dyn EmbeddingProvider> = Arc::new(LocalHashEmbedder::new(768));
    let service = Arc::new(RagService::new(Arc::new(repo.clone()), embedder));

    let tag = format!("tdd_{}", std::process::id());
    let doc_id = format!("{tag}::doc1");
    let node_a = format!("node_a_{tag}");
    let node_b = format!("node_b_{tag}");

    // 1. Ingest document + graph
    service
        .ingest_document(IngestDocument {
            collection: tag.clone(),
            document_id: "doc1".into(),
            source_path: "arch.md".into(),
            content: format!("Guide for {node_a} using pgvector HNSW and tsvector RRF"),
            metadata: json!({"kind": "spec"}),
        })
        .await
        .unwrap();

    service
        .ingest_graph(
            vec![
                GraphNode { id: node_a.clone(), label: node_a.clone(), node_type: "module".into(), source_repo: tag.clone(), metadata: json!({}) },
                GraphNode { id: node_b.clone(), label: node_b.clone(), node_type: "service".into(), source_repo: tag.clone(), metadata: json!({}) },
            ],
            vec![GraphEdge { source_id: node_a.clone(), target_id: node_b.clone(), relation: "calls".into(), weight: 1.0, metadata: json!({}) }],
        )
        .await
        .unwrap();

    // 2. First query (cache miss) -> finds doc + A* neighbor
    let r1 = service
        .query(QueryOptions {
            query: format!("{node_a} pgvector"),
            collection: tag.clone(),
            limit: 3,
            include_graph: true,
        })
        .await
        .unwrap();
    assert!(!r1.cache_hit);
    assert_eq!(r1.results.len(), 1);
    assert_eq!(r1.results[0].id, doc_id);
    assert!(r1.results[0].fts_score > 0.0);
    assert!(r1.graph_neighbors.iter().any(|n| n.id == node_b));

    // 3. Second identical query -> L1 cache hit
    let r2 = service
        .query(QueryOptions {
            query: format!("{node_a} pgvector"),
            collection: tag.clone(),
            limit: 3,
            include_graph: true,
        })
        .await
        .unwrap();
    assert!(r2.cache_hit);

    // 4. Axum HTTP API test
    let app = build_router(service.clone());
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["status"], "ok");

    // 5. MCP JSON-RPC 2.0 test
    let rpc_resp = handle_jsonrpc(
        &service,
        &json!({"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "rag_query", "arguments": {"query": format!("{node_a} pgvector"), "collection": tag}}})
            .to_string(),
    )
    .await;
    let rpc_val: Value = serde_json::from_str(&rpc_resp).unwrap();
    assert_eq!(rpc_val["id"], 7);
    assert!(rpc_val["result"]["results"].as_array().unwrap().len() >= 1);

    // Cleanup
    repo.delete_test_data(&tag, &[node_a, node_b]).await.unwrap();
}
