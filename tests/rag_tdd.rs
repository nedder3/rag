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
use rag::infrastructure::embeddings::{LocalHashEmbedder, OllamaEmbedder};
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
    assert_eq!(fused[0].0, "doc_b");

    let queries = vec![
        (vec!["d1".to_string(), "d2".to_string(), "d3".to_string()], "d2".to_string()),
        (vec!["d9".to_string(), "d8".to_string(), "d7".to_string()], "d9".to_string()),
    ];
    let (hit3, mrr) = compute_mrr_and_hit_at_k(&queries, 3);
    assert!((hit3 - 1.0).abs() < 1e-9);
    assert!((mrr - 0.75).abs() < 1e-9);
}

#[tokio::test]
async fn tdd_strict_ollama_embedder_fails_503_without_silent_fallback() {
    let local = LocalHashEmbedder::new(768);
    let v1 = local.embed("pgvector HNSW hybrid search").await.unwrap();
    let v2 = local.embed("pgvector HNSW hybrid search").await.unwrap();
    assert_eq!(v1, v2);
    let norm: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4);

    // Unreachable Ollama endpoint MUST error instead of corrupting the vector space
    let ollama = OllamaEmbedder::new("http://127.0.0.1:59999".into(), "nomic-embed-text".into(), 768);
    let err = ollama.embed("test strict").await.unwrap_err();
    assert!(err.contains("unavailable"));

    let repo = PostgresRagRepository::connect(&test_db_url()).await.unwrap();
    repo.run_migrations().await.unwrap();
    let svc = Arc::new(RagService::new(Arc::new(repo), Arc::new(ollama)));
    let app = build_router(svc);

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/ingest")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"collection":"strict_test","document_id":"d1","content":"hello"}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

fn test_db_url() -> String {
    std::env::var("RAG_DATABASE_URL")
        .unwrap_or_else(|_| "postgresql://rag_admin:rag_dev_password@localhost:5433/rag_db".to_string())
}

#[tokio::test]
async fn tdd_postgres_hybrid_graph_notify_invalidation_dpo_and_model_guard() {
    let dsn = test_db_url();
    let repo = PostgresRagRepository::connect(&dsn).await.expect("Postgres must be reachable");
    repo.run_migrations().await.expect("Migrations must succeed");

    let embedder: Arc<dyn EmbeddingProvider> = Arc::new(LocalHashEmbedder::new(768));
    let service = Arc::new(RagService::new(Arc::new(repo.clone()), embedder));
    let listener_handle = repo.spawn_cache_invalidation_listener(service.clone());

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

    // 2. Collection model guard: querying/ingesting with a different model must fail with mismatch
    let other_embedder: Arc<dyn EmbeddingProvider> =
        Arc::new(OllamaEmbedder::new("http://127.0.0.1:59999".into(), "nomic-embed-text".into(), 768));
    let other_svc = RagService::new(Arc::new(repo.clone()), other_embedder);
    let mismatch_err = other_svc
        .ingest_document(IngestDocument {
            collection: tag.clone(),
            document_id: "doc2".into(),
            source_path: "".into(),
            content: "incompatible".into(),
            metadata: json!({}),
        })
        .await
        .unwrap_err();
    assert!(mismatch_err.contains("mismatch"));

    // 3. First query (cache miss) -> returns trace_id + A* neighbor
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
    assert!(r1.trace_id.is_some());
    assert_eq!(r1.results[0].id, doc_id);
    assert!(r1.graph_neighbors.iter().any(|n| n.id == node_b));

    // 4. Second identical query -> L1 cache hit
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

    // 5. Cross-process L1 invalidation via Postgres LISTEN/NOTIFY
    // Simulate another process writing directly via a separate RagService (whose local clear_cache does NOT touch `service`)
    let external_svc = RagService::new(Arc::new(repo.clone()), Arc::new(LocalHashEmbedder::new(768)));
    external_svc
        .ingest_document(IngestDocument {
            collection: tag.clone(),
            document_id: "doc_ext".into(),
            source_path: "ext.md".into(),
            content: format!("External write for {node_a}"),
            metadata: json!({}),
        })
        .await
        .unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;

    let r3 = service
        .query(QueryOptions {
            query: format!("{node_a} pgvector"),
            collection: tag.clone(),
            limit: 3,
            include_graph: true,
        })
        .await
        .unwrap();
    assert!(!r3.cache_hit, "PgListener should have invalidated L1 cache on external write");

    // 6. DPO feedback + export endpoint
    let app = build_router(service.clone());
    let fb_resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/feedback")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "trace_id": r1.trace_id.unwrap(),
                        "role": "juez",
                        "chosen": "Veredicto fundamentado con cita exacta.",
                        "rejected": "Afirmación sin respaldo.",
                        "reward": 1.0
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(fb_resp.status(), StatusCode::OK);

    // Default TRL/Unsloth/Axolotl JSONL format
    let dpo_jsonl_resp = app
        .clone()
        .oneshot(Request::builder().uri("/v1/export/dpo?limit=10").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(dpo_jsonl_resp.status(), StatusCode::OK);
    let dpo_jsonl_body = String::from_utf8(dpo_jsonl_resp.into_body().collect().await.unwrap().to_bytes().to_vec()).unwrap();
    let first_line: Value = serde_json::from_str(dpo_jsonl_body.lines().next().unwrap()).unwrap();
    assert!(first_line["prompt"].as_str().unwrap().contains("Context:\n[1]"));
    assert_eq!(first_line["chosen"], "Veredicto fundamentado con cita exacta.");
    assert_eq!(first_line["rejected"], "Afirmación sin respaldo.");

    // Explicit JSON inspection format
    let dpo_resp = app
        .oneshot(Request::builder().uri("/v1/export/dpo?limit=10&format=json").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(dpo_resp.status(), StatusCode::OK);
    let dpo_body = dpo_resp.into_body().collect().await.unwrap().to_bytes();
    let dpo_json: Value = serde_json::from_slice(&dpo_body).unwrap();
    assert!(dpo_json["count"].as_u64().unwrap() >= 1);
    assert!(dpo_json["records"][0]["retrieved_contexts"].as_array().unwrap().len() >= 1);

    // 7. MCP JSON-RPC 2.0 read-only test
    let rpc_resp = handle_jsonrpc(
        &service,
        &json!({"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "rag_query", "arguments": {"query": format!("{node_a} pgvector"), "collection": tag}}})
            .to_string(),
    )
    .await;
    let rpc_val: Value = serde_json::from_str(&rpc_resp).unwrap();
    assert_eq!(rpc_val["id"], 7);
    assert!(rpc_val["result"]["results"].as_array().unwrap().len() >= 1);

    let rpc_ingest = handle_jsonrpc(
        &service,
        &json!({"jsonrpc": "2.0", "id": 8, "method": "tools/call", "params": {"name": "rag_ingest", "arguments": {"collection": tag, "document_id": "x", "content": "y"}}})
            .to_string(),
    )
    .await;
    let rpc_err: Value = serde_json::from_str(&rpc_ingest).unwrap();
    assert_eq!(rpc_err["error"]["code"], -32601);

    listener_handle.abort();
    repo.delete_test_data(&tag, &[node_a, node_b]).await.unwrap();
}
