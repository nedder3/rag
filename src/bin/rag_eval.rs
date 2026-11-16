use serde_json::json;
use std::sync::Arc;
use std::time::Instant;

use rag::application::ports::EmbeddingProvider;
use rag::application::service::RagService;
use rag::domain::chunking::compute_mrr_and_hit_at_k;
use rag::domain::models::{IngestDocument, QueryOptions};
use rag::infrastructure::embeddings::LocalHashEmbedder;
use rag::infrastructure::postgres::PostgresRagRepository;

#[tokio::main]
async fn main() {
    let dsn = std::env::var("RAG_DATABASE_URL")
        .unwrap_or_else(|_| "postgresql://rag_admin:rag_dev_password@localhost:5433/rag_db".to_string());
    let repo = PostgresRagRepository::connect(&dsn).await.expect("Postgres required for rag-eval");
    repo.run_migrations().await.expect("Migrations failed");

    let embedder: Arc<dyn EmbeddingProvider> = Arc::new(LocalHashEmbedder::new(768));
    let service = Arc::new(RagService::new(Arc::new(repo.clone()), embedder));

    let col = format!("eval_bench_{}", std::process::id());
    let corpus = [
        ("d1", "PostgreSQL 17 pgvector HNSW cosine similarity index for dense semantic retrieval"),
        ("d2", "Reciprocal Rank Fusion merges dense vector ranks with tsvector full-text BM25 ranks"),
        ("d3", "Axum async web framework in Rust running on Tokio multi-threaded runtime"),
        ("d4", "Model Context Protocol MCP JSON-RPC 2.0 stdio transport for IDE agents"),
    ];

    for (id, text) in corpus {
        service
            .ingest_document(IngestDocument {
                collection: col.clone(),
                document_id: id.into(),
                source_path: format!("{id}.md"),
                content: text.into(),
                metadata: json!({}),
            })
            .await
            .unwrap();
    }

    let queries = [
        ("pgvector HNSW cosine", format!("{col}::d1")),
        ("Reciprocal Rank Fusion tsvector", format!("{col}::d2")),
        ("Axum Tokio Rust", format!("{col}::d3")),
        ("MCP JSON-RPC stdio", format!("{col}::d4")),
    ];

    let mut eval_pairs = Vec::new();
    let mut cold_ms = Vec::new();
    let mut warm_ms = Vec::new();

    for (q, expected) in &queries {
        let t0 = Instant::now();
        let r1 = service
            .query(QueryOptions {
                query: (*q).to_string(),
                collection: col.clone(),
                limit: 3,
                include_graph: false,
            })
            .await
            .unwrap();
        cold_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        let ranked: Vec<String> = r1.results.into_iter().map(|h| h.id).collect();
        eval_pairs.push((ranked, expected.clone()));

        let t1 = Instant::now();
        let _r2 = service
            .query(QueryOptions {
                query: (*q).to_string(),
                collection: col.clone(),
                limit: 3,
                include_graph: false,
            })
            .await
            .unwrap();
        warm_ms.push(t1.elapsed().as_secs_f64() * 1000.0);
    }

    let (hit_at_3, mrr) = compute_mrr_and_hit_at_k(&eval_pairs, 3);
    cold_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    warm_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "corpus_size": corpus.len(),
            "hit_at_3": hit_at_3,
            "mrr": mrr,
            "cold_p50_ms": cold_ms[cold_ms.len() / 2],
            "warm_l1_p50_ms": warm_ms[warm_ms.len() / 2],
        }))
        .unwrap()
    );

    let _ = repo.delete_test_data(&col, &[]).await;
}
