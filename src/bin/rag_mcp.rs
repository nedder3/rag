use std::sync::Arc;
use tokio::io::{self, AsyncBufReadExt, AsyncWriteExt, BufReader};

use rag::application::service::RagService;
use rag::infrastructure::embeddings::build_embedder_from_env;
use rag::infrastructure::mcp::handle_jsonrpc;
use rag::infrastructure::postgres::PostgresRagRepository;
use rag::infrastructure::reranker::build_reranker_from_env;

#[tokio::main]
async fn main() {
    let dsn = std::env::var("RAG_DATABASE_URL")
        .unwrap_or_else(|_| "postgresql://rag_admin:rag_dev_password@localhost:5433/rag_db".to_string());

    let repo = PostgresRagRepository::connect(&dsn).await.expect("Failed to connect to Postgres");
    repo.run_migrations().await.expect("Failed to run migrations");

    let embedder = build_embedder_from_env();
    let reranker = build_reranker_from_env();
    let service = Arc::new(RagService::with_reranker(Arc::new(repo.clone()), embedder, reranker));
    let _listener_handle = repo.spawn_cache_invalidation_listener(service.clone());

    let stdin = BufReader::new(io::stdin());
    let mut lines = stdin.lines();
    let mut stdout = io::stdout();

    while let Ok(Some(line)) = lines.next_line().await {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let resp = handle_jsonrpc(&service, trimmed).await;
        let _ = stdout.write_all(resp.as_bytes()).await;
        let _ = stdout.write_all(b"\n").await;
        let _ = stdout.flush().await;
    }
}
