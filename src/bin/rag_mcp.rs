use std::sync::Arc;
use tokio::io::{self, AsyncBufReadExt, AsyncWriteExt, BufReader};

use rag::application::ports::EmbeddingProvider;
use rag::application::service::RagService;
use rag::infrastructure::embeddings::FallbackEmbedder;
use rag::infrastructure::mcp::handle_jsonrpc;
use rag::infrastructure::postgres::PostgresRagRepository;

#[tokio::main]
async fn main() {
    let dsn = std::env::var("RAG_DATABASE_URL")
        .unwrap_or_else(|_| "postgresql://rag_admin:rag_dev_password@localhost:5433/rag_db".to_string());
    let ollama_url = std::env::var("RAG_OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());
    let embed_model = std::env::var("RAG_EMBED_MODEL").unwrap_or_else(|_| "nomic-embed-text".to_string());

    let repo = PostgresRagRepository::connect(&dsn).await.expect("Failed to connect to Postgres");
    repo.run_migrations().await.expect("Failed to run migrations");

    let embedder: Arc<dyn EmbeddingProvider> = Arc::new(FallbackEmbedder::new(ollama_url, embed_model, 768));
    let service = Arc::new(RagService::new(Arc::new(repo), embedder));

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
