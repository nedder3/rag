use std::sync::Arc;
use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;

use rag::application::ports::EmbeddingProvider;
use rag::application::service::RagService;
use rag::infrastructure::embeddings::FallbackEmbedder;
use rag::infrastructure::http::build_router;
use rag::infrastructure::postgres::PostgresRagRepository;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .init();

    let dsn = std::env::var("RAG_DATABASE_URL")
        .unwrap_or_else(|_| "postgresql://rag_admin:rag_dev_password@localhost:5433/rag_db".to_string());
    let ollama_url = std::env::var("RAG_OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());
    let embed_model = std::env::var("RAG_EMBED_MODEL").unwrap_or_else(|_| "nomic-embed-text".to_string());
    let addr = std::env::var("RAG_BIND_ADDR").unwrap_or_else(|_| "0.0.0.0:8010".to_string());

    let repo = PostgresRagRepository::connect(&dsn).await.expect("Failed to connect to Postgres");
    repo.run_migrations().await.expect("Failed to run SQL migrations");

    let embedder: Arc<dyn EmbeddingProvider> = Arc::new(FallbackEmbedder::new(ollama_url, embed_model, 768));
    let service = Arc::new(RagService::new(Arc::new(repo), embedder));
    let app = build_router(service);

    tracing::info!("rag-server listening on {}", addr);
    let listener = TcpListener::bind(&addr).await.expect("Failed to bind TCP listener");
    axum::serve(listener, app).await.expect("Axum server error");
}
