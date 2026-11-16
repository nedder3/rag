use std::sync::Arc;
use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;

use rag::application::service::RagService;
use rag::infrastructure::embeddings::build_embedder_from_env;
use rag::infrastructure::http::build_router;
use rag::infrastructure::postgres::PostgresRagRepository;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .init();

    let dsn = std::env::var("RAG_DATABASE_URL")
        .unwrap_or_else(|_| "postgresql://rag_admin:rag_dev_password@localhost:5433/rag_db".to_string());
    let addr = std::env::var("RAG_BIND_ADDR").unwrap_or_else(|_| "0.0.0.0:8010".to_string());

    let repo = PostgresRagRepository::connect(&dsn).await.expect("Failed to connect to Postgres");
    repo.run_migrations().await.expect("Failed to run SQL migrations");

    let embedder = build_embedder_from_env();
    let service = Arc::new(RagService::new(Arc::new(repo.clone()), embedder));
    let _listener_handle = repo.spawn_cache_invalidation_listener(service.clone());
    let app = build_router(service);

    tracing::info!("rag-server listening on {}", addr);
    let listener = TcpListener::bind(&addr).await.expect("Failed to bind TCP listener");
    axum::serve(listener, app).await.expect("Axum server error");
}
