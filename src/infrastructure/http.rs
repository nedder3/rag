use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;

use crate::application::service::RagService;
use crate::domain::models::{GraphEdge, GraphNode, IngestDocument, QueryOptions};

#[derive(Deserialize)]
pub struct GraphIngestBody {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

pub fn build_router(service: Arc<RagService>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/ingest", post(ingest))
        .route("/v1/query", post(query))
        .route("/v1/graph/ingest", post(ingest_graph))
        .route("/v1/metrics", get(metrics))
        .route("/v1/cache/clear", post(clear_cache))
        .with_state(service)
}

async fn health(State(svc): State<Arc<RagService>>) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    match svc.health().await {
        Ok(docs) => Ok(Json(json!({"status": "ok", "documents": docs}))),
        Err(e) => Err((StatusCode::SERVICE_UNAVAILABLE, Json(json!({"status": "error", "detail": e})))),
    }
}

async fn ingest(
    State(svc): State<Arc<RagService>>,
    Json(doc): Json<IngestDocument>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    match svc.ingest_document(doc).await {
        Ok(id) => Ok(Json(json!({"status": "ok", "id": id}))),
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e})))),
    }
}

async fn query(
    State(svc): State<Arc<RagService>>,
    Json(opts): Json<QueryOptions>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    match svc.query(opts).await {
        Ok(res) => Ok(Json(serde_json::to_value(res).unwrap())),
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e})))),
    }
}

async fn ingest_graph(
    State(svc): State<Arc<RagService>>,
    Json(body): Json<GraphIngestBody>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    match svc.ingest_graph(body.nodes, body.edges).await {
        Ok((nodes, edges)) => Ok(Json(json!({"nodes": nodes, "edges": edges}))),
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e})))),
    }
}

async fn metrics(State(svc): State<Arc<RagService>>) -> Json<Value> {
    Json(serde_json::to_value(svc.telemetry()).unwrap())
}

async fn clear_cache(State(svc): State<Arc<RagService>>) -> Json<Value> {
    svc.clear_cache();
    Json(json!({"status": "ok"}))
}
