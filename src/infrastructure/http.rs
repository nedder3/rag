use axum::extract::{Query, State};
use axum::http::{header, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;

use crate::application::service::RagService;
use crate::domain::models::{GraphEdge, GraphNode, IngestDocument, QueryOptions, TraceFeedbackRequest};

#[derive(Deserialize)]
pub struct GraphIngestBody {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

#[derive(Deserialize)]
pub struct ExportQuery {
    pub limit: Option<usize>,
    pub format: Option<String>,
}

pub fn build_router(service: Arc<RagService>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/ingest", post(ingest))
        .route("/v1/query", post(query))
        .route("/v1/graph/ingest", post(ingest_graph))
        .route("/v1/feedback", post(feedback))
        .route("/v1/export/dpo", get(export_dpo))
        .route("/v1/metrics", get(metrics))
        .route("/v1/cache/clear", post(clear_cache))
        .with_state(service)
}

fn map_service_err(err: String) -> (StatusCode, Json<Value>) {
    let status = if err.contains("unavailable") {
        StatusCode::SERVICE_UNAVAILABLE
    } else if err.contains("mismatch") {
        StatusCode::CONFLICT
    } else {
        StatusCode::INTERNAL_SERVER_ERROR
    };
    (status, Json(json!({"error": err})))
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
        Err(e) => Err(map_service_err(e)),
    }
}

async fn query(
    State(svc): State<Arc<RagService>>,
    Json(opts): Json<QueryOptions>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    match svc.query(opts).await {
        Ok(res) => Ok(Json(serde_json::to_value(res).unwrap())),
        Err(e) => Err(map_service_err(e)),
    }
}

async fn ingest_graph(
    State(svc): State<Arc<RagService>>,
    Json(body): Json<GraphIngestBody>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    match svc.ingest_graph(body.nodes, body.edges).await {
        Ok((nodes, edges)) => Ok(Json(json!({"nodes": nodes, "edges": edges}))),
        Err(e) => Err(map_service_err(e)),
    }
}

async fn feedback(
    State(svc): State<Arc<RagService>>,
    Json(fb): Json<TraceFeedbackRequest>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    match svc.record_feedback(fb).await {
        Ok(id) => Ok(Json(json!({"status": "ok", "feedback_id": id}))),
        Err(e) => Err(map_service_err(e)),
    }
}

async fn export_dpo(
    State(svc): State<Arc<RagService>>,
    Query(q): Query<ExportQuery>,
) -> Result<impl IntoResponse, (StatusCode, Json<Value>)> {
    let records = svc.export_dpo(q.limit.unwrap_or(500)).await.map_err(map_service_err)?;
    let fmt = q.format.unwrap_or_else(|| "trl_jsonl".to_string());
    if fmt == "json" {
        return Ok((
            [(header::CONTENT_TYPE, "application/json")],
            serde_json::to_string(&json!({"count": records.len(), "records": records})).unwrap(),
        ));
    }
    let mut lines = Vec::with_capacity(records.len());
    for rec in records {
        let ctx_str = rec
            .retrieved_contexts
            .as_array()
            .map(|arr| {
                arr.iter()
                    .enumerate()
                    .filter_map(|(i, item)| {
                        item.get("content")
                            .and_then(|c| c.as_str())
                            .map(|c| format!("[{}] {}", i + 1, c))
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default();
        let formatted_prompt = if ctx_str.is_empty() {
            rec.prompt
        } else {
            format!("Context:\n{ctx_str}\n\nQuery:\n{}", rec.prompt)
        };
        let row = json!({
            "prompt": formatted_prompt,
            "chosen": rec.chosen,
            "rejected": rec.rejected.unwrap_or_default(),
        });
        lines.push(row.to_string());
    }
    let body = if lines.is_empty() {
        String::new()
    } else {
        format!("{}\n", lines.join("\n"))
    };
    Ok(([(header::CONTENT_TYPE, "application/x-ndjson")], body))
}

async fn metrics(State(svc): State<Arc<RagService>>) -> Json<Value> {
    Json(serde_json::to_value(svc.telemetry()).unwrap())
}

async fn clear_cache(State(svc): State<Arc<RagService>>) -> Json<Value> {
    svc.clear_cache();
    Json(json!({"status": "ok"}))
}
