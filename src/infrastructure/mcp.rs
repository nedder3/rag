use serde_json::{json, Value};
use std::sync::Arc;

use crate::application::service::RagService;
use crate::domain::models::{IngestDocument, QueryOptions};

fn is_read_only() -> bool {
    !matches!(
        std::env::var("RAG_MCP_READ_ONLY").as_deref(),
        Ok("false") | Ok("0")
    )
}

/// Handles a single JSON-RPC 2.0 MCP request line.
/// By default (`RAG_MCP_READ_ONLY=true`), `rag-mcp` exposes read-only `rag_query`;
/// any writes (when enabled) still propagate L1 invalidation via PostgreSQL `LISTEN/NOTIFY`.
pub async fn handle_jsonrpc(service: &Arc<RagService>, raw: &str) -> String {
    let req: Value = match serde_json::from_str(raw) {
        Ok(v) => v,
        Err(e) => {
            return json!({
                "jsonrpc": "2.0",
                "id": Value::Null,
                "error": {"code": -32700, "message": format!("Parse error: {e}")}
            })
            .to_string();
        }
    };

    let id = req.get("id").cloned().unwrap_or(Value::Null);
    let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");
    let read_only = is_read_only();

    match method {
        "initialize" => json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "rag-mcp-rust", "version": "0.1.0"},
                "capabilities": {"tools": {}}
            }
        })
        .to_string(),
        "tools/list" => {
            let mut tools = vec![json!({
                "name": "rag_query",
                "description": "Hybrid pgvector + tsvector RRF + Graph A* search",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "collection": {"type": "string", "default": "auto"},
                        "limit": {"type": "integer", "default": 4},
                        "include_graph": {"type": "boolean", "default": true}
                    },
                    "required": ["query"]
                }
            })];
            if !read_only {
                tools.push(json!({
                    "name": "rag_ingest",
                    "description": "Ingest a document into rag_store",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "collection": {"type": "string"},
                            "document_id": {"type": "string"},
                            "content": {"type": "string"}
                        },
                        "required": ["collection", "document_id", "content"]
                    }
                }));
            }
            json!({"jsonrpc": "2.0", "id": id, "result": {"tools": tools}}).to_string()
        }
        "tools/call" => {
            let params = req.get("params").and_then(|p| p.as_object());
            let name = params.and_then(|p| p.get("name")).and_then(|n| n.as_str()).unwrap_or("");
            let args = params.and_then(|p| p.get("arguments")).cloned().unwrap_or(json!({}));

            if name == "rag_query" {
                match serde_json::from_value::<QueryOptions>(args) {
                    Ok(opts) => match service.query(opts).await {
                        Ok(res) => json!({"jsonrpc": "2.0", "id": id, "result": res}).to_string(),
                        Err(e) => json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32000, "message": e}}).to_string(),
                    },
                    Err(e) => json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32602, "message": e.to_string()}}).to_string(),
                }
            } else if name == "rag_ingest" {
                if read_only {
                    return json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": {"code": -32601, "message": "rag-mcp is in read-only mode (set RAG_MCP_READ_ONLY=false or POST /v1/ingest)"}
                    })
                    .to_string();
                }
                match serde_json::from_value::<IngestDocument>(args) {
                    Ok(doc) => match service.ingest_document(doc).await {
                        Ok(doc_id) => json!({"jsonrpc": "2.0", "id": id, "result": {"status": "ok", "id": doc_id}}).to_string(),
                        Err(e) => json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32000, "message": e}}).to_string(),
                    },
                    Err(e) => json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32602, "message": e.to_string()}}).to_string(),
                }
            } else {
                json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32601, "message": "Unknown tool"}}).to_string()
            }
        }
        _ => json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": {"code": -32601, "message": "Method not found"}
        })
        .to_string(),
    }
}
