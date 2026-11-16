use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub id: String,
    pub source_path: String,
    pub content: String,
    pub chunk_index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestDocument {
    pub collection: String,
    pub document_id: String,
    #[serde(default)]
    pub source_path: String,
    pub content: String,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub label: String,
    #[serde(default = "default_node_type")]
    pub node_type: String,
    #[serde(default = "default_repo")]
    pub source_repo: String,
    #[serde(default)]
    pub metadata: Value,
}

fn default_node_type() -> String {
    "concept".to_string()
}

fn default_repo() -> String {
    "default".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub source_id: String,
    pub target_id: String,
    #[serde(default = "default_relation")]
    pub relation: String,
    #[serde(default = "default_weight")]
    pub weight: f64,
    #[serde(default)]
    pub metadata: Value,
}

fn default_relation() -> String {
    "related".to_string()
}

fn default_weight() -> f64 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryOptions {
    pub query: String,
    #[serde(default = "default_collection")]
    pub collection: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default = "default_true")]
    pub include_graph: bool,
}

fn default_collection() -> String {
    "auto".to_string()
}

fn default_limit() -> usize {
    4
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub id: String,
    pub collection: String,
    pub source_path: String,
    pub content: String,
    pub metadata: Value,
    pub vec_sim: f64,
    pub fts_score: f64,
    pub rrf_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNeighbor {
    pub id: String,
    pub label: String,
    pub node_type: String,
    pub relation: String,
    pub from_node: String,
    pub hops: usize,
    pub f_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResponse {
    pub collection: String,
    pub cache_hit: bool,
    pub latency_ms: f64,
    pub results: Vec<SearchHit>,
    pub graph_neighbors: Vec<GraphNeighbor>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetrySnapshot {
    pub total_queries: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub embed_cache_hits: u64,
    pub avg_latency_ms: f64,
    pub last_latency_ms: f64,
}
