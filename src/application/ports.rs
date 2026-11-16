use async_trait::async_trait;

use crate::domain::models::{
    DpoExportRecord, GraphEdge, GraphNeighbor, GraphNode, IngestDocument, SearchHit, TraceFeedbackRequest,
};

#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    fn model_name(&self) -> &str;
    fn dimensions(&self) -> usize;
    async fn embed(&self, text: &str) -> Result<Vec<f32>, String>;
}

#[async_trait]
pub trait Reranker: Send + Sync {
    async fn rerank(&self, query: &str, hits: Vec<SearchHit>, limit: usize) -> Vec<SearchHit>;
}

#[async_trait]
pub trait RagRepository: Send + Sync {
    async fn health_count(&self) -> Result<i64, String>;
    async fn ensure_collection_model(&self, collection: &str, model: &str, dimensions: usize) -> Result<(), String>;
    async fn upsert_document(&self, doc: &IngestDocument, scoped_id: &str, embedding: &[f32]) -> Result<(), String>;
    async fn upsert_graph(&self, nodes: &[GraphNode], edges: &[GraphEdge]) -> Result<(usize, usize), String>;
    async fn hybrid_search(
        &self,
        collection: &str,
        query_text: &str,
        query_vec: &[f32],
        limit: usize,
    ) -> Result<Vec<SearchHit>, String>;
    async fn query_graph_astar(
        &self,
        tokens: &[String],
        max_hops: usize,
        max_neighbors: usize,
    ) -> Result<Vec<GraphNeighbor>, String>;
    async fn record_trace(
        &self,
        collection: &str,
        query_text: &str,
        hits: &[SearchHit],
        latency_ms: f64,
        cache_hit: bool,
    ) -> Result<i64, String>;
    async fn record_feedback(&self, fb: &TraceFeedbackRequest) -> Result<i64, String>;
    async fn export_dpo(&self, limit: usize) -> Result<Vec<DpoExportRecord>, String>;
}
