use lru::LruCache;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::application::ports::{EmbeddingProvider, RagRepository, Reranker};
use crate::domain::models::{
    DpoExportRecord, GraphEdge, GraphNeighbor, GraphNode, IngestDocument, QueryOptions, QueryResponse,
    SearchHit, TelemetrySnapshot, TraceFeedbackRequest,
};
use crate::infrastructure::reranker::LexicalBoostReranker;

struct CacheState {
    embed: LruCache<String, Vec<f32>>,
    query: LruCache<(String, String, usize), Vec<SearchHit>>,
    graph: LruCache<(String, usize, usize), Vec<GraphNeighbor>>,
    total_queries: u64,
    cache_hits: u64,
    cache_misses: u64,
    embed_cache_hits: u64,
    total_latency_ms: f64,
    last_latency_ms: f64,
}

pub struct RagService {
    repo: Arc<dyn RagRepository>,
    embedder: Arc<dyn EmbeddingProvider>,
    reranker: Arc<dyn Reranker>,
    state: Mutex<CacheState>,
}

impl RagService {
    pub fn new(repo: Arc<dyn RagRepository>, embedder: Arc<dyn EmbeddingProvider>) -> Self {
        Self::with_reranker(repo, embedder, Arc::new(LexicalBoostReranker))
    }

    pub fn with_reranker(
        repo: Arc<dyn RagRepository>,
        embedder: Arc<dyn EmbeddingProvider>,
        reranker: Arc<dyn Reranker>,
    ) -> Self {
        let cap = NonZeroUsize::new(512).unwrap();
        Self {
            repo,
            embedder,
            reranker,
            state: Mutex::new(CacheState {
                embed: LruCache::new(cap),
                query: LruCache::new(cap),
                graph: LruCache::new(cap),
                total_queries: 0,
                cache_hits: 0,
                cache_misses: 0,
                embed_cache_hits: 0,
                total_latency_ms: 0.0,
                last_latency_ms: 0.0,
            }),
        }
    }

    pub async fn health(&self) -> Result<i64, String> {
        self.repo.health_count().await
    }

    pub fn clear_cache(&self) {
        if let Ok(mut st) = self.state.lock() {
            st.query.clear();
            st.graph.clear();
        }
    }

    async fn embed_cached(&self, text: &str) -> Result<Vec<f32>, String> {
        let key = format!("{}:{}", self.embedder.model_name(), text.trim());
        if let Ok(mut st) = self.state.lock() {
            if let Some(vec) = st.embed.get(&key).cloned() {
                st.embed_cache_hits += 1;
                return Ok(vec);
            }
        }
        let vec = self.embedder.embed(text.trim()).await?;
        if let Ok(mut st) = self.state.lock() {
            st.embed.put(key, vec.clone());
        }
        Ok(vec)
    }

    pub async fn ingest_document(&self, doc: IngestDocument) -> Result<String, String> {
        self.repo
            .ensure_collection_model(&doc.collection, self.embedder.model_name(), self.embedder.dimensions())
            .await?;
        let scoped_id = if doc.document_id.contains("::") {
            doc.document_id.clone()
        } else {
            format!("{}::{}", doc.collection, doc.document_id)
        };
        let emb = self.embed_cached(&doc.content).await?;
        self.repo.upsert_document(&doc, &scoped_id, &emb).await?;
        self.clear_cache();
        Ok(scoped_id)
    }

    pub async fn ingest_graph(&self, nodes: Vec<GraphNode>, edges: Vec<GraphEdge>) -> Result<(usize, usize), String> {
        let res = self.repo.upsert_graph(&nodes, &edges).await?;
        self.clear_cache();
        Ok(res)
    }

    fn tokenize(text: &str) -> Vec<String> {
        text.split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
            .map(|s| s.to_lowercase())
            .filter(|s| s.len() >= 3)
            .take(5)
            .collect()
    }

    pub async fn query(&self, opts: QueryOptions) -> Result<QueryResponse, String> {
        let t0 = Instant::now();
        let q_trim = opts.query.trim().to_string();
        if q_trim.is_empty() {
            return Ok(QueryResponse {
                trace_id: None,
                collection: opts.collection,
                cache_hit: false,
                latency_ms: 0.0,
                results: vec![],
                graph_neighbors: vec![],
            });
        }

        self.repo
            .ensure_collection_model(&opts.collection, self.embedder.model_name(), self.embedder.dimensions())
            .await?;

        let limit = opts.limit.clamp(1, 50);
        let qkey = (opts.collection.clone(), q_trim.clone(), limit);
        let mut cache_hit = false;

        let results = if let Some(cached) = self.state.lock().ok().and_then(|mut s| s.query.get(&qkey).cloned()) {
            cache_hit = true;
            if let Ok(mut st) = self.state.lock() {
                st.cache_hits += 1;
            }
            cached
        } else {
            if let Ok(mut st) = self.state.lock() {
                st.cache_misses += 1;
            }
            let qvec = self.embed_cached(&q_trim).await?;
            let raw_hits = self.repo.hybrid_search(&opts.collection, &q_trim, &qvec, (limit * 3).max(12)).await?;
            let hits = self.reranker.rerank(&q_trim, raw_hits, limit).await;
            if let Ok(mut st) = self.state.lock() {
                st.query.put(qkey, hits.clone());
            }
            hits
        };

        let graph_neighbors = if opts.include_graph {
            let gkey = (q_trim.clone(), 2usize, 6usize);
            if let Some(cached_g) = self.state.lock().ok().and_then(|mut s| s.graph.get(&gkey).cloned()) {
                cached_g
            } else {
                let tokens = Self::tokenize(&q_trim);
                let neighbors = if tokens.is_empty() {
                    vec![]
                } else {
                    self.repo.query_graph_astar(&tokens, 2, 6).await?
                };
                if let Ok(mut st) = self.state.lock() {
                    st.graph.put(gkey, neighbors.clone());
                }
                neighbors
            }
        } else {
            vec![]
        };

        let latency_ms = (t0.elapsed().as_secs_f64() * 1000.0 * 1000.0).round() / 1000.0;
        if let Ok(mut st) = self.state.lock() {
            st.total_queries += 1;
            st.total_latency_ms += latency_ms;
            st.last_latency_ms = latency_ms;
        }

        let trace_id = self
            .repo
            .record_trace(&opts.collection, &q_trim, &results, latency_ms, cache_hit)
            .await
            .ok();

        Ok(QueryResponse {
            trace_id,
            collection: opts.collection,
            cache_hit,
            latency_ms,
            results,
            graph_neighbors,
        })
    }

    pub async fn record_feedback(&self, fb: TraceFeedbackRequest) -> Result<i64, String> {
        self.repo.record_feedback(&fb).await
    }

    pub async fn export_dpo(&self, limit: usize) -> Result<Vec<DpoExportRecord>, String> {
        self.repo.export_dpo(limit).await
    }

    pub fn telemetry(&self) -> TelemetrySnapshot {
        let st = self.state.lock().unwrap();
        let avg = if st.total_queries > 0 {
            st.total_latency_ms / (st.total_queries as f64)
        } else {
            0.0
        };
        TelemetrySnapshot {
            total_queries: st.total_queries,
            cache_hits: st.cache_hits,
            cache_misses: st.cache_misses,
            embed_cache_hits: st.embed_cache_hits,
            avg_latency_ms: (avg * 1000.0).round() / 1000.0,
            last_latency_ms: st.last_latency_ms,
        }
    }
}
