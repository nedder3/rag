use async_trait::async_trait;
use pgvector::Vector;
use serde_json::{json, Value};
use sqlx::postgres::{PgListener, PgPoolOptions};
use sqlx::{PgPool, Row};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::sync::Arc;

use crate::application::ports::RagRepository;
use crate::application::service::RagService;
use crate::domain::models::{
    DpoExportRecord, GraphEdge, GraphNeighbor, GraphNode, IngestDocument, SearchHit, TraceFeedbackRequest,
};

#[derive(Clone)]
pub struct PostgresRagRepository {
    pool: PgPool,
    fts_language: String,
}

#[derive(Clone)]
struct AStarState {
    cost: f64,
    depth: usize,
    node_id: String,
}

impl PartialEq for AStarState {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost && self.depth == other.depth && self.node_id == other.node_id
    }
}
impl Eq for AStarState {}
impl Ord for AStarState {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for AStarState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PostgresRagRepository {
    pub async fn connect(dsn: &str) -> Result<Self, String> {
        let pool = PgPoolOptions::new()
            .max_connections(16)
            .connect(dsn)
            .await
            .map_err(|e| format!("Postgres connection failed: {e}"))?;
        Ok(Self {
            pool,
            fts_language: "spanish".to_string(),
        })
    }

    pub async fn run_migrations(&self) -> Result<(), String> {
        sqlx::migrate!("./migrations")
            .run(&self.pool)
            .await
            .map_err(|e| format!("Migration failed: {e}"))
    }

    /// Spawns a background Tokio task listening to `rag_cache_invalidate` NOTIFY events
    /// so L1 caches stay consistent across `rag-server`, `rag-mcp`, and replicas.
    pub fn spawn_cache_invalidation_listener(&self, service: Arc<RagService>) -> tokio::task::JoinHandle<()> {
        let pool = self.pool.clone();
        tokio::spawn(async move {
            loop {
                match PgListener::connect_with(&pool).await {
                    Ok(mut listener) => {
                        if listener.listen("rag_cache_invalidate").await.is_err() {
                            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                            continue;
                        }
                        while let Ok(_notif) = listener.recv().await {
                            service.clear_cache();
                        }
                    }
                    Err(_) => {
                        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    }
                }
            }
        })
    }

    pub async fn delete_test_data(&self, collection: &str, node_ids: &[String]) -> Result<(), String> {
        sqlx::query("DELETE FROM rag_store.documents WHERE collection = $1")
            .bind(collection)
            .execute(&self.pool)
            .await
            .map_err(|e| e.to_string())?;
        sqlx::query("DELETE FROM rag_store.collections WHERE name = $1")
            .bind(collection)
            .execute(&self.pool)
            .await
            .map_err(|e| e.to_string())?;
        sqlx::query("DELETE FROM rag_store.query_traces WHERE collection = $1")
            .bind(collection)
            .execute(&self.pool)
            .await
            .map_err(|e| e.to_string())?;
        if !node_ids.is_empty() {
            sqlx::query("DELETE FROM rag_store.graph_nodes WHERE id = ANY($1)")
                .bind(node_ids)
                .execute(&self.pool)
                .await
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

#[async_trait]
impl RagRepository for PostgresRagRepository {
    async fn health_count(&self) -> Result<i64, String> {
        let row = sqlx::query("SELECT COUNT(*) AS cnt FROM rag_store.documents")
            .fetch_one(&self.pool)
            .await
            .map_err(|e| e.to_string())?;
        Ok(row.get::<i64, _>("cnt"))
    }

    async fn ensure_collection_model(&self, collection: &str, model: &str, dimensions: usize) -> Result<(), String> {
        if matches!(collection, "*" | "all" | "auto" | "") {
            return Ok(());
        }
        let existing = sqlx::query(
            "SELECT embedding_model, dimensions FROM rag_store.collections WHERE name = $1",
        )
        .bind(collection)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| e.to_string())?;

        if let Some(row) = existing {
            let reg_model: String = row.get("embedding_model");
            let reg_dim: i32 = row.get("dimensions");
            if reg_model != model || reg_dim != dimensions as i32 {
                return Err(format!(
                    "Embedding model mismatch for collection '{collection}': registered '{reg_model}' ({reg_dim}d), active '{model}' ({dimensions}d)"
                ));
            }
        } else {
            sqlx::query(
                "INSERT INTO rag_store.collections (name, embedding_model, dimensions) VALUES ($1, $2, $3) ON CONFLICT (name) DO NOTHING",
            )
            .bind(collection)
            .bind(model)
            .bind(dimensions as i32)
            .execute(&self.pool)
            .await
            .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    async fn upsert_document(&self, doc: &IngestDocument, scoped_id: &str, embedding: &[f32]) -> Result<(), String> {
        let vec = Vector::from(embedding.to_vec());
        sqlx::query(
            r#"
            INSERT INTO rag_store.documents (id, collection, source_path, content, metadata, embedding)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (id) DO UPDATE SET
                collection = EXCLUDED.collection,
                source_path = EXCLUDED.source_path,
                content = EXCLUDED.content,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding,
                updated_at = now()
            "#,
        )
        .bind(scoped_id)
        .bind(&doc.collection)
        .bind(&doc.source_path)
        .bind(&doc.content)
        .bind(&doc.metadata)
        .bind(vec)
        .execute(&self.pool)
        .await
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    async fn upsert_graph(&self, nodes: &[GraphNode], edges: &[GraphEdge]) -> Result<(usize, usize), String> {
        let mut n_cnt = 0usize;
        let mut e_cnt = 0usize;
        for n in nodes {
            sqlx::query(
                r#"
                INSERT INTO rag_store.graph_nodes (id, label, node_type, source_repo, metadata)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (id) DO UPDATE SET
                    label = EXCLUDED.label,
                    node_type = EXCLUDED.node_type,
                    updated_at = now()
                "#,
            )
            .bind(&n.id)
            .bind(&n.label)
            .bind(&n.node_type)
            .bind(&n.source_repo)
            .bind(&n.metadata)
            .execute(&self.pool)
            .await
            .map_err(|e| e.to_string())?;
            n_cnt += 1;
        }
        for e in edges {
            for ep in [&e.source_id, &e.target_id] {
                sqlx::query("INSERT INTO rag_store.graph_nodes (id, label) VALUES ($1, $1) ON CONFLICT DO NOTHING")
                    .bind(ep)
                    .execute(&self.pool)
                    .await
                    .map_err(|err| err.to_string())?;
            }
            sqlx::query(
                r#"
                INSERT INTO rag_store.graph_edges (source_id, target_id, relation, weight, metadata)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (source_id, target_id, relation) DO NOTHING
                "#,
            )
            .bind(&e.source_id)
            .bind(&e.target_id)
            .bind(&e.relation)
            .bind(e.weight)
            .bind(&e.metadata)
            .execute(&self.pool)
            .await
            .map_err(|err| err.to_string())?;
            e_cnt += 1;
        }
        Ok((n_cnt, e_cnt))
    }

    async fn hybrid_search(
        &self,
        collection: &str,
        query_text: &str,
        query_vec: &[f32],
        limit: usize,
    ) -> Result<Vec<SearchHit>, String> {
        let vec = Vector::from(query_vec.to_vec());
        let filter_all = matches!(collection, "*" | "all" | "auto" | "");
        let pool_k = (limit * 3).max(12) as i64;
        let lim = limit as i64;

        let rows = sqlx::query(
            r#"
            WITH vec_hits AS (
                SELECT id, collection, source_path, content, metadata,
                       (1.0 - (embedding <=> $1))::double precision AS vec_sim,
                       ROW_NUMBER() OVER (ORDER BY embedding <=> $1 ASC) AS vec_rank
                FROM rag_store.documents
                WHERE ($2 OR collection = $3)
                ORDER BY embedding <=> $1 ASC
                LIMIT $4
            ),
            fts_hits AS (
                SELECT id,
                       ts_rank_cd(tsv, plainto_tsquery($5::regconfig, $6))::double precision AS fts_score,
                       ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tsv, plainto_tsquery($5::regconfig, $6)) DESC) AS fts_rank
                FROM rag_store.documents
                WHERE ($2 OR collection = $3)
                  AND tsv @@ plainto_tsquery($5::regconfig, $6)
                ORDER BY fts_score DESC
                LIMIT $4
            )
            SELECT v.id, v.collection, v.source_path, v.content, v.metadata,
                   v.vec_sim,
                   COALESCE(f.fts_score, 0.0) AS fts_score,
                   ((1.0 / (60.0 + v.vec_rank)) + COALESCE(1.0 / (60.0 + f.fts_rank), 0.0))::double precision AS rrf_score
            FROM vec_hits v
            LEFT JOIN fts_hits f ON v.id = f.id
            ORDER BY rrf_score DESC, v.vec_sim DESC
            LIMIT $7
            "#,
        )
        .bind(vec)
        .bind(filter_all)
        .bind(collection)
        .bind(pool_k)
        .bind(&self.fts_language)
        .bind(query_text)
        .bind(lim)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| format!("Hybrid search query failed: {e}"))?;

        Ok(rows
            .into_iter()
            .map(|r| SearchHit {
                id: r.get("id"),
                collection: r.get("collection"),
                source_path: r.get("source_path"),
                content: r.get("content"),
                metadata: r.get::<Value, _>("metadata"),
                vec_sim: r.get("vec_sim"),
                fts_score: r.get("fts_score"),
                rrf_score: r.get("rrf_score"),
            })
            .collect())
    }

    async fn query_graph_astar(
        &self,
        tokens: &[String],
        max_hops: usize,
        max_neighbors: usize,
    ) -> Result<Vec<GraphNeighbor>, String> {
        let patterns: Vec<String> = tokens.iter().map(|t| format!("%{t}%")).collect();
        let seeds = sqlx::query(
            "SELECT id FROM rag_store.graph_nodes WHERE lower(label) LIKE ANY($1) OR lower(id) LIKE ANY($1) LIMIT 5",
        )
        .bind(&patterns)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| e.to_string())?;

        if seeds.is_empty() {
            return Ok(vec![]);
        }

        let mut visited: HashSet<String> = HashSet::new();
        let mut pq = BinaryHeap::new();
        for s in seeds {
            let sid: String = s.get("id");
            visited.insert(sid.clone());
            pq.push(AStarState {
                cost: 0.0,
                depth: 0,
                node_id: sid,
            });
        }

        let mut neighbors = Vec::new();
        while let Some(cur) = pq.pop() {
            if neighbors.len() >= max_neighbors {
                break;
            }
            if cur.depth >= max_hops {
                continue;
            }
            let edges = sqlx::query(
                r#"
                SELECT e.target_id, n.label, n.node_type, e.relation, e.weight
                FROM rag_store.graph_edges e
                JOIN rag_store.graph_nodes n ON n.id = e.target_id
                WHERE e.source_id = $1
                LIMIT 12
                "#,
            )
            .bind(&cur.node_id)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| e.to_string())?;

            for edge in edges {
                let tgt_id: String = edge.get("target_id");
                if !visited.insert(tgt_id.clone()) {
                    continue;
                }
                let weight: f64 = edge.get("weight");
                let step_cost = cur.cost + (1.0 / weight.max(0.1));
                neighbors.push(GraphNeighbor {
                    id: tgt_id.clone(),
                    label: edge.get("label"),
                    node_type: edge.get("node_type"),
                    relation: edge.get("relation"),
                    from_node: cur.node_id.clone(),
                    hops: cur.depth + 1,
                    f_score: step_cost,
                });
                pq.push(AStarState {
                    cost: step_cost,
                    depth: cur.depth + 1,
                    node_id: tgt_id,
                });
                if neighbors.len() >= max_neighbors {
                    break;
                }
            }
        }
        Ok(neighbors)
    }

    async fn record_trace(
        &self,
        collection: &str,
        query_text: &str,
        hits: &[SearchHit],
        latency_ms: f64,
        cache_hit: bool,
    ) -> Result<i64, String> {
        let retrieved_ids: Vec<String> = hits.iter().map(|h| h.id.clone()).collect();
        let contexts = Value::Array(
            hits.iter()
                .map(|h| {
                    json!({
                        "id": h.id,
                        "source_path": h.source_path,
                        "content": h.content,
                        "rrf_score": h.rrf_score,
                    })
                })
                .collect(),
        );
        let row = sqlx::query(
            "INSERT INTO rag_store.query_traces (collection, query_text, retrieved_ids, retrieved_contexts, latency_ms, cache_hit) VALUES ($1, $2, $3, $4, $5, $6) RETURNING id",
        )
        .bind(collection)
        .bind(query_text)
        .bind(&retrieved_ids)
        .bind(&contexts)
        .bind(latency_ms)
        .bind(cache_hit)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| e.to_string())?;
        Ok(row.get("id"))
    }

    async fn record_feedback(&self, fb: &TraceFeedbackRequest) -> Result<i64, String> {
        let row = sqlx::query(
            "INSERT INTO rag_store.trace_feedback (trace_id, role, chosen, rejected, reward) VALUES ($1, $2, $3, $4, $5) RETURNING id",
        )
        .bind(fb.trace_id)
        .bind(&fb.role)
        .bind(&fb.chosen)
        .bind(&fb.rejected)
        .bind(fb.reward)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| e.to_string())?;
        Ok(row.get("id"))
    }

    async fn export_dpo(&self, limit: usize) -> Result<Vec<DpoExportRecord>, String> {
        let rows = sqlx::query(
            r#"
            SELECT t.id AS trace_id, t.collection, f.role, t.query_text AS prompt,
                   t.retrieved_contexts, f.chosen, f.rejected, f.reward
            FROM rag_store.trace_feedback f
            JOIN rag_store.query_traces t ON t.id = f.trace_id
            ORDER BY f.id DESC
            LIMIT $1
            "#,
        )
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| e.to_string())?;

        Ok(rows
            .into_iter()
            .map(|r| DpoExportRecord {
                trace_id: r.get("trace_id"),
                collection: r.get("collection"),
                role: r.get("role"),
                prompt: r.get("prompt"),
                retrieved_contexts: r.get::<Value, _>("retrieved_contexts"),
                chosen: r.get("chosen"),
                rejected: r.get("rejected"),
                reward: r.get("reward"),
            })
            .collect())
    }
}
