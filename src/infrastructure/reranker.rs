use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::application::ports::Reranker;
use crate::domain::models::SearchHit;

/// Fast in-process microsecond reranker: boosts exact phrase/identifier proximity on top of RRF.
pub struct LexicalBoostReranker;

impl LexicalBoostReranker {
    fn boost_score(query: &str, hit: &SearchHit) -> f64 {
        let q_lower = query.trim().to_lowercase();
        let c_lower = hit.content.to_lowercase();
        let s_lower = hit.source_path.to_lowercase();

        let mut bonus = 0.0f64;
        if !q_lower.is_empty() && c_lower.contains(&q_lower) {
            bonus += 0.02;
        }
        let tokens: Vec<&str> = q_lower
            .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
            .filter(|t| t.len() >= 3)
            .collect();
        if !tokens.is_empty() {
            let matched = tokens.iter().filter(|t| c_lower.contains(**t) || s_lower.contains(**t)).count();
            bonus += 0.01 * (matched as f64 / tokens.len() as f64);
        }
        hit.rrf_score + bonus
    }
}

#[async_trait]
impl Reranker for LexicalBoostReranker {
    async fn rerank(&self, query: &str, mut hits: Vec<SearchHit>, limit: usize) -> Vec<SearchHit> {
        hits.sort_by(|a, b| {
            Self::boost_score(query, b)
                .partial_cmp(&Self::boost_score(query, a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(limit);
        hits
    }
}

#[derive(Serialize)]
struct RerankReq<'a> {
    query: &'a str,
    documents: Vec<&'a str>,
    top_n: usize,
}

#[derive(Deserialize)]
struct RerankItem {
    index: usize,
    relevance_score: f64,
}

#[derive(Deserialize)]
struct RerankResp {
    results: Vec<RerankItem>,
}

/// Optional HTTP Cross-Encoder reranker (e.g., TEI / vLLM rerank endpoint), falling back to `LexicalBoostReranker` on error.
pub struct HttpCrossEncoderReranker {
    client: reqwest::Client,
    url: String,
    fallback: LexicalBoostReranker,
}

impl HttpCrossEncoderReranker {
    pub fn new(url: String) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(3))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            client,
            url,
            fallback: LexicalBoostReranker,
        }
    }
}

#[async_trait]
impl Reranker for HttpCrossEncoderReranker {
    async fn rerank(&self, query: &str, hits: Vec<SearchHit>, limit: usize) -> Vec<SearchHit> {
        if hits.is_empty() {
            return hits;
        }
        let docs: Vec<&str> = hits.iter().map(|h| h.content.as_str()).collect();
        if let Ok(resp) = self
            .client
            .post(&self.url)
            .json(&RerankReq {
                query,
                documents: docs,
                top_n: limit,
            })
            .send()
            .await
        {
            if resp.status().is_success() {
                if let Ok(parsed) = resp.json::<RerankResp>().await {
                    let mut out = Vec::with_capacity(limit.min(hits.len()));
                    for item in parsed.results {
                        if let Some(h) = hits.get(item.index) {
                            let mut cloned = h.clone();
                            cloned.rrf_score = item.relevance_score;
                            out.push(cloned);
                        }
                    }
                    if !out.is_empty() {
                        out.truncate(limit);
                        return out;
                    }
                }
            }
        }
        self.fallback.rerank(query, hits, limit).await
    }
}

pub fn build_reranker_from_env() -> std::sync::Arc<dyn Reranker> {
    if let Ok(url) = std::env::var("RAG_RERANK_URL") {
        let trimmed = url.trim();
        if !trimmed.is_empty() {
            return std::sync::Arc::new(HttpCrossEncoderReranker::new(trimmed.to_string()));
        }
    }
    std::sync::Arc::new(LexicalBoostReranker)
}
