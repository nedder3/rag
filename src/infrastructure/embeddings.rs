use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::application::ports::EmbeddingProvider;

/// Deterministic local unit-norm feature-hashed embedder (zero external runtime required).
pub struct LocalHashEmbedder {
    dim: usize,
}

impl LocalHashEmbedder {
    pub fn new(dim: usize) -> Self {
        Self { dim: dim.max(8) }
    }
}

#[async_trait]
impl EmbeddingProvider for LocalHashEmbedder {
    fn model_name(&self) -> &str {
        "local-sha256-hash"
    }

    fn dimensions(&self) -> usize {
        self.dim
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        let mut vec = vec![0.0f32; self.dim];
        for tok in text
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .map(|s| s.to_lowercase())
            .filter(|s| !s.is_empty())
        {
            let digest = Sha256::digest(tok.as_bytes());
            let idx = u32::from_le_bytes([digest[0], digest[1], digest[2], digest[3]]) as usize % self.dim;
            vec[idx] += 1.0;
        }
        let norm: f32 = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut vec {
                *v /= norm;
            }
        } else {
            vec[0] = 1.0;
        }
        Ok(vec)
    }
}

#[derive(Serialize)]
struct OllamaEmbedReq<'a> {
    model: &'a str,
    input: &'a str,
}

#[derive(Deserialize)]
struct OllamaEmbedResp {
    embeddings: Option<Vec<Vec<f32>>>,
    embedding: Option<Vec<f32>>,
}

/// Robust embedding provider: calls Ollama `/api/embed` when reachable and falls back cleanly to `LocalHashEmbedder`.
pub struct FallbackEmbedder {
    client: reqwest::Client,
    ollama_url: String,
    model: String,
    dim: usize,
    fallback: LocalHashEmbedder,
}

impl FallbackEmbedder {
    pub fn new(ollama_url: String, model: String, dim: usize) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            client,
            ollama_url: ollama_url.trim_end_matches('/').to_string(),
            model,
            dim,
            fallback: LocalHashEmbedder::new(dim),
        }
    }
}

#[async_trait]
impl EmbeddingProvider for FallbackEmbedder {
    fn model_name(&self) -> &str {
        &self.model
    }

    fn dimensions(&self) -> usize {
        self.dim
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        let url = format!("{}/api/embed", self.ollama_url);
        if let Ok(resp) = self
            .client
            .post(&url)
            .json(&OllamaEmbedReq {
                model: &self.model,
                input: text,
            })
            .send()
            .await
        {
            if resp.status().is_success() {
                if let Ok(parsed) = resp.json::<OllamaEmbedResp>().await {
                    let maybe_vec = parsed
                        .embeddings
                        .and_then(|mut v| if v.is_empty() { None } else { Some(v.remove(0)) })
                        .or(parsed.embedding);
                    if let Some(v) = maybe_vec {
                        if v.len() == self.dim {
                            return Ok(v);
                        }
                    }
                }
            }
        }
        self.fallback.embed(text).await
    }
}
