use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::application::ports::EmbeddingProvider;

/// Deterministic local unit-norm feature-hashed embedder (for tests/CI or explicit `local` collections).
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

/// Strict Ollama `/api/embed` provider. Never falls back silently to an incompatible vector space.
pub struct OllamaEmbedder {
    client: reqwest::Client,
    ollama_url: String,
    model: String,
    dim: usize,
}

impl OllamaEmbedder {
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
        }
    }
}

#[async_trait]
impl EmbeddingProvider for OllamaEmbedder {
    fn model_name(&self) -> &str {
        &self.model
    }

    fn dimensions(&self) -> usize {
        self.dim
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        let url = format!("{}/api/embed", self.ollama_url);
        let resp = self
            .client
            .post(&url)
            .json(&OllamaEmbedReq {
                model: &self.model,
                input: text,
            })
            .send()
            .await
            .map_err(|e| format!("Embedding provider unavailable ({}): {e}", self.ollama_url))?;

        if !resp.status().is_success() {
            return Err(format!(
                "Embedding provider unavailable ({}): HTTP {}",
                self.ollama_url,
                resp.status()
            ));
        }

        let parsed = resp
            .json::<OllamaEmbedResp>()
            .await
            .map_err(|e| format!("Invalid embedding payload: {e}"))?;

        let vec = parsed
            .embeddings
            .and_then(|mut v| if v.is_empty() { None } else { Some(v.remove(0)) })
            .or(parsed.embedding)
            .ok_or_else(|| "Embedding response missing vector".to_string())?;

        if vec.len() != self.dim {
            return Err(format!(
                "Embedding dimension mismatch for '{}': expected {}, got {}",
                self.model,
                self.dim,
                vec.len()
            ));
        }
        Ok(vec)
    }
}

/// Builds the configured embedding provider (`ollama` by default, `local` for tests/CI).
pub fn build_embedder_from_env() -> std::sync::Arc<dyn EmbeddingProvider> {
    let provider = std::env::var("RAG_EMBED_PROVIDER").unwrap_or_else(|_| "ollama".to_string());
    let dim: usize = std::env::var("RAG_EMBED_DIM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(768);
    if provider.eq_ignore_ascii_case("local") {
        std::sync::Arc::new(LocalHashEmbedder::new(dim))
    } else {
        let ollama_url = std::env::var("RAG_OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434".to_string());
        let model = std::env::var("RAG_EMBED_MODEL").unwrap_or_else(|_| "nomic-embed-text".to_string());
        std::sync::Arc::new(OllamaEmbedder::new(ollama_url, model, dim))
    }
}
