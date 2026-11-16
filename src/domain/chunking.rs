use std::collections::HashMap;

use crate::domain::models::DocumentChunk;

/// Splits Markdown into structural chunks by headings while preserving fenced code blocks.
pub fn chunk_markdown(content: &str, source_path: &str, max_chars: usize) -> Vec<DocumentChunk> {
    let mut sections: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut in_code_fence = false;

    for line in content.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") {
            in_code_fence = !in_code_fence;
        }
        let is_heading = !in_code_fence
            && (trimmed.starts_with("# ") || trimmed.starts_with("## ") || trimmed.starts_with("### "));
        if is_heading && !current.trim().is_empty() {
            sections.push(current.trim().to_string());
            current.clear();
        }
        current.push_str(line);
        current.push('\n');
    }
    if !current.trim().is_empty() {
        sections.push(current.trim().to_string());
    }

    let mut chunks = Vec::new();
    let mut idx = 0usize;
    let limit = max_chars.max(80);

    for sec in sections {
        if sec.len() <= limit {
            chunks.push(DocumentChunk {
                id: format!("{}#c{}", source_path, idx),
                source_path: source_path.to_string(),
                content: sec,
                chunk_index: idx,
            });
            idx += 1;
            continue;
        }

        let mut buf = String::new();
        for para in sec.split("\n\n") {
            if !buf.is_empty() && buf.len() + para.len() + 2 > limit {
                chunks.push(DocumentChunk {
                    id: format!("{}#c{}", source_path, idx),
                    source_path: source_path.to_string(),
                    content: buf.trim().to_string(),
                    chunk_index: idx,
                });
                idx += 1;
                buf.clear();
            }
            if !buf.is_empty() {
                buf.push_str("\n\n");
            }
            buf.push_str(para);
        }
        if !buf.trim().is_empty() {
            chunks.push(DocumentChunk {
                id: format!("{}#c{}", source_path, idx),
                source_path: source_path.to_string(),
                content: buf.trim().to_string(),
                chunk_index: idx,
            });
            idx += 1;
        }
    }
    chunks
}

/// Fuses dense and sparse ranked lists using Reciprocal Rank Fusion: `1 / (k + rank)`.
pub fn reciprocal_rank_fusion(
    dense: &[(String, f64)],
    sparse: &[(String, f64)],
    k: f64,
) -> Vec<(String, f64)> {
    let mut scores: HashMap<String, f64> = HashMap::new();
    for (rank, (id, _)) in dense.iter().enumerate() {
        *scores.entry(id.clone()).or_insert(0.0) += 1.0 / (k + (rank + 1) as f64);
    }
    for (rank, (id, _)) in sparse.iter().enumerate() {
        *scores.entry(id.clone()).or_insert(0.0) += 1.0 / (k + (rank + 1) as f64);
    }
    let mut out: Vec<(String, f64)> = scores.into_iter().collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// Computes `(Hit@K, MRR)` over a list of `(ranked_ids, expected_id)` pairs.
pub fn compute_mrr_and_hit_at_k(evals: &[(Vec<String>, String)], k: usize) -> (f64, f64) {
    if evals.is_empty() {
        return (0.0, 0.0);
    }
    let mut hits = 0usize;
    let mut rr_sum = 0.0f64;
    for (ranked, target) in evals {
        if let Some(pos) = ranked.iter().take(k).position(|id| id == target) {
            hits += 1;
            rr_sum += 1.0 / ((pos + 1) as f64);
        }
    }
    let n = evals.len() as f64;
    ((hits as f64) / n, rr_sum / n)
}
