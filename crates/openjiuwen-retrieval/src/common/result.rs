//! Retrieval result types: [`SearchResult`], [`RetrievalResult`], and
//! [`MultiKBRetrievalResult`].
//!
//! Port of the Python `openjiuwen/core/retrieval/common/retrieval_result.py`
//! models. Field names and nullability match Python; `metadata` defaults to an
//! empty object and the optional ids serialize as `null`.

use serde::{Deserialize, Serialize};

use super::document::Metadata;

/// A single vector-store hit: id, text, relevance score, and metadata.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SearchResult {
    /// Result (chunk) identifier.
    pub id: String,
    /// Result text content.
    pub text: String,
    /// Relevance score; higher is more relevant.
    pub score: f32,
    /// Result metadata (empty when absent from the payload).
    #[serde(default)]
    pub metadata: Metadata,
}

/// A retrieval result presented to callers after optional reranking.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RetrievalResult {
    /// Result text content.
    pub text: String,
    /// Relevance score; higher is more relevant.
    pub score: f32,
    /// Result metadata (empty when absent from the payload).
    #[serde(default)]
    pub metadata: Metadata,
    /// Identifier of the parent document, when known.
    #[serde(default)]
    pub doc_id: Option<String>,
    /// Identifier of the source chunk, when known.
    #[serde(default)]
    pub chunk_id: Option<String>,
}

/// A retrieval result merged across several knowledge bases.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MultiKBRetrievalResult {
    /// Result text content.
    pub text: String,
    /// Relevance score; higher is more relevant.
    pub score: f32,
    /// Unmerged relevance score.
    pub raw_score: f32,
    /// Unmerged score scaled into `[0, 1]`.
    pub raw_score_scaled: f32,
    /// Knowledge base ids in which this result was found.
    #[serde(default)]
    pub kb_ids: Vec<String>,
    /// Result metadata (empty when absent from the payload).
    #[serde(default)]
    pub metadata: Metadata,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn search_result_round_trips() {
        let result = SearchResult {
            id: "c1".to_owned(),
            text: "hello".to_owned(),
            score: 0.75,
            metadata: Metadata::new(),
        };
        let value = serde_json::to_value(&result).unwrap();
        assert_eq!(
            value,
            json!({"id": "c1", "text": "hello", "score": 0.75, "metadata": {}})
        );
        let restored: SearchResult = serde_json::from_value(value).unwrap();
        assert_eq!(restored, result);
    }

    #[test]
    fn retrieval_result_optionals_serialize_as_null() {
        let value = serde_json::to_value(RetrievalResult::default()).unwrap();
        assert_eq!(
            value,
            json!({"text": "", "score": 0.0, "metadata": {}, "doc_id": null, "chunk_id": null})
        );
    }

    #[test]
    fn multi_kb_result_defaults_collections() {
        let parsed: MultiKBRetrievalResult = serde_json::from_value(json!({
            "text": "t",
            "score": 1.0,
            "raw_score": 2.0,
            "raw_score_scaled": 0.5
        }))
        .unwrap();
        assert_eq!(parsed.kb_ids, Vec::<String>::new());
        assert_eq!(parsed.metadata, Metadata::new());
        assert_eq!(parsed.raw_score, 2.0);
    }
}
