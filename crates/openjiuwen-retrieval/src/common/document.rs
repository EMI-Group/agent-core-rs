//! Document data model: [`Document`] and [`TextChunk`].
//!
//! Port of the Python `openjiuwen/core/retrieval/common/document.py` models plus
//! the foundation `Document` (`base_reranker.py`). The Python wire names are
//! preserved: a document serializes as `{"id_", "text", "metadata"}` and a chunk
//! as `{"id_", "text", "doc_id", "metadata", "embedding"}`.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Free-form metadata attached to documents, chunks, and results.
///
/// Mirrors Python's `Dict[str, Any]`; insertion order is preserved so that
/// round-tripped JSON keeps the caller's field order.
pub type Metadata = serde_json::Map<String, serde_json::Value>;

/// Generates the dashed `uuid4` string Python uses for default identifiers.
fn generate_id() -> String {
    Uuid::new_v4().to_string()
}

/// A source document: text plus identity and metadata.
///
/// `id` is serialized as `id_` (Python's attribute name) and also accepts `id`
/// on input; when the field is absent a fresh `uuid4` is generated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Document {
    /// Document identifier; serialized as `id_`, defaulting to a fresh `uuid4`.
    #[serde(rename = "id_", alias = "id", default = "generate_id")]
    pub id: String,
    /// Document text content.
    pub text: String,
    /// Document metadata (empty when absent from the payload).
    #[serde(default)]
    pub metadata: Metadata,
}

impl Document {
    /// Creates a document with a generated `uuid4` id and empty metadata.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            id: generate_id(),
            text: text.into(),
            metadata: Metadata::new(),
        }
    }

    /// Creates a document with an explicit id and empty metadata.
    pub fn with_id(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            metadata: Metadata::new(),
        }
    }

    /// Replaces the metadata wholesale.
    #[must_use]
    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Inserts a single metadata entry, keeping any existing keys.
    #[must_use]
    pub fn metadata_entry(
        mut self,
        key: impl Into<String>,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// A chunk produced from a [`Document`], optionally carrying its embedding.
///
/// `id` is serialized as `id_`; `embedding` serializes as `null` when absent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextChunk {
    /// Chunk identifier; serialized as `id_`.
    #[serde(rename = "id_", alias = "id", default = "generate_id")]
    pub id: String,
    /// Chunk text content.
    pub text: String,
    /// Identifier of the parent document.
    pub doc_id: String,
    /// Chunk metadata, copied from the parent document by [`TextChunk::from_document`].
    #[serde(default)]
    pub metadata: Metadata,
    /// Embedding vector, `None` until the chunk is embedded.
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
}

impl TextChunk {
    /// Builds a chunk from a document, copying its `doc_id` and metadata.
    ///
    /// A generated `uuid4` is used when `id` is `None` or empty, matching
    /// Python's `id_ if id_ else str(uuid.uuid4())`.
    pub fn from_document(doc: &Document, text: impl Into<String>, id: Option<String>) -> Self {
        Self {
            id: id
                .filter(|value| !value.is_empty())
                .unwrap_or_else(generate_id),
            text: text.into(),
            doc_id: doc.id.clone(),
            metadata: doc.metadata.clone(),
            embedding: None,
        }
    }

    /// Attaches an embedding vector to the chunk.
    #[must_use]
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashSet;

    #[test]
    fn document_serializes_python_keys() {
        let doc = Document::with_id("doc-1", "hello")
            .metadata_entry("author", json!("alice"))
            .metadata_entry("year", json!(2024));
        let value = serde_json::to_value(&doc).unwrap();
        assert_eq!(
            value,
            json!({"id_": "doc-1", "text": "hello", "metadata": {"author": "alice", "year": 2024}})
        );
        assert!(value.get("id").is_none());
    }

    #[test]
    fn document_deserializes_alias_and_defaults_uuid() {
        let doc: Document =
            serde_json::from_value(json!({"id": "alias-id", "text": "hi"})).unwrap();
        assert_eq!(doc.id, "alias-id");
        assert!(doc.metadata.is_empty());

        let generated: Document = serde_json::from_value(json!({"text": "hi"})).unwrap();
        assert_eq!(generated.id.len(), 36);
        assert!(Uuid::parse_str(&generated.id).is_ok());
    }

    #[test]
    fn generated_ids_are_unique() {
        let ids: HashSet<String> = (0..16).map(|_| Document::new("x").id).collect();
        assert_eq!(ids.len(), 16);
    }

    #[test]
    fn text_chunk_from_document_copies_ids_and_metadata() {
        let doc = Document::with_id("doc-7", "source").metadata_entry("source", json!("file"));
        let chunk = TextChunk::from_document(&doc, "piece", None);
        assert_eq!(chunk.doc_id, "doc-7");
        assert_eq!(chunk.text, "piece");
        assert_eq!(chunk.metadata, doc.metadata);
        assert!(chunk.embedding.is_none());
        assert!(Uuid::parse_str(&chunk.id).is_ok());

        let explicit = TextChunk::from_document(&doc, "piece", Some("chunk-1".to_owned()));
        assert_eq!(explicit.id, "chunk-1");

        // Python parity: an empty id falls back to a generated uuid.
        let empty = TextChunk::from_document(&doc, "piece", Some(String::new()));
        assert!(Uuid::parse_str(&empty.id).is_ok());
    }

    #[test]
    fn text_chunk_round_trips_with_null_embedding() {
        let chunk =
            TextChunk::from_document(&Document::with_id("d", "t"), "body", Some("c".into()))
                .with_embedding(vec![0.5, -1.0]);
        let value = serde_json::to_value(&chunk).unwrap();
        assert_eq!(
            value,
            json!({"id_": "c", "text": "body", "doc_id": "d", "metadata": {}, "embedding": [0.5, -1.0]})
        );
        let restored: TextChunk = serde_json::from_value(value).unwrap();
        assert_eq!(restored, chunk);

        let bare = TextChunk::from_document(&Document::with_id("d", "t"), "body", Some("c".into()));
        let value = serde_json::to_value(&bare).unwrap();
        assert_eq!(value.get("embedding"), Some(&serde_json::Value::Null));
    }
}
