//! `JSON` file loader.
//!
//! Port of the Python `JSONParser` (`json_parser.py`): the file is parsed and
//! re-serialized with two-space indentation so the indexed text is stable; a
//! malformed file keeps its raw text instead of failing (Python parity).

use async_trait::async_trait;
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::logging::error;

use crate::common::Document;

use super::{
    apply_load_options, file_extension, read_source, DocumentBytes, DocumentLoader, LoadOptions,
};

/// Loads `.json` files as one document.
#[derive(Debug, Default, Clone, Copy)]
pub struct JsonLoader;

impl JsonLoader {
    /// Creates a `JSON` loader.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Turns raw file content into the document text.
    ///
    /// Valid `JSON` is pretty-printed (Python's `json.dumps(indent=2,
    /// ensure_ascii=False)`); malformed content is returned verbatim.
    fn render(source: &str, raw: &str) -> String {
        match serde_json::from_str::<Value>(raw) {
            Ok(value) => serde_json::to_string_pretty(&value).unwrap_or_else(|_| raw.to_owned()),
            Err(error) => {
                error!(file = source, %error, "JSON format error, keeping the raw text");
                raw.to_owned()
            }
        }
    }

    /// Turns raw content into the loader's single document.
    ///
    /// `name` is the path or in-memory source name and appears in errors.
    fn document(name: &str, raw: &str) -> Result<Document> {
        if raw.trim().is_empty() {
            return Err(Error::Validation(format!("File {name} is empty")));
        }
        Ok(Document::new(Self::render(name, raw)))
    }
}

#[async_trait]
impl DocumentLoader for JsonLoader {
    fn supports(&self, source: &str) -> bool {
        file_extension(source) == ".json"
    }

    async fn load(&self, source: &str) -> Result<Vec<Document>> {
        let raw = read_source(source).await?;
        Ok(vec![Self::document(source, &raw)?])
    }

    async fn load_bytes(
        &self,
        source: &DocumentBytes,
        options: &LoadOptions,
    ) -> Result<Vec<Document>> {
        let raw = String::from_utf8_lossy(&source.bytes);
        let mut documents = vec![Self::document(&source.name, &raw)?];
        apply_load_options(&source.name, &mut documents, options);
        Ok(documents)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    #[tokio::test]
    async fn valid_json_is_pretty_printed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        std::fs::write(&path, r#"{"a":1,"nested":{"b":[true,null]}}"#).unwrap();

        let documents = JsonLoader::new()
            .load(path.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(documents.len(), 1);
        assert!(documents[0].text.contains('\n'));
        let parsed: Value = serde_json::from_str(&documents[0].text).unwrap();
        assert_eq!(parsed["nested"]["b"][0], Value::Bool(true));
    }

    #[tokio::test]
    async fn malformed_json_keeps_the_raw_text() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("broken.json");
        std::fs::write(&path, "{not json}\n").unwrap();

        let documents = JsonLoader::new()
            .load(path.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(documents[0].text, "{not json}\n");
    }

    #[tokio::test]
    async fn empty_and_missing_files_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let empty = dir.path().join("empty.json");
        std::fs::write(&empty, "\n \n").unwrap();
        let error = JsonLoader::new()
            .load(empty.to_str().unwrap())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);

        let missing = dir.path().join("missing.json");
        let error = JsonLoader::new()
            .load(missing.to_str().unwrap())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);
    }

    #[tokio::test]
    async fn load_bytes_pretty_prints_valid_json_like_the_path() {
        let source = DocumentBytes::new(
            "config.json",
            br#"{"a":1,"nested":{"b":[true,null]}}"#.to_vec(),
        );
        let documents = JsonLoader::new()
            .load_bytes(&source, &LoadOptions::new())
            .await
            .unwrap();
        assert_eq!(documents.len(), 1);
        assert!(documents[0].text.contains('\n'));
        let parsed: Value = serde_json::from_str(&documents[0].text).unwrap();
        assert_eq!(parsed["nested"]["b"][0], Value::Bool(true));
        assert_eq!(
            documents[0].metadata.get("doc_id").and_then(Value::as_str),
            Some(documents[0].id.as_str())
        );
    }

    #[tokio::test]
    async fn load_bytes_keeps_malformed_json_raw() {
        let source = DocumentBytes::new("broken.json", b"{not json}\n".to_vec());
        let documents = JsonLoader::new()
            .load_bytes(&source, &LoadOptions::new())
            .await
            .unwrap();
        assert_eq!(documents[0].text, "{not json}\n");
    }

    #[tokio::test]
    async fn load_bytes_applies_doc_id_and_title() {
        let source = DocumentBytes::new("config.json", b"{}".to_vec());
        let options = LoadOptions::new().with_doc_id("doc-2").with_title("Config");
        let documents = JsonLoader::new()
            .load_bytes(&source, &options)
            .await
            .unwrap();
        assert_eq!(documents[0].id, "doc-2");
        assert_eq!(
            documents[0].metadata.get("title").and_then(Value::as_str),
            Some("Config")
        );
    }

    #[tokio::test]
    async fn load_bytes_rejects_empty_content() {
        let source = DocumentBytes::new("empty.json", b"\n \n".to_vec());
        let error = JsonLoader::new()
            .load_bytes(&source, &LoadOptions::new())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.to_string().contains("File empty.json is empty"),
            "unexpected message: {error}"
        );
    }

    #[test]
    fn supports_json_only() {
        assert!(JsonLoader::new().supports("data.JSON"));
        assert!(!JsonLoader::new().supports("data.csv"));
    }
}
