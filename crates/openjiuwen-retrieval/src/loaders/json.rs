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

use super::{file_extension, read_source, DocumentLoader};

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
}

#[async_trait]
impl DocumentLoader for JsonLoader {
    fn supports(&self, source: &str) -> bool {
        file_extension(source) == ".json"
    }

    async fn load(&self, source: &str) -> Result<Vec<Document>> {
        let raw = read_source(source).await?;
        if raw.trim().is_empty() {
            return Err(Error::Validation(format!("File {source} is empty")));
        }
        let text = Self::render(source, &raw);
        Ok(vec![Document::new(text)])
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

    #[test]
    fn supports_json_only() {
        assert!(JsonLoader::new().supports("data.JSON"));
        assert!(!JsonLoader::new().supports("data.csv"));
    }
}
