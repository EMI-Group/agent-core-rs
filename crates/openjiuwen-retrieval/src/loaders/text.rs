//! Plain-text and Markdown loaders.
//!
//! Port of the Python `TxtMdParser` (`txt_md_parser.py`): one document holding
//! the trimmed file content. `HTMLFileParser` reuses the same `_parse` in Python;
//! in Rust [`crate::loaders::HtmlLoader`] has its own reader.

use async_trait::async_trait;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::logging::debug;

use crate::common::Document;

use super::{file_extension, read_source, DocumentLoader};

/// Loads `.txt` files as a single document.
#[derive(Debug, Default, Clone, Copy)]
pub struct TextLoader;

impl TextLoader {
    /// Creates a plain-text loader.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DocumentLoader for TextLoader {
    fn supports(&self, source: &str) -> bool {
        file_extension(source) == ".txt"
    }

    async fn load(&self, source: &str) -> Result<Vec<Document>> {
        load_trimmed_document(source).await
    }
}

/// Loads `.md`/`.markdown` files as a single document.
#[derive(Debug, Default, Clone, Copy)]
pub struct MarkdownLoader;

impl MarkdownLoader {
    /// Creates a Markdown loader.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DocumentLoader for MarkdownLoader {
    fn supports(&self, source: &str) -> bool {
        matches!(file_extension(source).as_str(), ".md" | ".markdown")
    }

    async fn load(&self, source: &str) -> Result<Vec<Document>> {
        load_trimmed_document(source).await
    }
}

/// Reads `source` and returns it as one document with trimmed text.
///
/// # Errors
///
/// Returns [`Error::NotFound`] for a missing file and [`Error::Validation`] when
/// the file holds no non-whitespace content.
async fn load_trimmed_document(source: &str) -> Result<Vec<Document>> {
    let content = read_source(source).await?;
    let text = content.trim();
    if text.is_empty() {
        return Err(Error::Validation(format!("File {source} is empty")));
    }
    debug!(file = source, "loaded text document");
    Ok(vec![Document::new(text)])
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    #[tokio::test]
    async fn text_loader_trims_and_keeps_one_document() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("notes.txt");
        std::fs::write(&path, "  hello\nworld  \n").unwrap();

        let documents = TextLoader::new()
            .load(path.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(documents.len(), 1);
        assert_eq!(documents[0].text, "hello\nworld");
        assert_eq!(documents[0].id.len(), 36);
        assert!(documents[0].metadata.is_empty());
    }

    #[tokio::test]
    async fn text_loader_reports_missing_and_empty_files() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("missing.txt");
        let error = TextLoader::new()
            .load(missing.to_str().unwrap())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);

        let empty = dir.path().join("empty.txt");
        std::fs::write(&empty, " \n\t").unwrap();
        let error = TextLoader::new()
            .load(empty.to_str().unwrap())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
    }

    #[test]
    fn loaders_declare_their_extensions() {
        let text = TextLoader::new();
        assert!(text.supports("a.txt"));
        assert!(text.supports("A.TXT"));
        assert!(!text.supports("a.md"));

        let markdown = MarkdownLoader;
        assert!(markdown.supports("a.md"));
        assert!(markdown.supports("a.markdown"));
        assert!(!markdown.supports("a.txt"));
    }
}
