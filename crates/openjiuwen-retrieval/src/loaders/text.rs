//! Plain-text and Markdown loaders.
//!
//! Port of the Python `TxtMdParser` (`txt_md_parser.py`): one document holding
//! the trimmed file content. `HTMLFileParser` reuses the same `_parse` in Python;
//! in Rust [`crate::loaders::HtmlLoader`] has its own reader.

use async_trait::async_trait;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::logging::debug;

use crate::common::Document;

use super::{
    apply_load_options, file_extension, read_source, DocumentBytes, DocumentLoader, LoadOptions,
};

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
        let content = read_source(source).await?;
        Ok(vec![trimmed_document(source, &content)?])
    }

    async fn load_bytes(
        &self,
        source: &DocumentBytes,
        options: &LoadOptions,
    ) -> Result<Vec<Document>> {
        let content = String::from_utf8_lossy(&source.bytes);
        let mut documents = vec![trimmed_document(&source.name, &content)?];
        apply_load_options(&source.name, &mut documents, options);
        Ok(documents)
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
        let content = read_source(source).await?;
        Ok(vec![trimmed_document(source, &content)?])
    }

    async fn load_bytes(
        &self,
        source: &DocumentBytes,
        options: &LoadOptions,
    ) -> Result<Vec<Document>> {
        let content = String::from_utf8_lossy(&source.bytes);
        let mut documents = vec![trimmed_document(&source.name, &content)?];
        apply_load_options(&source.name, &mut documents, options);
        Ok(documents)
    }
}

/// Turns file content into one document with trimmed text.
///
/// `name` is the path or in-memory source name and appears verbatim in errors.
///
/// # Errors
///
/// Returns [`Error::Validation`] when the content holds no non-whitespace text.
fn trimmed_document(name: &str, content: &str) -> Result<Document> {
    let text = content.trim();
    if text.is_empty() {
        return Err(Error::Validation(format!("File {name} is empty")));
    }
    debug!(file = name, "loaded text document");
    Ok(Document::new(text))
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

    #[tokio::test]
    async fn text_loader_load_bytes_matches_the_path_result() {
        let source = DocumentBytes::new("notes.txt", b"  hello\nworld  \n".to_vec());
        let documents = TextLoader::new()
            .load_bytes(&source, &LoadOptions::new())
            .await
            .unwrap();
        assert_eq!(documents.len(), 1);
        assert_eq!(documents[0].text, "hello\nworld");
        assert_eq!(documents[0].id.len(), 36);
        assert_eq!(
            documents[0]
                .metadata
                .get("doc_id")
                .and_then(serde_json::Value::as_str),
            Some(documents[0].id.as_str())
        );
    }

    #[tokio::test]
    async fn text_loader_load_bytes_applies_doc_id_and_title() {
        let source = DocumentBytes::new("notes.txt", b"body".to_vec());
        let options = LoadOptions::new().with_doc_id("doc-1").with_title("Custom");
        let documents = TextLoader::new()
            .load_bytes(&source, &options)
            .await
            .unwrap();
        assert_eq!(documents.len(), 1);
        assert_eq!(documents[0].id, "doc-1");
        assert_eq!(documents[0].text, "body");
        assert_eq!(
            documents[0]
                .metadata
                .get("doc_id")
                .and_then(serde_json::Value::as_str),
            Some("doc-1")
        );
        assert_eq!(
            documents[0]
                .metadata
                .get("title")
                .and_then(serde_json::Value::as_str),
            Some("Custom")
        );
    }

    #[tokio::test]
    async fn text_and_markdown_load_bytes_reject_empty_content() {
        let source = DocumentBytes::new("empty.txt", b" \n\t".to_vec());
        let error = TextLoader::new()
            .load_bytes(&source, &LoadOptions::new())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.to_string().contains("File empty.txt is empty"),
            "unexpected message: {error}"
        );

        let source = DocumentBytes::new("empty.md", b" \n\t".to_vec());
        let error = MarkdownLoader
            .load_bytes(&source, &LoadOptions::new())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error.to_string().contains("File empty.md is empty"),
            "unexpected message: {error}"
        );
    }

    #[tokio::test]
    async fn markdown_loader_load_bytes_keeps_the_raw_markdown() {
        let source = DocumentBytes::new("readme.md", b"# Title\n\ntext\n".to_vec());
        let documents = MarkdownLoader
            .load_bytes(&source, &LoadOptions::new())
            .await
            .unwrap();
        assert_eq!(documents.len(), 1);
        assert_eq!(documents[0].text, "# Title\n\ntext");
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
