//! `DOCX` loader: one text document per Word file.
//!
//! Port of the Python `WordParser` registration in `AutoFileParser`: the file is
//! parsed by [`DocxParser`] and the rendered body becomes the document text.
//! Metadata mirrors what the Python parser keeps for a Word file plus the
//! `dc:title` the core-properties part carries, when it is present.

use async_trait::async_trait;
use serde_json::Value;

use openjiuwen_common::error::Result;
use openjiuwen_common::logging::debug;

use crate::common::{Document, Metadata};
use crate::parsing::DocxParser;

use super::{file_extension, DocumentLoader};

/// Loads `.docx` files as one document holding the rendered body text.
#[derive(Debug, Default, Clone, Copy)]
pub struct DocxLoader;

impl DocxLoader {
    /// Creates a `DOCX` loader.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DocumentLoader for DocxLoader {
    fn supports(&self, source: &str) -> bool {
        file_extension(source) == ".docx"
    }

    async fn load(&self, source: &str) -> Result<Vec<Document>> {
        let document = DocxParser::new().parse_path(source).await?;
        let mut metadata = Metadata::new();
        if let Some(title) = &document.title {
            metadata.insert("title".to_owned(), Value::String(title.clone()));
        }
        metadata.insert(
            "paragraph_count".to_owned(),
            Value::from(document.paragraph_count),
        );
        metadata.insert("table_count".to_owned(), Value::from(document.table_count));
        metadata.insert(
            "source_type".to_owned(),
            Value::String("document".to_owned()),
        );
        debug!(file = source, "loaded DOCX document");
        Ok(vec![Document::new(document.text).with_metadata(metadata)])
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    use crate::loaders::fixtures::build_zip;

    /// `word/document.xml` fixture: a heading, a paragraph, and a table.
    const DOCUMENT_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Overview</w:t></w:r></w:p>
    <w:p><w:r><w:t>Body text</w:t></w:r></w:p>
    <w:tbl>
      <w:tr>
        <w:tc><w:p><w:r><w:t>Name</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>Notes</w:t></w:r></w:p></w:tc>
      </w:tr>
      <w:tr>
        <w:tc><w:p><w:r><w:t>alice</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>ok</w:t></w:r></w:p></w:tc>
      </w:tr>
    </w:tbl>
  </w:body>
</w:document>"#;

    /// `docProps/core.xml` fixture carrying `dc:title`.
    const CORE_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<cp:coreProperties xmlns:cp="urn:cp" xmlns:dc="http://purl.org/dc/elements/1.1/">
  <dc:title>Design notes</dc:title>
</cp:coreProperties>"#;

    #[tokio::test]
    async fn load_returns_one_document_with_rendered_text_and_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("notes.docx");
        std::fs::write(
            &path,
            build_zip(&[
                ("word/document.xml", DOCUMENT_XML),
                ("docProps/core.xml", CORE_XML),
            ])
            .unwrap(),
        )
        .unwrap();

        let documents = DocxLoader::new()
            .load(path.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(documents.len(), 1);
        let document = &documents[0];
        assert_eq!(
            document.text,
            "## Overview\nBody text\n| Name | Notes |\n| --- | --- |\n| alice | ok |"
        );
        assert_eq!(
            document.metadata.get("title"),
            Some(&Value::String("Design notes".to_owned()))
        );
        assert_eq!(
            document.metadata.get("paragraph_count"),
            Some(&Value::from(2))
        );
        assert_eq!(document.metadata.get("table_count"), Some(&Value::from(1)));
        assert_eq!(
            document.metadata.get("source_type"),
            Some(&Value::String("document".to_owned()))
        );
        assert_eq!(document.id.len(), 36);
    }

    #[tokio::test]
    async fn documents_without_core_properties_have_no_title_and_zero_counts() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bare.docx");
        std::fs::write(
            &path,
            build_zip(&[("word/document.xml", "<w:document><w:body/></w:document>")]).unwrap(),
        )
        .unwrap();

        let documents = DocxLoader::new()
            .load(path.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(documents[0].text, "");
        assert!(documents[0].metadata.get("title").is_none());
        assert_eq!(
            documents[0].metadata.get("paragraph_count"),
            Some(&Value::from(0))
        );
        assert_eq!(
            documents[0].metadata.get("table_count"),
            Some(&Value::from(0))
        );
    }

    #[tokio::test]
    async fn missing_and_corrupt_files_report_typed_errors() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("absent.docx");
        let error = DocxLoader::new()
            .load(missing.to_str().unwrap())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);

        let corrupt = dir.path().join("corrupt.docx");
        std::fs::write(&corrupt, b"not a zip").unwrap();
        let error = DocxLoader::new()
            .load(corrupt.to_str().unwrap())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);

        let empty_document = dir.path().join("nopart.docx");
        std::fs::write(
            &empty_document,
            build_zip(&[("docProps/core.xml", CORE_XML)]).unwrap(),
        )
        .unwrap();
        let error = DocxLoader::new()
            .load(empty_document.to_str().unwrap())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Retrieval);
        assert!(error.to_string().contains("word/document.xml"));
    }

    #[test]
    fn supports_docx_only_case_insensitively() {
        let loader = DocxLoader::new();
        assert!(loader.supports("a.docx"));
        assert!(loader.supports("A.DOCX"));
        assert!(!loader.supports("a.doc"));
        assert!(!loader.supports("a.xlsx"));
    }
}
