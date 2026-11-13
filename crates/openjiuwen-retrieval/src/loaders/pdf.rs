//! `PDF` loader: one text document per PDF file.
//!
//! Port of the Python `PDFParser` registration in `AutoFileParser`: text is
//! extracted by [`PdfParser`] and the joined page text becomes the document.
//! A document with no extractable text (a scanned, image-only PDF) is reported as
//! [`Error::Unsupported`] instead
//! of yielding an empty document, so batch ingestion cannot silently index a
//! blank record; encrypted documents and object streams keep the parser's own
//! `Unsupported` messages.

use async_trait::async_trait;
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::logging::debug;

use crate::common::{Document, Metadata};
use crate::parsing::PdfParser;

use super::{file_extension, DocumentLoader};

/// Loads `.pdf` files as one document holding the extracted page text.
#[derive(Debug, Default, Clone, Copy)]
pub struct PdfLoader;

impl PdfLoader {
    /// Creates a PDF loader.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DocumentLoader for PdfLoader {
    fn supports(&self, source: &str) -> bool {
        file_extension(source) == ".pdf"
    }

    async fn load(&self, source: &str) -> Result<Vec<Document>> {
        let document = PdfParser::new().parse_path(source).await?;
        if !document.has_text {
            return Err(Error::Unsupported(format!(
                "PDF has no extractable text: {source}"
            )));
        }
        let mut metadata = Metadata::new();
        metadata.insert("page_count".to_owned(), Value::from(document.page_count));
        metadata.insert("has_text".to_owned(), Value::Bool(document.has_text));
        metadata.insert(
            "source_type".to_owned(),
            Value::String("document".to_owned()),
        );
        debug!(
            file = source,
            pages = document.page_count,
            "loaded PDF document"
        );
        Ok(vec![Document::new(document.text).with_metadata(metadata)])
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    use crate::loaders::fixtures::{pdf_document, plain_stream};

    #[tokio::test]
    async fn load_returns_one_document_with_page_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("paper.pdf");
        std::fs::write(&path, pdf_document(&plain_stream("BT (Hello PDF) Tj ET"))).unwrap();

        let documents = PdfLoader::new().load(path.to_str().unwrap()).await.unwrap();
        assert_eq!(documents.len(), 1);
        let document = &documents[0];
        assert_eq!(document.text, "Hello PDF");
        assert_eq!(document.metadata.get("page_count"), Some(&Value::from(1)));
        assert_eq!(document.metadata.get("has_text"), Some(&Value::Bool(true)));
        assert_eq!(
            document.metadata.get("source_type"),
            Some(&Value::String("document".to_owned()))
        );
        assert_eq!(document.id.len(), 36);
    }

    #[tokio::test]
    async fn scanned_pdf_without_text_is_unsupported() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("scanned.pdf");
        std::fs::write(
            &path,
            pdf_document(
                "4 0 obj\n<< /Filter /DCTDecode /Length 4 >>\nstream\nscan\nendstream\nendobj",
            ),
        )
        .unwrap();

        let error = PdfLoader::new()
            .load(path.to_str().unwrap())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(
            error.to_string().contains("PDF has no extractable text"),
            "unexpected message: {error}"
        );
    }

    #[tokio::test]
    async fn encrypted_pdfs_keep_the_parser_unsupported_message() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("locked.pdf");
        let mut bytes = b"%PDF-1.7\n1 0 obj\n<< /Encrypt 2 0 R >>\nendobj\n".to_vec();
        bytes.extend_from_slice(b"%%EOF\n");
        std::fs::write(&path, bytes).unwrap();

        let error = PdfLoader::new()
            .load(path.to_str().unwrap())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(
            error.to_string().contains("encrypted PDF documents"),
            "unexpected message: {error}"
        );
    }

    #[tokio::test]
    async fn missing_files_report_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("absent.pdf");
        let error = PdfLoader::new()
            .load(missing.to_str().unwrap())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert!(error.to_string().contains("does not exist"));
    }

    #[test]
    fn supports_pdf_only_case_insensitively() {
        let loader = PdfLoader::new();
        assert!(loader.supports("a.pdf"));
        assert!(loader.supports("A.PDF"));
        assert!(!loader.supports("a.txt"));
        assert!(!loader.supports("a.docx"));
    }
}
