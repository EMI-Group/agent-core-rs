//! `PDF` loader: one text document per PDF file.
//!
//! Port of the Python `PDFParser` registration in `AutoFileParser`: text is
//! extracted by [`PdfParser`] and the joined page text becomes the document.
//! A document with no extractable text (a scanned, image-only PDF) is reported as
//! [`Error::Unsupported`] instead
//! of yielding an empty document, so batch ingestion cannot silently index a
//! blank record; encrypted documents and object streams keep the parser's own
//! `Unsupported` messages.
//!
//! A partially decodable document still loads, and says so: `pdf_incomplete`
//! (`false`/`true`) and `pdf_skipped_filters` (the filter names of the streams the
//! parser had to skip) are added to the document metadata next to `page_count`,
//! `has_text`, and `source_type`, and a `warn` log names the skipped filters.
//!
//! [`DocumentLoader::load_bytes`] extracts text from in-memory bytes
//! ([`PdfParser::parse_bytes`]) under the same rules, so an uploaded PDF needs no
//! temporary path.

use async_trait::async_trait;
use serde_json::Value;

use openjiuwen_common::error::{Error, Result};
use openjiuwen_common::logging::{debug, warn};

use crate::common::{Document, Metadata};
use crate::parsing::{PdfDocument, PdfParser};

use super::{apply_load_options, file_extension, DocumentBytes, DocumentLoader, LoadOptions};

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

/// Renders a parsed PDF as the single document the loader returns.
///
/// `name` is the source the caller asked for (path or in-memory file name) and
/// only appears in the scanned-PDF error message.
fn document_from(document: PdfDocument, name: &str) -> Result<Document> {
    if !document.has_text {
        return Err(Error::Unsupported(format!(
            "PDF has no extractable text: {name}"
        )));
    }
    if document.incomplete {
        warn!(
            file = name,
            skipped_filters = ?document.skipped_filters,
            skipped_streams = document.skipped_streams.len(),
            "PDF text extraction skipped undecodable streams"
        );
    }
    let mut metadata = Metadata::new();
    metadata.insert("page_count".to_owned(), Value::from(document.page_count));
    metadata.insert("has_text".to_owned(), Value::Bool(document.has_text));
    metadata.insert(
        "source_type".to_owned(),
        Value::String("document".to_owned()),
    );
    metadata.insert(
        "pdf_incomplete".to_owned(),
        Value::Bool(document.incomplete),
    );
    metadata.insert(
        "pdf_skipped_filters".to_owned(),
        Value::Array(
            document
                .skipped_filters
                .iter()
                .cloned()
                .map(Value::String)
                .collect(),
        ),
    );
    Ok(Document::new(document.text).with_metadata(metadata))
}

#[async_trait]
impl DocumentLoader for PdfLoader {
    fn supports(&self, source: &str) -> bool {
        file_extension(source) == ".pdf"
    }

    async fn load(&self, source: &str) -> Result<Vec<Document>> {
        let document = PdfParser::new().parse_path(source).await?;
        let pages = document.page_count;
        let document = document_from(document, source)?;
        debug!(file = source, pages, "loaded PDF document");
        Ok(vec![document])
    }

    async fn load_bytes(
        &self,
        source: &DocumentBytes,
        options: &LoadOptions,
    ) -> Result<Vec<Document>> {
        let document = PdfParser::new().parse_bytes(&source.bytes)?;
        let document = document_from(document, &source.name)?;
        let mut documents = vec![document];
        apply_load_options(&source.name, &mut documents, options);
        debug!(
            file = source.name.as_str(),
            "loaded PDF document from bytes"
        );
        Ok(documents)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    use crate::loaders::fixtures::{pdf_document, plain_stream};
    use crate::loaders::{DocumentBytes, LoadOptions};

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
        assert_eq!(
            document.metadata.get("pdf_incomplete"),
            Some(&Value::Bool(false))
        );
        assert_eq!(
            document.metadata.get("pdf_skipped_filters"),
            Some(&Value::Array(Vec::new()))
        );
        assert_eq!(document.id.len(), 36);
    }

    #[tokio::test]
    async fn partially_decoded_pdfs_report_the_skipped_filters() {
        let body = format!(
            "{}\n3 0 obj\n<< /Filter /DCTDecode /Length 4 >>\nstream\nscan\nendstream\nendobj",
            plain_stream("BT (Readable) Tj ET")
        );
        let bytes = pdf_document(&body);
        let document = PdfLoader::new()
            .load_bytes(&DocumentBytes::new("mixed.pdf", bytes), &LoadOptions::new())
            .await
            .unwrap();

        assert_eq!(document.len(), 1);
        assert_eq!(document[0].text, "Readable");
        let metadata = &document[0].metadata;
        assert_eq!(metadata.get("pdf_incomplete"), Some(&Value::Bool(true)));
        assert_eq!(
            metadata.get("pdf_skipped_filters"),
            Some(&Value::Array(vec![Value::String("DCTDecode".to_owned())]))
        );
        // The metadata keys that predate the completeness markers still apply.
        assert_eq!(metadata.get("page_count"), Some(&Value::from(1)));
        assert_eq!(metadata.get("has_text"), Some(&Value::Bool(true)));
        assert_eq!(
            metadata.get("source_type"),
            Some(&Value::String("document".to_owned()))
        );
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

    #[tokio::test]
    async fn load_bytes_matches_the_path_loader() {
        let bytes = pdf_document(&plain_stream("BT (Hello PDF) Tj ET"));
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("paper.pdf");
        std::fs::write(&path, &bytes).unwrap();

        let from_path = PdfLoader::new().load(path.to_str().unwrap()).await.unwrap();
        let from_bytes = PdfLoader::new()
            .load_bytes(&DocumentBytes::new("paper.pdf", bytes), &LoadOptions::new())
            .await
            .unwrap();

        assert_eq!(from_bytes.len(), 1);
        assert_eq!(from_bytes[0].text, from_path[0].text);
        assert_eq!(from_bytes[0].text, "Hello PDF");
        for key in [
            "page_count",
            "has_text",
            "source_type",
            "pdf_incomplete",
            "pdf_skipped_filters",
        ] {
            assert_eq!(
                from_bytes[0].metadata.get(key),
                from_path[0].metadata.get(key),
                "metadata key {key}"
            );
        }
        assert_eq!(from_bytes[0].id.len(), 36);
        assert_eq!(
            from_bytes[0].metadata.get("doc_id"),
            Some(&Value::String(from_bytes[0].id.clone()))
        );
    }

    #[tokio::test]
    async fn load_bytes_applies_doc_id_and_title_overrides() {
        let source = DocumentBytes::new(
            "/uploads/paper.pdf",
            pdf_document(&plain_stream("BT (Hello PDF) Tj ET")),
        );
        let documents = PdfLoader::new()
            .load_bytes(
                &source,
                &LoadOptions::new().with_doc_id("doc-9").with_title("Paper"),
            )
            .await
            .unwrap();

        assert_eq!(documents.len(), 1);
        assert_eq!(documents[0].id, "doc-9");
        assert_eq!(
            documents[0].metadata.get("doc_id"),
            Some(&Value::String("doc-9".to_owned()))
        );
        assert_eq!(
            documents[0].metadata.get("title"),
            Some(&Value::String("Paper".to_owned()))
        );
        assert_eq!(
            documents[0].metadata.get("has_text"),
            Some(&Value::Bool(true))
        );
    }

    #[tokio::test]
    async fn load_bytes_of_a_scanned_pdf_is_unsupported() {
        let source = DocumentBytes::new(
            "scanned.pdf",
            pdf_document(
                "4 0 obj\n<< /Filter /DCTDecode /Length 4 >>\nstream\nscan\nendstream\nendobj",
            ),
        );
        let error = PdfLoader::new()
            .load_bytes(&source, &LoadOptions::new())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert_eq!(
            error.to_string(),
            "unsupported: PDF has no extractable text: scanned.pdf"
        );
    }

    #[tokio::test]
    async fn load_bytes_of_an_encrypted_pdf_keeps_the_parser_message() {
        let mut bytes = b"%PDF-1.7\n1 0 obj\n<< /Encrypt 2 0 R >>\nendobj\n".to_vec();
        bytes.extend_from_slice(b"%%EOF\n");
        let source = DocumentBytes::new("locked.pdf", bytes);
        let error = PdfLoader::new()
            .load_bytes(&source, &LoadOptions::new())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(
            error.to_string().contains("encrypted PDF documents"),
            "unexpected message: {error}"
        );
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
