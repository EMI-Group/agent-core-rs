//! `ODT` loader: one text document per `OpenDocument` Text file.
//!
//! Port of the `odt` registration in Python's `AutoFileParser`: the file is
//! parsed by [`OdtParser`] and the rendered body becomes the document text.
//! Metadata follows the other office loaders (`source_type = "document"`) and
//! adds the paragraph and table counts plus the `dc:title` the `meta.xml` part
//! carries, when it is present.
//!
//! Elements the parser renders no text for (drawings, footnotes, tables of
//! contents, tracked changes, annotations, unknown blocks) are not dropped
//! silently: `odt_unsupported_elements` lists each one as
//! `<prefixed name> (×<count>)` and `odt_incomplete` flags whether the list is
//! non-empty, next to the other metadata keys.

use async_trait::async_trait;
use serde_json::Value;

use openjiuwen_common::error::Result;
use openjiuwen_common::logging::debug;

use crate::common::{Document, Metadata};
use crate::parsing::{OdtDocument, OdtParser};

use super::{apply_load_options, file_extension, DocumentBytes, DocumentLoader, LoadOptions};

/// Loads `.odt` documents as one document holding the rendered body text.
#[derive(Debug, Default, Clone, Copy)]
pub struct OdtLoader;

impl OdtLoader {
    /// Creates an `ODT` loader.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

/// Builds the single document for one parsed `ODT` file.
///
/// The text is [`OdtDocument::text`] (Markdown-flavoured headings, paragraphs,
/// and tables joined by newlines), and the metadata keeps `paragraph_count`,
/// `table_count`, `source_type = "document"`, and `title` only when the
/// properties part carries one. `odt_unsupported_elements` (a JSON array of the
/// elements that contributed no text) and `odt_incomplete` (`true` iff that
/// array is non-empty) are always inserted, so a consumer can tell a complete
/// extraction from a partial one without parsing logs.
fn text_document(document: OdtDocument) -> Document {
    let OdtDocument {
        text,
        paragraph_count,
        table_count,
        title,
        unsupported_elements,
    } = document;
    let mut metadata = Metadata::new();
    if let Some(title) = title {
        metadata.insert("title".to_owned(), Value::String(title));
    }
    metadata.insert("paragraph_count".to_owned(), Value::from(paragraph_count));
    metadata.insert("table_count".to_owned(), Value::from(table_count));
    metadata.insert(
        "source_type".to_owned(),
        Value::String("document".to_owned()),
    );
    metadata.insert(
        "odt_incomplete".to_owned(),
        Value::Bool(!unsupported_elements.is_empty()),
    );
    metadata.insert(
        "odt_unsupported_elements".to_owned(),
        Value::Array(
            unsupported_elements
                .into_iter()
                .map(Value::String)
                .collect(),
        ),
    );
    Document::new(text).with_metadata(metadata)
}

#[async_trait]
impl DocumentLoader for OdtLoader {
    fn supports(&self, source: &str) -> bool {
        file_extension(source) == ".odt"
    }

    async fn load(&self, source: &str) -> Result<Vec<Document>> {
        let document = OdtParser::new().parse_path(source).await?;
        debug!(
            file = source,
            paragraphs = document.paragraph_count,
            "loaded ODT document"
        );
        Ok(vec![text_document(document)])
    }

    async fn load_bytes(
        &self,
        source: &DocumentBytes,
        options: &LoadOptions,
    ) -> Result<Vec<Document>> {
        let document = OdtParser::new().parse_bytes(&source.bytes)?;
        let mut documents = vec![text_document(document)];
        apply_load_options(&source.name, &mut documents, options);
        debug!(
            file = source.name.as_str(),
            "loaded ODT document from in-memory bytes"
        );
        Ok(documents)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    use crate::loaders::fixtures::{build_zip, odt_document};

    /// `content.xml` fixture with one paragraph and one two-column table row.
    const CONTENT_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<office:document-content
    xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
    xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
    xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0">
  <office:body><office:text>
    <text:p>Body text</text:p>
    <table:table>
      <table:table-row>
        <table:table-cell><text:p>Name</text:p></table:table-cell>
        <table:table-cell><text:p>Notes</text:p></table:table-cell>
      </table:table-row>
      <table:table-row>
        <table:table-cell><text:p>alice</text:p></table:table-cell>
        <table:table-cell><text:p>ok</text:p></table:table-cell>
      </table:table-row>
    </table:table>
  </office:text></office:body>
</office:document-content>"#;

    /// `meta.xml` fixture carrying `dc:title`.
    const META_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<office:document-meta
    xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
    xmlns:dc="http://purl.org/dc/elements/1.1/">
  <office:meta><dc:title>Meeting notes</dc:title></office:meta>
</office:document-meta>"#;

    /// `content.xml` fixture with a body paragraph, a drawing, and a footnote.
    const PARTIAL_CONTENT_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<office:document-content
    xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
    xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
    xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0">
  <office:body><office:text>
    <text:p>Rendered <draw:frame><draw:image href="Pictures/a.png"/></draw:frame>body</text:p>
    <text:p>Footnote<text:note><text:note-citation>1</text:note-citation>
      <text:note-body><text:p>hidden</text:p></text:note-body></text:note></text:p>
  </office:text></office:body>
</office:document-content>"#;

    /// Two-paragraph document built by the shared fixture.
    fn document_bytes() -> Vec<u8> {
        odt_document(&["Intro", "Body & <text>"]).unwrap()
    }

    #[tokio::test]
    async fn load_returns_one_document_with_rendered_text_and_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("notes.odt");
        std::fs::write(&path, document_bytes()).unwrap();

        let documents = OdtLoader::new().load(path.to_str().unwrap()).await.unwrap();
        assert_eq!(documents.len(), 1);
        let document = &documents[0];
        assert_eq!(document.text, "Intro\nBody & <text>");
        assert_eq!(
            document.metadata.get("paragraph_count"),
            Some(&Value::from(2))
        );
        assert_eq!(document.metadata.get("table_count"), Some(&Value::from(0)));
        assert_eq!(
            document.metadata.get("source_type"),
            Some(&Value::String("document".to_owned()))
        );
        assert!(document.metadata.get("title").is_none());
        assert_eq!(
            document.metadata.get("odt_incomplete"),
            Some(&Value::Bool(false))
        );
        assert_eq!(
            document.metadata.get("odt_unsupported_elements"),
            Some(&Value::Array(Vec::new()))
        );
        assert_eq!(document.id.len(), 36);
        assert!(uuid::Uuid::parse_str(&document.id).is_ok());
    }

    #[tokio::test]
    async fn partial_documents_report_the_unsupported_elements() {
        let bytes = build_zip(&[("content.xml", PARTIAL_CONTENT_XML)]).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("partial.odt");
        std::fs::write(&path, &bytes).unwrap();

        let documents = OdtLoader::new().load(path.to_str().unwrap()).await.unwrap();
        let document = &documents[0];
        assert_eq!(document.text, "Rendered body\nFootnote");
        assert_eq!(
            document.metadata.get("odt_incomplete"),
            Some(&Value::Bool(true))
        );
        let elements = document
            .metadata
            .get("odt_unsupported_elements")
            .and_then(Value::as_array)
            .expect("odt_unsupported_elements is a JSON array");
        assert_eq!(
            elements,
            &vec![
                Value::String("draw:frame (×1)".to_owned()),
                Value::String("draw:image (×1)".to_owned()),
                Value::String("text:note (×1)".to_owned()),
            ]
        );
        assert!(
            elements
                .iter()
                .any(|value| value.as_str() == Some("draw:image (×1)")),
            "drawings are reported: {elements:?}"
        );
        // The keys that predate the completeness markers still apply.
        assert_eq!(
            document.metadata.get("paragraph_count"),
            Some(&Value::from(2))
        );
        assert_eq!(document.metadata.get("table_count"), Some(&Value::from(0)));
        assert_eq!(
            document.metadata.get("source_type"),
            Some(&Value::String("document".to_owned()))
        );

        // In-memory loading reports exactly the same list.
        let from_bytes = OdtLoader::new()
            .load_bytes(
                &DocumentBytes::new("partial.odt", bytes),
                &LoadOptions::new(),
            )
            .await
            .unwrap();
        for key in ["odt_incomplete", "odt_unsupported_elements"] {
            assert_eq!(
                from_bytes[0].metadata.get(key),
                document.metadata.get(key),
                "metadata key {key}"
            );
        }
    }

    #[tokio::test]
    async fn load_reads_the_properties_title_and_counts_tables() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("table.odt");
        std::fs::write(
            &path,
            build_zip(&[("content.xml", CONTENT_XML), ("meta.xml", META_XML)]).unwrap(),
        )
        .unwrap();

        let documents = OdtLoader::new().load(path.to_str().unwrap()).await.unwrap();
        let document = &documents[0];
        assert_eq!(
            document.text,
            "Body text\n| Name | Notes |\n| --- | --- |\n| alice | ok |"
        );
        assert_eq!(
            document.metadata.get("paragraph_count"),
            Some(&Value::from(1))
        );
        assert_eq!(document.metadata.get("table_count"), Some(&Value::from(1)));
        assert_eq!(
            document.metadata.get("title"),
            Some(&Value::String("Meeting notes".to_owned()))
        );
    }

    #[tokio::test]
    async fn load_bytes_matches_the_path_documents() {
        let bytes = document_bytes();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("notes.odt");
        std::fs::write(&path, &bytes).unwrap();

        let from_path = OdtLoader::new().load(path.to_str().unwrap()).await.unwrap();
        let from_bytes = OdtLoader::new()
            .load_bytes(&DocumentBytes::new("notes.odt", bytes), &LoadOptions::new())
            .await
            .unwrap();
        assert_eq!(from_bytes.len(), from_path.len());
        assert_eq!(from_bytes[0].text, from_path[0].text);
        assert_eq!(
            from_bytes[0].metadata.get("paragraph_count"),
            from_path[0].metadata.get("paragraph_count")
        );
        assert_eq!(
            from_bytes[0].metadata.get("table_count"),
            from_path[0].metadata.get("table_count")
        );
        assert_eq!(
            from_bytes[0].metadata.get("source_type"),
            from_path[0].metadata.get("source_type")
        );
        assert_eq!(
            from_bytes[0].metadata.get("odt_incomplete"),
            from_path[0].metadata.get("odt_incomplete")
        );
        assert_eq!(
            from_bytes[0].metadata.get("odt_unsupported_elements"),
            from_path[0].metadata.get("odt_unsupported_elements")
        );
        assert_eq!(from_bytes[0].id.len(), 36);
    }

    #[tokio::test]
    async fn load_bytes_stamps_doc_id_and_title_from_the_options() {
        let documents = OdtLoader::new()
            .load_bytes(
                &DocumentBytes::new("notes.odt", document_bytes()),
                &LoadOptions::new().with_doc_id("doc-9").with_title("Notes"),
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
            Some(&Value::String("Notes".to_owned()))
        );
    }

    #[tokio::test]
    async fn missing_and_broken_documents_report_typed_errors() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("absent.odt");
        let error = OdtLoader::new()
            .load(missing.to_str().unwrap())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);

        let corrupt = dir.path().join("corrupt.odt");
        std::fs::write(&corrupt, b"not a zip").unwrap();
        let error = OdtLoader::new()
            .load(corrupt.to_str().unwrap())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);

        let no_content = dir.path().join("nocontent.odt");
        std::fs::write(&no_content, build_zip(&[("meta.xml", META_XML)]).unwrap()).unwrap();
        let error = OdtLoader::new()
            .load(no_content.to_str().unwrap())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);

        let error = OdtLoader::new()
            .load_bytes(
                &DocumentBytes::new("corrupt.odt", b"not a zip".to_vec()),
                &LoadOptions::new(),
            )
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
    }

    #[test]
    fn supports_odt_only_case_insensitively() {
        let loader = OdtLoader::new();
        assert!(loader.supports("a.odt"));
        assert!(loader.supports("A.ODT"));
        assert!(!loader.supports("a.ott"));
        assert!(!loader.supports("a.docx"));
    }
}
