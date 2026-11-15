//! `PPTX` loader: one text document per `PowerPoint` presentation.
//!
//! Port of the Python `PptxParser` registration in `AutoFileParser`: the file is
//! parsed by [`PptxParser`] and the rendered deck becomes the document text.
//! Metadata follows the other office loaders (`source_type = "document"`) and
//! adds the slide count, whether any slide has a speaker-notes part, and the
//! `dc:title` the core-properties part carries, when it is present.
//!
//! A deck whose slide parts do not all resolve still loads, and says so:
//! `pptx_incomplete` (`false`/`true`) and `pptx_problems` (the unresolved slide
//! entries, in presentation order) are added to the document metadata, always
//! present, so an ingestion pipeline can reject or flag a lossy deck instead of
//! indexing it as if every slide had been read.

use async_trait::async_trait;
use serde_json::Value;

use openjiuwen_common::error::Result;
use openjiuwen_common::logging::debug;

use crate::common::{Document, Metadata};
use crate::parsing::{PptxDocument, PptxParser};

use super::{apply_load_options, file_extension, DocumentBytes, DocumentLoader, LoadOptions};

/// Loads `.pptx` presentations as one document holding the rendered slides.
#[derive(Debug, Default, Clone, Copy)]
pub struct PptxLoader;

impl PptxLoader {
    /// Creates a `PPTX` loader.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

/// Builds the single document for one parsed presentation.
///
/// The text is the rendered deck (`## Slide N` blocks, with `### Notes` blocks
/// when speaker notes exist), and the metadata keeps `slide_count`, `has_notes`
/// (true when any slide has a notes part, even an empty one), `source_type =
/// "document"`, and `title` only when the core properties carry one.
///
/// `pptx_incomplete` (`true` iff any slide part failed to resolve) and
/// `pptx_problems` (those unresolved entries, in presentation order; an empty
/// array for a complete deck) are always inserted.
fn presentation_document(presentation: PptxDocument) -> Document {
    let PptxDocument {
        text,
        slides,
        slide_count,
        title,
        problems,
    } = presentation;
    let mut metadata = Metadata::new();
    if let Some(title) = title {
        metadata.insert("title".to_owned(), Value::String(title));
    }
    metadata.insert("slide_count".to_owned(), Value::from(slide_count));
    let has_notes = slides.iter().any(|slide| slide.has_notes);
    metadata.insert("has_notes".to_owned(), Value::from(has_notes));
    metadata.insert(
        "source_type".to_owned(),
        Value::String("document".to_owned()),
    );
    metadata.insert(
        "pptx_incomplete".to_owned(),
        Value::Bool(!problems.is_empty()),
    );
    metadata.insert(
        "pptx_problems".to_owned(),
        Value::Array(problems.into_iter().map(Value::String).collect()),
    );
    Document::new(text).with_metadata(metadata)
}

#[async_trait]
impl DocumentLoader for PptxLoader {
    fn supports(&self, source: &str) -> bool {
        file_extension(source) == ".pptx"
    }

    async fn load(&self, source: &str) -> Result<Vec<Document>> {
        let presentation = PptxParser::new().parse_path(source).await?;
        debug!(
            file = source,
            slides = presentation.slide_count,
            problems = presentation.problems.len(),
            "loaded PPTX presentation"
        );
        Ok(vec![presentation_document(presentation)])
    }

    async fn load_bytes(
        &self,
        source: &DocumentBytes,
        options: &LoadOptions,
    ) -> Result<Vec<Document>> {
        let presentation = PptxParser::new().parse_bytes(&source.bytes)?;
        let mut documents = vec![presentation_document(presentation)];
        apply_load_options(&source.name, &mut documents, options);
        debug!(
            file = source.name.as_str(),
            "loaded PPTX presentation from in-memory bytes"
        );
        Ok(documents)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::fmt::Write as _;

    use super::*;
    use openjiuwen_common::error::ErrorKind;

    use crate::loaders::fixtures::{build_zip, pptx_slides};

    /// One `ppt/slides/slide1.xml` part with a single title text run.
    const SLIDE_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
       xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:sp>
        <p:nvSpPr><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr>
        <p:txBody><a:bodyPr/><a:p><a:r><a:t>Titled deck</a:t></a:r></a:p></p:txBody>
      </p:sp>
    </p:spTree>
  </p:cSld>
</p:sld>"#;

    /// `docProps/core.xml` fixture carrying `dc:title`.
    const CORE_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<cp:coreProperties xmlns:cp="urn:cp" xmlns:dc="http://purl.org/dc/elements/1.1/">
  <dc:title>Quarterly review</dc:title>
</cp:coreProperties>"#;

    /// Two-slide deck: a titled slide with two body lines, then a bare title.
    fn deck_bytes() -> Vec<u8> {
        pptx_slides(&[
            ("Rust rewrite", &["Agenda", "Offline"]),
            ("Closing & <thanks>", &[]),
        ])
        .unwrap()
    }

    /// A presentation part listing `ids` in order.
    fn presentation(ids: &[&str]) -> String {
        let mut entries = String::new();
        for id in ids {
            write!(entries, r#"<p:sldId id="256" r:id="{id}"/>"#).unwrap();
        }
        format!(
            r#"<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><p:sldIdLst>{entries}</p:sldIdLst></p:presentation>"#
        )
    }

    /// A relationships part mapping `(id, target)` pairs.
    fn relationships(entries: &[(&str, &str)]) -> String {
        let mut items = String::new();
        for (id, target) in entries {
            write!(
                items,
                r#"<Relationship Id="{id}" Type="slide" Target="{target}"/>"#
            )
            .unwrap();
        }
        format!(
            r#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{items}</Relationships>"#
        )
    }

    /// A two-slide deck whose second `<p:sldId>` (`rId2`) is never mapped.
    fn dangling_deck_bytes() -> Vec<u8> {
        let presentation = presentation(&["rId1", "rId2"]);
        let relationships = relationships(&[("rId1", "slides/slide1.xml")]);
        build_zip(&[
            ("ppt/presentation.xml", presentation.as_str()),
            ("ppt/_rels/presentation.xml.rels", relationships.as_str()),
            ("ppt/slides/slide1.xml", SLIDE_XML),
        ])
        .unwrap()
    }

    #[tokio::test]
    async fn load_returns_one_document_with_rendered_slides_and_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("deck.pptx");
        std::fs::write(&path, deck_bytes()).unwrap();

        let documents = PptxLoader::new()
            .load(path.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(documents.len(), 1);
        let document = &documents[0];
        assert_eq!(
            document.text,
            "## Slide 1\nRust rewrite\nAgenda\nOffline\n## Slide 2\nClosing & <thanks>"
        );
        assert_eq!(document.metadata.get("slide_count"), Some(&Value::from(2)));
        assert_eq!(
            document.metadata.get("has_notes"),
            Some(&Value::from(false))
        );
        assert_eq!(
            document.metadata.get("source_type"),
            Some(&Value::String("document".to_owned()))
        );
        assert!(document.metadata.get("title").is_none());
        assert_eq!(document.id.len(), 36);
        assert!(uuid::Uuid::parse_str(&document.id).is_ok());
    }

    #[tokio::test]
    async fn load_reads_the_core_properties_title() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("titled.pptx");
        std::fs::write(
            &path,
            build_zip(&[
                ("ppt/slides/slide1.xml", SLIDE_XML),
                ("docProps/core.xml", CORE_XML),
            ])
            .unwrap(),
        )
        .unwrap();

        let documents = PptxLoader::new()
            .load(path.to_str().unwrap())
            .await
            .unwrap();
        let document = &documents[0];
        assert_eq!(document.text, "## Slide 1\nTitled deck");
        assert_eq!(document.metadata.get("slide_count"), Some(&Value::from(1)));
        assert_eq!(
            document.metadata.get("title"),
            Some(&Value::String("Quarterly review".to_owned()))
        );
    }

    #[tokio::test]
    async fn dangling_slide_relationships_are_reported_in_the_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dangling.pptx");
        std::fs::write(&path, dangling_deck_bytes()).unwrap();

        let documents = PptxLoader::new()
            .load(path.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(documents.len(), 1);
        let document = &documents[0];
        assert_eq!(document.metadata.get("slide_count"), Some(&Value::from(2)));
        assert_eq!(
            document.metadata.get("pptx_incomplete"),
            Some(&Value::Bool(true))
        );
        let problems = document
            .metadata
            .get("pptx_problems")
            .and_then(Value::as_array)
            .expect("pptx_problems is an array");
        assert_eq!(problems.len(), 1);
        assert_eq!(
            problems[0],
            Value::String(
                "slide 2: rId2 is not mapped by ppt/_rels/presentation.xml.rels".to_owned()
            )
        );
        assert!(document.text.contains("! missing slide part: rId2"));
        assert!(document.text.contains("## Slide 2"));
    }

    #[tokio::test]
    async fn clean_decks_report_no_pptx_problems() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("clean.pptx");
        std::fs::write(&path, deck_bytes()).unwrap();

        let documents = PptxLoader::new()
            .load(path.to_str().unwrap())
            .await
            .unwrap();
        let document = &documents[0];
        assert_eq!(
            document.metadata.get("pptx_incomplete"),
            Some(&Value::Bool(false))
        );
        assert_eq!(
            document.metadata.get("pptx_problems"),
            Some(&Value::Array(Vec::new()))
        );
        assert!(!document.text.contains("! missing slide part:"));
    }

    #[tokio::test]
    async fn load_bytes_matches_the_path_documents() {
        let bytes = deck_bytes();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("deck.pptx");
        std::fs::write(&path, &bytes).unwrap();

        let from_path = PptxLoader::new()
            .load(path.to_str().unwrap())
            .await
            .unwrap();
        let from_bytes = PptxLoader::new()
            .load_bytes(&DocumentBytes::new("deck.pptx", bytes), &LoadOptions::new())
            .await
            .unwrap();
        assert_eq!(from_bytes.len(), from_path.len());
        assert_eq!(from_bytes[0].text, from_path[0].text);
        assert_eq!(
            from_bytes[0].metadata.get("slide_count"),
            from_path[0].metadata.get("slide_count")
        );
        assert_eq!(
            from_bytes[0].metadata.get("has_notes"),
            from_path[0].metadata.get("has_notes")
        );
        assert_eq!(
            from_bytes[0].metadata.get("source_type"),
            from_path[0].metadata.get("source_type")
        );
        assert_eq!(
            from_bytes[0].metadata.get("pptx_incomplete"),
            from_path[0].metadata.get("pptx_incomplete")
        );
        assert_eq!(
            from_bytes[0].metadata.get("pptx_problems"),
            from_path[0].metadata.get("pptx_problems")
        );
        assert_eq!(from_bytes[0].id.len(), 36);
    }

    #[tokio::test]
    async fn load_bytes_matches_the_path_problems_of_a_lossy_deck() {
        let bytes = dangling_deck_bytes();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("dangling.pptx");
        std::fs::write(&path, &bytes).unwrap();

        let from_path = PptxLoader::new()
            .load(path.to_str().unwrap())
            .await
            .unwrap();
        let from_bytes = PptxLoader::new()
            .load_bytes(
                &DocumentBytes::new("dangling.pptx", bytes),
                &LoadOptions::new(),
            )
            .await
            .unwrap();
        assert_eq!(from_bytes[0].text, from_path[0].text);
        for key in ["pptx_incomplete", "pptx_problems", "slide_count"] {
            assert_eq!(
                from_bytes[0].metadata.get(key),
                from_path[0].metadata.get(key),
                "{key}"
            );
        }
        assert_eq!(
            from_bytes[0].metadata.get("pptx_incomplete"),
            Some(&Value::Bool(true))
        );
    }

    #[tokio::test]
    async fn load_bytes_stamps_doc_id_and_title_from_the_options() {
        let documents = PptxLoader::new()
            .load_bytes(
                &DocumentBytes::new("deck.pptx", deck_bytes()),
                &LoadOptions::new().with_doc_id("doc-9").with_title("Deck"),
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
            Some(&Value::String("Deck".to_owned()))
        );
    }

    #[tokio::test]
    async fn missing_and_broken_decks_report_typed_errors() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("absent.pptx");
        let error = PptxLoader::new()
            .load(missing.to_str().unwrap())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);

        let corrupt = dir.path().join("corrupt.pptx");
        std::fs::write(&corrupt, b"not a zip").unwrap();
        let error = PptxLoader::new()
            .load(corrupt.to_str().unwrap())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);

        let no_slides = dir.path().join("noslides.pptx");
        std::fs::write(
            &no_slides,
            build_zip(&[("ppt/presentation.xml", "<p:presentation/>")]).unwrap(),
        )
        .unwrap();
        let error = PptxLoader::new()
            .load(no_slides.to_str().unwrap())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);

        let error = PptxLoader::new()
            .load_bytes(
                &DocumentBytes::new("corrupt.pptx", b"not a zip".to_vec()),
                &LoadOptions::new(),
            )
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
    }

    #[test]
    fn supports_pptx_only_case_insensitively() {
        let loader = PptxLoader::new();
        assert!(loader.supports("a.pptx"));
        assert!(loader.supports("A.PPTX"));
        assert!(!loader.supports("a.ppt"));
        assert!(!loader.supports("a.docx"));
    }
}
