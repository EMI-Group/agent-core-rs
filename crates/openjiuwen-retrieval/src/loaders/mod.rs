//! Document loaders for text-like and office sources.
//!
//! Port of the offline subset of the Python
//! `openjiuwen/core/retrieval/indexing/processor/parser/` package: plain text,
//! Markdown, `JSON`, `CSV`/`TSV`, local `HTML`, `HTTP(S)` URLs, `PDF`, `DOCX`,
//! `XLSX`, images, and the auto-dispatch behavior of `AutoFileParser` /
//! `AutoLinkParser` / `AutoParser`.
//!
//! The binary formats are read in-crate by [`crate::parsing`] (no external
//! office or PDF crate) and images by [`crate::multimodal::ImageLoader`]; both
//! are registered in [`LoaderRegistry::with_defaults`], so [`AutoLoader`] picks
//! them up by extension. Formats with no registered loader still fail with
//! [`openjiuwen_common::error::Error::Unsupported`].

pub mod auto;
pub mod csv;
pub mod docx;
pub mod fixtures;
pub mod html;
pub mod json;
pub mod pdf;
pub mod text;
pub mod url;
pub mod xlsx;

pub use auto::{AutoLoader, LoaderRegistry};
pub use csv::CsvLoader;
pub use docx::DocxLoader;
pub use html::{extract_title, html_to_text, HtmlLoader};
pub use json::JsonLoader;
pub use pdf::PdfLoader;
pub use text::{MarkdownLoader, TextLoader};
pub use url::UrlLoader;
pub use xlsx::XlsxLoader;

use async_trait::async_trait;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use openjiuwen_common::error::{Error, Result};

use crate::common::Document;

/// Per-call overrides for the documents a loader produces.
///
/// Mirrors the `doc_id` and `file_name` arguments of Python's `Parser.parse`:
/// `doc_id` replaces the generated id (verbatim for single-document loaders and
/// as the id base prefix for multi-document sources), and `title` replaces the
/// metadata title. Both fields are optional, tolerant of unknown JSON fields,
/// and default to `None`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoadOptions {
    /// Document id override; `None` keeps the loader-generated ids.
    #[serde(default)]
    pub doc_id: Option<String>,
    /// Document title override; `None` keeps the loader-provided title.
    #[serde(default)]
    pub title: Option<String>,
}

impl LoadOptions {
    /// Creates options that change nothing.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the document id override.
    #[must_use]
    pub fn with_doc_id(mut self, id: impl Into<String>) -> Self {
        self.doc_id = Some(id.into());
        self
    }

    /// Sets the document title override.
    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }
}

/// An in-memory document source: a display file name plus its bytes.
///
/// `name` decides extension dispatch, the id base, and the error messages, so a
/// caller passes the file name the bytes came from (`"report.docx"`) rather than
/// a path that may not exist.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DocumentBytes {
    /// File name (with extension) the bytes belong to.
    pub name: String,
    /// Raw document bytes.
    pub bytes: Vec<u8>,
}

impl DocumentBytes {
    /// Creates an in-memory source from a name and its bytes.
    pub fn new(name: impl Into<String>, bytes: impl Into<Vec<u8>>) -> Self {
        Self {
            name: name.into(),
            bytes: bytes.into(),
        }
    }

    /// Returns the lowercase extension of [`DocumentBytes::name`], with its dot.
    ///
    /// The rule is [`file_extension`]: case-insensitive, last suffix only, and an
    /// empty string when the name has no extension.
    #[must_use]
    pub fn extension(&self) -> String {
        file_extension(&self.name)
    }

    /// Returns the file name these bytes are labelled with.
    #[must_use]
    pub fn source(&self) -> &str {
        &self.name
    }

    /// Returns the byte length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Returns `true` when the source holds no bytes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

/// A source of [`Document`]s addressed by a path, a URL, or in-memory bytes.
///
/// Mirrors the Python `Parser` ABC (`parse`/`supports`) without the streaming
/// `lazy_parse` variant, which has no caller in the Rust port. The byte-input
/// path ([`DocumentLoader::load_bytes`]) is a Rust extension: Python's parsers
/// only accept paths.
#[async_trait]
pub trait DocumentLoader: Send + Sync {
    /// Returns `true` when this loader accepts `source`.
    fn supports(&self, source: &str) -> bool;

    /// Loads every document contained in `source`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFound`] for a missing file, [`Error::Validation`] for
    /// empty content, [`Error::Unsupported`] for an unsupported format, and
    /// [`Error::Retrieval`]/[`Error::Transport`] for network or I/O failures.
    async fn load(&self, source: &str) -> Result<Vec<Document>>;

    /// Loads every document contained in `source`, then applies `options`.
    ///
    /// The id and title rules match [`AutoLoader::load_as`]: `doc_id` is used
    /// verbatim for single-document loaders and as the base prefix for
    /// multi-document sources (CSV/TSV, XLSX), while `title` overwrites the
    /// metadata title. Every returned document carries `metadata["doc_id"]` set
    /// to its final id, and an empty id is replaced by a generated `uuid4`.
    ///
    /// # Errors
    ///
    /// Same as [`DocumentLoader::load`].
    async fn load_with_options(
        &self,
        source: &str,
        options: &LoadOptions,
    ) -> Result<Vec<Document>> {
        let mut documents = self.load(source).await?;
        apply_load_options(source, &mut documents, options);
        Ok(documents)
    }

    /// Loads every document contained in in-memory `source`.
    ///
    /// `source.name` supplies the file name (extension, id base, and error
    /// messages) and `source.bytes` is parsed directly — no temporary file is
    /// written. `options` are applied exactly as in
    /// [`DocumentLoader::load_with_options`].
    ///
    /// The default implementation keeps loaders that need a path or a network
    /// source (and downstream implementors) compiling by failing with a typed
    /// [`Error::Unsupported`]; every loader in this module overrides it.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] when the loader has no byte-input path, plus the
    /// same errors as [`DocumentLoader::load`] for malformed bytes.
    async fn load_bytes(
        &self,
        source: &DocumentBytes,
        options: &LoadOptions,
    ) -> Result<Vec<Document>> {
        let loader = std::any::type_name::<Self>()
            .rsplit("::")
            .next()
            .unwrap_or("loader");
        Err(Error::Unsupported(format!(
            "{loader} does not support in-memory bytes: {}",
            source.name
        )))
    }
}

/// Applies `options` to the documents a loader produced from `source`.
///
/// Shared by [`DocumentLoader::load_with_options`] and
/// [`AutoLoader::load_as`]/[`AutoLoader::load_bytes`] so the Python parity rules
/// live in one place: a multi-document id keeps its `_s{sheet}_r{row}` /
/// `_s{sheet}_c{col}` suffix behind the replacement base, and every document
/// ends with `metadata["doc_id"]` equal to its final id.
pub(crate) fn apply_load_options(source: &str, documents: &mut [Document], options: &LoadOptions) {
    let doc_id = options
        .doc_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let title = options.title.as_deref().map(str::trim);
    for document in documents.iter_mut() {
        if let Some(doc_id) = doc_id {
            apply_doc_id(source, doc_id, &mut document.id);
        }
        if document.id.trim().is_empty() {
            document.id = generate_document_id();
        }
        let id = document.id.clone();
        document
            .metadata
            .insert("doc_id".to_owned(), Value::String(id));
        if let Some(title) = title {
            document
                .metadata
                .insert("title".to_owned(), Value::String(title.to_owned()));
        }
    }
}

/// Applies the caller-supplied `doc_id` to one document id.
///
/// Multi-document file loaders generate ids prefixed with a base string: `CSV`/`TSV`
/// use the whole `source` path, `XLSX` uses the file stem. Either prefix is
/// replaced by `doc_id` (`format!("{doc_id}{rest}")`), so every generated id stays
/// distinct. Every other loader's id is replaced verbatim.
pub(crate) fn apply_doc_id(source: &str, doc_id: &str, id: &mut String) {
    let stem = file_stem(source);
    for base in [source, stem.as_str()] {
        if base.is_empty() {
            continue;
        }
        if let Some(rest) = id.strip_prefix(base) {
            // Only a multi-document suffix (`_s{sheet}_r{row}` / `_s{sheet}_c{col}`)
            // makes the prefix a base id; a single-document id is replaced whole.
            if rest.starts_with("_s") {
                *id = format!("{doc_id}{rest}");
                return;
            }
        }
    }
    doc_id.clone_into(id);
}

/// Generates the dashed `uuid4` string Python uses for default document ids.
pub(crate) fn generate_document_id() -> String {
    Uuid::new_v4().to_string()
}

/// Returns the lowercase final extension of `source`, including the leading dot.
///
/// Matches Python's `os.path.splitext(path)[-1].lower()`: only the last suffix
/// is considered, a dot-file (`.env`) has no extension, and a source without a
/// suffix yields an empty string.
pub(crate) fn file_extension(source: &str) -> String {
    let name = source.rsplit(['/', '\\']).next().unwrap_or(source);
    match name.rfind('.') {
        Some(index) if index > 0 => name[index..].to_lowercase(),
        _ => String::new(),
    }
}

/// Returns `true` when `source` is an absolute `HTTP(S)` URL.
///
/// Python parity with `HTTP_URL_PATTERN = ^https?://\S+`: surrounding whitespace
/// is ignored, the scheme is case-insensitive, and a URL never contains
/// whitespace.
pub(crate) fn is_url(source: &str) -> bool {
    let source = source.trim();
    if source.is_empty() || source.chars().any(char::is_whitespace) {
        return false;
    }
    let lowered = source.to_ascii_lowercase();
    ["http://", "https://"].iter().any(|scheme| {
        lowered
            .strip_prefix(scheme)
            .is_some_and(|rest| !rest.is_empty())
    })
}

/// Returns the stem of `source`: the last path component without its extension.
///
/// Falls back to `source` itself when the path has no file name (a trailing
/// separator), mirroring Python's `os.path.splitext(os.path.basename(path))[0]`
/// usage for sheet and document titles.
pub(crate) fn file_stem(source: &str) -> String {
    std::path::Path::new(source)
        .file_stem()
        .and_then(std::ffi::OsStr::to_str)
        .map_or_else(|| source.to_owned(), str::to_owned)
}

/// Reads `source` as (lossy) UTF-8 text.
///
/// Python detects the encoding with `charset_normalizer` and falls back to
/// `errors="ignore"`; the Rust port keeps the lossy UTF-8 replacement instead.
///
/// # Errors
///
/// Returns [`Error::NotFound`] with Python's message (`File <source> does not
/// exist`) for a missing file, and [`Error::Transport`] for any other I/O error.
async fn read_source(source: &str) -> Result<String> {
    match tokio::fs::read(source).await {
        Ok(bytes) => Ok(String::from_utf8_lossy(&bytes).into_owned()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            Err(Error::NotFound(format!("File {source} does not exist")))
        }
        Err(error) => Err(Error::Transport(format!("reading {source}: {error}"))),
    }
}

/// Compiles one of the module's hard-coded regular expressions.
///
/// The patterns are literals covered by unit tests, so construction can only fail
/// on a programming error.
#[allow(clippy::expect_used)]
fn compile_pattern(pattern: &str) -> Regex {
    Regex::new(pattern).expect("loader patterns are valid regular expressions")
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn generated_document_ids_are_unique_uuids() {
        let first = generate_document_id();
        let second = generate_document_id();
        assert_ne!(first, second);
        assert!(Uuid::parse_str(&first).is_ok());
    }

    #[test]
    fn file_extension_matches_python_splitext() {
        assert_eq!(file_extension("report.CSV"), ".csv");
        assert_eq!(file_extension("/tmp/data.tar.gz"), ".gz");
        assert_eq!(file_extension("notes"), "");
        assert_eq!(file_extension(".env"), "");
        assert_eq!(file_extension("C:\\docs\\page.HTML"), ".html");
        assert_eq!(file_extension("trailing."), ".");
    }

    #[test]
    fn is_url_accepts_http_schemes_only() {
        assert!(is_url("https://example.com/a?b=1"));
        assert!(is_url("  HTTP://example.com  "));
        assert!(!is_url("https://example.com/a b"));
        assert!(!is_url("ftp://example.com"));
        assert!(!is_url("https://"));
        assert!(!is_url("/tmp/file.txt"));
        assert!(!is_url(""));
    }

    #[test]
    fn is_url_is_not_prefix_matched_on_the_path() {
        assert!(!is_url("file:///https://example.com"));
        assert!(is_url("http://x"));
    }

    #[test]
    fn patterns_compile() {
        assert!(compile_pattern(r"\s+").is_match("a b"));
    }

    #[test]
    fn load_options_default_to_nothing_and_chain() {
        let options = LoadOptions::default();
        assert_eq!(options, LoadOptions::new());
        assert_eq!(options.doc_id, None);
        assert_eq!(options.title, None);

        let options = LoadOptions::new().with_doc_id("doc-1").with_title("Report");
        assert_eq!(options.doc_id.as_deref(), Some("doc-1"));
        assert_eq!(options.title.as_deref(), Some("Report"));
    }

    #[test]
    fn load_options_round_trip_through_json() {
        let options = LoadOptions::new().with_doc_id("doc-1");
        let json = serde_json::to_string(&options).unwrap();
        assert_eq!(json, r#"{"doc_id":"doc-1","title":null}"#);
        let parsed: LoadOptions = serde_json::from_str(r#"{"title":"T","unknown":1}"#).unwrap();
        assert_eq!(parsed.title.as_deref(), Some("T"));
        assert_eq!(parsed.doc_id, None);
    }

    #[test]
    fn document_bytes_expose_name_extension_and_length() {
        let source = DocumentBytes::new("Report.DOCX", vec![1_u8, 2, 3]);
        assert_eq!(source.source(), "Report.DOCX");
        assert_eq!(source.name, "Report.DOCX");
        assert_eq!(source.extension(), ".docx");
        assert_eq!(source.len(), 3);
        assert!(!source.is_empty());

        let nameless = DocumentBytes::new("README", Vec::new());
        assert_eq!(nameless.extension(), "");
        assert!(nameless.is_empty());
        assert_eq!(nameless.len(), 0);

        let from_slice = DocumentBytes::new("notes.txt", &b"hi"[..]);
        assert_eq!(from_slice.bytes, b"hi".to_vec());
    }

    #[test]
    fn document_bytes_round_trip_through_json() {
        let source = DocumentBytes::new("a.txt", b"hi".to_vec());
        let json = serde_json::to_string(&source).unwrap();
        let parsed: DocumentBytes = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, source);
    }

    #[test]
    fn apply_doc_id_replaces_single_ids_and_keeps_multi_document_suffixes() {
        let mut single = "f47ac10b-58cc-4372-a567-0e02b2c3d479".to_owned();
        apply_doc_id("/tmp/notes.txt", "doc-1", &mut single);
        assert_eq!(single, "doc-1");

        let mut row = "people.csv_s0_r2".to_owned();
        apply_doc_id("people.csv_s0_r2", "doc-2", &mut row);
        assert_eq!(row, "doc-2");

        let mut xlsx_row = "people_s0_r2".to_owned();
        apply_doc_id("people.xlsx", "doc-3", &mut xlsx_row);
        assert_eq!(xlsx_row, "doc-3_s0_r2");

        let mut xlsx_column = "people_s1_c4".to_owned();
        apply_doc_id("/tmp/people.xlsx", "doc-4", &mut xlsx_column);
        assert_eq!(xlsx_column, "doc-4_s1_c4");
    }

    #[test]
    fn apply_load_options_stamps_the_final_id_and_the_title() {
        let mut documents = vec![Document::new("body"), Document::new("other")];
        apply_load_options(
            "notes.txt",
            &mut documents,
            &LoadOptions::new().with_doc_id("doc-7").with_title("Custom"),
        );
        assert_eq!(documents[0].id, "doc-7");
        assert_eq!(documents[1].id, "doc-7");
        assert_eq!(
            documents[0].metadata.get("doc_id"),
            Some(&Value::String("doc-7".to_owned()))
        );
        assert_eq!(
            documents[1].metadata.get("title"),
            Some(&Value::String("Custom".to_owned()))
        );
    }

    #[test]
    fn apply_load_options_keeps_loader_ids_when_no_doc_id_is_given() {
        let mut documents = vec![Document::new("body")];
        documents[0].id = "generated".to_owned();
        apply_load_options("notes.txt", &mut documents, &LoadOptions::new());
        assert_eq!(documents[0].id, "generated");
        assert_eq!(
            documents[0].metadata.get("doc_id"),
            Some(&Value::String("generated".to_owned()))
        );
        assert!(documents[0].metadata.get("title").is_none());
    }

    #[test]
    fn apply_load_options_generates_an_id_for_an_empty_one() {
        let mut documents = vec![Document::new("body")];
        documents[0].id = String::new();
        apply_load_options("notes.txt", &mut documents, &LoadOptions::new());
        assert_eq!(documents[0].id.len(), 36);
        assert!(Uuid::parse_str(&documents[0].id).is_ok());
    }

    /// A loader that only implements the required methods.
    struct PathOnlyLoader;

    #[async_trait]
    impl DocumentLoader for PathOnlyLoader {
        fn supports(&self, _source: &str) -> bool {
            false
        }

        async fn load(&self, _source: &str) -> Result<Vec<Document>> {
            Ok(Vec::new())
        }
    }

    #[tokio::test]
    async fn the_default_byte_path_is_a_typed_unsupported_error() {
        let source = DocumentBytes::new("data.pdf", b"bytes".to_vec());
        let error = PathOnlyLoader
            .load_bytes(&source, &LoadOptions::new())
            .await
            .unwrap_err();
        assert_eq!(
            error.kind(),
            openjiuwen_common::error::ErrorKind::Unsupported
        );
        assert!(
            error.to_string().contains("PathOnlyLoader"),
            "unexpected message: {error}"
        );
        assert!(
            error
                .to_string()
                .contains("does not support in-memory bytes"),
            "unexpected message: {error}"
        );
    }
}
