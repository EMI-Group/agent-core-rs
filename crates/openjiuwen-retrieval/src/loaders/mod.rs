//! Document loaders for text-like sources.
//!
//! Port of the offline subset of the Python
//! `openjiuwen/core/retrieval/indexing/processor/parser/` package: plain text,
//! Markdown, `JSON`, `CSV`/`TSV`, local `HTML`, `HTTP(S)` URLs, and the
//! auto-dispatch behavior of `AutoFileParser` / `AutoLinkParser` / `AutoParser`.
//!
//! `PDF`, `DOCX`, `XLSX`, images, `WeChat` articles, and article-extraction
//! services are out of scope: [`AutoLoader`] rejects them with
//! [`openjiuwen_common::error::Error::Unsupported`].

pub mod auto;
pub mod csv;
pub mod html;
pub mod json;
pub mod text;
pub mod url;

pub use auto::{AutoLoader, LoaderRegistry};
pub use csv::CsvLoader;
pub use html::{extract_title, html_to_text, HtmlLoader};
pub use json::JsonLoader;
pub use text::{MarkdownLoader, TextLoader};
pub use url::UrlLoader;

use async_trait::async_trait;
use regex::Regex;
use uuid::Uuid;

use openjiuwen_common::error::{Error, Result};

use crate::common::Document;

/// A source of [`Document`]s addressed by a path or an `HTTP(S)` URL.
///
/// Mirrors the Python `Parser` ABC (`parse`/`supports`) without the streaming
/// `lazy_parse` variant, which has no caller in the Rust port.
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
}
