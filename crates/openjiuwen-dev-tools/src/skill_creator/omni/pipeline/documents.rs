//! Binary document extraction for the scrape stage.
//!
//! The scrape stage handles HTML, archives, and plain text itself; a resource
//! that names a document the kernel's retrieval loaders understand (`PDF`,
//! `DOCX`, `XLSX`, `PPTX`, `ODT`, plus the text, Markdown, `CSV`/`TSV`, and
//! `JSON` formats) is extracted here with
//! [`AutoLoader`](openjiuwen_core::retrieval::loaders::AutoLoader) instead, so a
//! `skill_omni_creation` run can derive a skill from a fetched document rather
//! than from a UTF-8 approximation of its container bytes.
//!
//! Dispatch uses the last path segment of the fetched URL first (query and
//! fragment ignored) and the response `content-type` second; a resource that
//! names neither keeps the plain-text route of
//! [`OmniCreation::scrape`](super::OmniCreation::scrape). A loader failure is
//! never fatal: the body is decoded as text instead, and only a document that
//! yields nothing that way is a validation error, naming the URL and the format.
//! Office documents are `ZIP` containers, which is why the scrape stage offers a
//! document resource to this module before the generic archive reader sees its
//! package parts.
//!
//! Everything here is offline and deterministic: the loaders parse in-memory
//! bytes, and no file, network, environment, or wall clock is read.

use openjiuwen_common::error::{Error, Result};
use openjiuwen_core::retrieval::common::Document;
use openjiuwen_core::retrieval::loaders::{AutoLoader, LoadOptions};

use super::page::{collect_fenced_blocks, collect_markdown_headings};
use super::ScrapedPage;
use crate::skill_creator::omni::fetch::FetchedResource;

/// Extensions whose document is dispatched to a retrieval loader.
///
/// `HTML` is absent because [`OmniCreation::scrape`](super::OmniCreation::scrape)
/// gives it its own route, and images are absent because a fetched image is an
/// asset to download, never the page the skill is generated from.
const DOCUMENT_EXTENSIONS: [&str; 11] = [
    ".pdf",
    ".docx",
    ".xlsx",
    ".pptx",
    ".odt",
    ".txt",
    ".md",
    ".markdown",
    ".csv",
    ".tsv",
    ".json",
];

/// Fallback stem for a resource whose URL names no last path segment.
const DEFAULT_STEM: &str = "document";

/// The `content-type` to extension table used when the URL names no extension.
///
/// The first matching prefix wins, and the comparison ignores case and any
/// `; charset=…` parameters.
const CONTENT_TYPE_EXTENSIONS: [(&str, &str); 10] = [
    ("application/pdf", ".pdf"),
    (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".docx",
    ),
    (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xlsx",
    ),
    (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".pptx",
    ),
    ("application/vnd.oasis.opendocument.text", ".odt"),
    ("text/markdown", ".md"),
    ("text/x-markdown", ".md"),
    ("text/csv", ".csv"),
    ("text/tab-separated-values", ".tsv"),
    ("application/json", ".json"),
];

/// The loader-facing identity of one fetched document.
///
/// `name` is what the loader sees (its extension selects the reader, its stem
/// becomes the provenance title); `extension` is the lowercase dispatching
/// suffix used in messages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DocumentSource {
    /// File name handed to the loader, for example `report.pdf`.
    pub(super) name: String,
    /// Lowercase dispatching extension, including the dot.
    pub(super) extension: String,
}

/// Resolves the loader identity of a fetched resource, if it is a document.
///
/// The last path segment of `url` decides first (an unsupported or absent
/// suffix falls through), then the `content-type` mapping. Returns `None` for a
/// resource that names no supported document, which keeps the caller's
/// plain-text route.
pub(super) fn document_source(url: &str, content_type: Option<&str>) -> Option<DocumentSource> {
    let segment = last_path_segment(url);
    if let Some(extension) = extension_of(segment).filter(|extension| is_supported(extension)) {
        return Some(DocumentSource {
            name: segment.to_owned(),
            extension,
        });
    }
    let extension = content_type_extension(content_type)?;
    let stem = segment.trim_end_matches('.');
    let name = if stem.is_empty() {
        format!("{DEFAULT_STEM}{extension}")
    } else {
        format!("{stem}{extension}")
    };
    Some(DocumentSource {
        name,
        extension: extension.to_owned(),
    })
}

/// Reports whether the resource is a document dispatched to a loader.
///
/// This is the cheap, synchronous check the scrape stage needs before it decides
/// between a document and a `ZIP` archive (office documents are both).
pub(super) fn is_document(url: &str, content_type: Option<&str>) -> bool {
    document_source(url, content_type).is_some()
}

/// Extracts a [`ScrapedPage`] from a document resource.
///
/// Returns `Ok(None)` when the resource names no supported document. Otherwise
/// the loader's documents are joined into one page: `text` is their text
/// joined with blank lines in loader order, `title` is the first metadata title
/// (the loaders stamp the document's own title or the file stem, Python
/// parity), else the first level-1 ATX heading of the text, else empty, and
/// `headings`/`code_blocks` are the markdown headings and fenced blocks of the
/// text. Document text carries no anchors or images, so `links`/`images` stay
/// empty and `description` stays empty, exactly like the plain-text route.
///
/// A loader failure, or an extraction that yields no text, logs
/// [`tracing::warn`] and falls back to the plain-text route (the body decoded as
/// text). When that is blank too the format is unusable, so the stage reports
/// [`Error::validation`](openjiuwen_common::error::Error::validation) naming the
/// URL, the extension, and the loader's own error rather than returning an empty
/// page.
///
/// # Errors
///
/// Returns a validation error when the document and its plain-text fallback both
/// carry no text.
pub(super) async fn document_page(
    resource: &FetchedResource,
    url: &str,
) -> Result<Option<ScrapedPage>> {
    let Some(source) = document_source(url, resource.content_type.as_deref()) else {
        return Ok(None);
    };
    match extract(&source, &resource.body).await {
        Ok(documents) => {
            let page = page_from_documents(url, &documents);
            if !page.is_blank() {
                return Ok(Some(page));
            }
            tracing::warn!(
                url,
                extension = %source.extension,
                "the document loader extracted no text; falling back to plain text"
            );
            plain_or_error(resource, url, &source, None)
        }
        Err(error) => {
            tracing::warn!(
                url,
                extension = %source.extension,
                %error,
                "the document loader failed; falling back to plain text"
            );
            plain_or_error(resource, url, &source, Some(&error))
        }
    }
}

/// Runs the auto loader on the in-memory body of one document.
async fn extract(source: &DocumentSource, body: &[u8]) -> Result<Vec<Document>> {
    AutoLoader::new()
        .load_bytes(&source.name, body.to_vec(), &LoadOptions::new())
        .await
}

/// Builds the page of a successfully loaded document set.
fn page_from_documents(url: &str, documents: &[Document]) -> ScrapedPage {
    let texts: Vec<&str> = documents
        .iter()
        .map(|document| document.text.as_str())
        .filter(|text| !text.trim().is_empty())
        .collect();
    let text = texts.join("\n\n");
    let mut headings = Vec::new();
    collect_markdown_headings(&text, &mut headings);
    let mut code_blocks = Vec::new();
    collect_fenced_blocks(&text, &mut code_blocks);
    let title = document_title(documents)
        .or_else(|| {
            headings
                .iter()
                .find(|heading| heading.level == 1)
                .map(|heading| heading.text.clone())
        })
        .unwrap_or_default();
    ScrapedPage {
        url: url.to_owned(),
        title,
        text,
        headings,
        code_blocks,
        ..ScrapedPage::default()
    }
}

/// Returns the first non-empty `title` the loaders stamped on `documents`.
fn document_title(documents: &[Document]) -> Option<String> {
    documents.iter().find_map(|document| {
        document
            .metadata
            .get("title")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|title| !title.is_empty())
            .map(str::to_owned)
    })
}

/// Decodes the body as plain text, or reports the unusable document format.
fn plain_or_error(
    resource: &FetchedResource,
    url: &str,
    source: &DocumentSource,
    loader_error: Option<&Error>,
) -> Result<Option<ScrapedPage>> {
    let page = ScrapedPage {
        url: url.to_owned(),
        text: resource.text(),
        ..ScrapedPage::default()
    };
    if !page.is_blank() {
        return Ok(Some(page));
    }
    let format = source.extension.as_str();
    let error = match loader_error {
        Some(error) => Error::validation(format!(
            "scrape of {url} extracted no text from a {format} document \
             (the loader reported: {error})"
        )),
        None => Error::validation(format!(
            "scrape of {url} extracted no text from a {format} document"
        )),
    };
    Err(error)
}

/// Returns the last non-empty path segment of `url`, ignoring query and fragment.
fn last_path_segment(url: &str) -> &str {
    let without_scheme = url.split_once("://").map_or(url, |(_scheme, rest)| rest);
    let path = without_scheme
        .split_once('/')
        .map_or("", |(_authority, path)| path);
    let path = path.split(['?', '#']).next().unwrap_or_default();
    path.rsplit('/')
        .find(|segment| !segment.is_empty())
        .unwrap_or_default()
}

/// Returns the lowercase final suffix of `name`, including the dot.
///
/// Mirrors `os.path.splitext`: only the last suffix counts, and a dot-file
/// (`.env`) has no extension.
fn extension_of(name: &str) -> Option<String> {
    let index = name.rfind('.')?;
    (index > 0).then(|| name[index..].to_ascii_lowercase())
}

/// Reports whether `extension` is one this module dispatches.
fn is_supported(extension: &str) -> bool {
    DOCUMENT_EXTENSIONS.contains(&extension)
}

/// Maps a response `content-type` to a dispatching extension.
fn content_type_extension(content_type: Option<&str>) -> Option<&'static str> {
    let mime = content_type?
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    CONTENT_TYPE_EXTENSIONS
        .iter()
        .find(|(prefix, _extension)| mime.as_str() == *prefix)
        .map(|(_prefix, extension)| *extension)
}

#[cfg(test)]
mod tests;
