//! `HTML` to text conversion, title extraction, and the local `HTML` loader.
//!
//! Port of the offline part of the Python `HTMLFileParser`
//! (`html_file_parser.py`): scripts, styles, comments, and tags are stripped,
//! entities are decoded, and the title comes from `og:title` before `<title>`.
//! Python's readability-style main-content selection and its 50-character
//! minimum are not ported; the Rust loader only requires non-empty text.

use std::sync::LazyLock;

use async_trait::async_trait;
use regex::Regex;
use serde_json::{Map, Value};

use openjiuwen_common::error::{Error, Result};

use crate::common::Document;

use super::{
    apply_load_options, compile_pattern, file_extension, read_source, DocumentBytes,
    DocumentLoader, LoadOptions,
};

/// Matches a `<script>` element, including an unterminated one.
static SCRIPT_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| compile_pattern(r"(?is)<script\b[^>]*>.*?(?:</script\s*>|$)"));
/// Matches a `<style>` element, including an unterminated one.
static STYLE_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| compile_pattern(r"(?is)<style\b[^>]*>.*?(?:</style\s*>|$)"));
/// Matches an `HTML` comment.
static COMMENT_PATTERN: LazyLock<Regex> = LazyLock::new(|| compile_pattern(r"(?s)<!--.*?-->"));
/// Matches any remaining tag.
static TAG_PATTERN: LazyLock<Regex> = LazyLock::new(|| compile_pattern(r"(?s)<[^>]*>"));
/// Matches one `<meta>` element.
static META_PATTERN: LazyLock<Regex> = LazyLock::new(|| compile_pattern(r"(?is)<meta\b[^>]*>"));
/// Matches the `<title>` element and captures its inner `HTML`.
static TITLE_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| compile_pattern(r"(?is)<title\b[^>]*>(.*?)(?:</title\s*>|$)"));
/// Captures the `property` attribute value of a tag.
static PROPERTY_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    compile_pattern(r#"(?is)(?:^|\s)property\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+))"#)
});
/// Captures the `name` attribute value of a tag.
static NAME_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    compile_pattern(r#"(?is)(?:^|\s)name\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+))"#)
});
/// Captures the `content` attribute value of a tag.
static CONTENT_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    compile_pattern(r#"(?is)(?:^|\s)content\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+))"#)
});

/// Converts an `HTML` fragment into whitespace-collapsed plain text.
///
/// Script and style bodies, comments, and tags are removed (tags become a space
/// so adjacent blocks do not run together), then named and numeric entities are
/// decoded and whitespace runs are collapsed to single spaces.
#[must_use]
pub fn html_to_text(html: &str) -> String {
    let without_scripts = SCRIPT_PATTERN.replace_all(html, " ");
    let without_styles = STYLE_PATTERN.replace_all(&without_scripts, " ");
    let without_comments = COMMENT_PATTERN.replace_all(&without_styles, " ");
    let spaced = TAG_PATTERN.replace_all(&without_comments, " ");
    collapse_whitespace(&decode_entities(&spaced))
}

/// Returns the page title, preferring `og:title` over `<title>`.
///
/// The `og:title` value is read from `<meta property="og:title">` or
/// `<meta name="og:title">` (either quote style). Only when neither element
/// exists does this return `None`; an element with an empty value yields
/// `Some(String::new())`.
#[must_use]
pub fn extract_title(html: &str) -> Option<String> {
    let open_graph = find_open_graph_title(html);
    let title = TITLE_PATTERN
        .captures(html)
        .map(|captures| html_to_text(&captures[1]));
    match (open_graph, title) {
        (Some(open_graph), _) if !open_graph.is_empty() => Some(open_graph),
        // Python falls back to <title> when og:title is empty.
        (Some(_), Some(title)) => Some(title),
        (Some(_), None) => Some(String::new()),
        (None, title) => title,
    }
}

/// Finds the `og:title` meta element, returning its text (possibly empty).
fn find_open_graph_title(html: &str) -> Option<String> {
    META_PATTERN.find_iter(html).find_map(|tag| {
        let tag = tag.as_str();
        let is_open_graph = PROPERTY_PATTERN
            .captures(tag)
            .or_else(|| NAME_PATTERN.captures(tag))
            .and_then(|captures| capture_value(&captures))
            .is_some_and(|value| value.eq_ignore_ascii_case("og:title"));
        if !is_open_graph {
            return None;
        }
        Some(
            CONTENT_PATTERN
                .captures(tag)
                .and_then(|captures| capture_value(&captures))
                .map_or_else(String::new, |content| html_to_text(&content)),
        )
    })
}

/// Returns the first non-empty capture group of an attribute pattern.
fn capture_value(captures: &regex::Captures<'_>) -> Option<String> {
    captures
        .get(1)
        .or_else(|| captures.get(2))
        .or_else(|| captures.get(3))
        .map(|value| value.as_str().to_owned())
}

/// Collapses every whitespace run to one space and trims the result.
fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Decodes named and numeric `HTML` entities in one pass.
fn decode_entities(text: &str) -> String {
    let mut decoded = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(index) = rest.find('&') {
        decoded.push_str(&rest[..index]);
        let candidate = &rest[index..];
        if let Some((value, consumed)) = decode_entity(candidate) {
            decoded.push_str(&value);
            rest = &candidate[consumed..];
        } else {
            decoded.push('&');
            rest = &candidate[1..];
        }
    }
    decoded.push_str(rest);
    decoded
}

/// Decodes the entity at the start of `text` (which begins with `&`).
///
/// Returns the replacement text and the number of bytes consumed, or `None` for
/// an unknown or malformed entity (the `&` is then kept literally).
fn decode_entity(text: &str) -> Option<(String, usize)> {
    /// Longest entity body accepted, in bytes (`&frac12;` is 6).
    const MAX_ENTITY_LEN: usize = 10;
    let end = text.find(';')?;
    let body = &text[1..end];
    if body.is_empty() || body.len() > MAX_ENTITY_LEN || body.chars().any(char::is_whitespace) {
        return None;
    }
    let decoded = if let Some(hex) = body
        .strip_prefix("#x")
        .or_else(|| body.strip_prefix("#X"))
        .filter(|digits| !digits.is_empty())
    {
        char::from_u32(u32::from_str_radix(hex, 16).ok()?)?.to_string()
    } else if let Some(decimal) = body.strip_prefix('#').filter(|digits| !digits.is_empty()) {
        char::from_u32(decimal.parse().ok()?)?.to_string()
    } else {
        named_entity(body)?.to_owned()
    };
    Some((decoded, end + 1))
}

/// Looks up one named `HTML` entity.
fn named_entity(name: &str) -> Option<&'static str> {
    let value = match name {
        "amp" => "&",
        "lt" => "<",
        "gt" => ">",
        "quot" => "\"",
        "apos" => "'",
        "nbsp" => " ",
        "mdash" => "\u{2014}",
        "ndash" => "\u{2013}",
        "hellip" => "\u{2026}",
        "copy" => "\u{00a9}",
        "reg" => "\u{00ae}",
        "trade" => "\u{2122}",
        "laquo" => "\u{00ab}",
        "raquo" => "\u{00bb}",
        "ldquo" => "\u{201c}",
        "rdquo" => "\u{201d}",
        "lsquo" => "\u{2018}",
        "rsquo" => "\u{2019}",
        "middot" => "\u{00b7}",
        "bull" => "\u{2022}",
        "deg" => "\u{00b0}",
        "euro" => "\u{20ac}",
        "pound" => "\u{00a3}",
        "yen" => "\u{00a5}",
        "cent" => "\u{00a2}",
        "sect" => "\u{00a7}",
        "para" => "\u{00b6}",
        "times" => "\u{00d7}",
        "divide" => "\u{00f7}",
        "plusmn" => "\u{00b1}",
        "frac12" => "\u{00bd}",
        "frac14" => "\u{00bc}",
        "frac34" => "\u{00be}",
        "sup2" => "\u{00b2}",
        "sup3" => "\u{00b3}",
        "micro" => "\u{00b5}",
        _ => return None,
    };
    Some(value)
}

/// Loads `.html`/`.htm` files as one plain-text document.
#[derive(Debug, Default, Clone, Copy)]
pub struct HtmlLoader;

impl HtmlLoader {
    /// Creates an `HTML` loader.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DocumentLoader for HtmlLoader {
    fn supports(&self, source: &str) -> bool {
        matches!(file_extension(source).as_str(), ".html" | ".htm")
    }

    async fn load(&self, source: &str) -> Result<Vec<Document>> {
        let html = read_source(source).await?;
        Ok(vec![html_document(source, &html)?])
    }

    async fn load_bytes(
        &self,
        source: &DocumentBytes,
        options: &LoadOptions,
    ) -> Result<Vec<Document>> {
        let html = String::from_utf8_lossy(&source.bytes);
        let mut documents = vec![html_document(&source.name, &html)?];
        apply_load_options(&source.name, &mut documents, options);
        Ok(documents)
    }
}

/// Converts one `HTML` string into its web-page document.
///
/// `name` is the path or in-memory source name and appears in errors.
///
/// # Errors
///
/// Returns [`Error::Validation`] when the page holds no extractable text.
fn html_document(name: &str, html: &str) -> Result<Document> {
    let text = html_to_text(html);
    if text.is_empty() {
        return Err(Error::Validation(format!(
            "No text content found in HTML: {name}"
        )));
    }
    let mut metadata = Map::new();
    metadata.insert(
        "title".to_owned(),
        Value::String(extract_title(html).unwrap_or_default()),
    );
    metadata.insert(
        "source_type".to_owned(),
        Value::String("web_page".to_owned()),
    );
    Ok(Document {
        id: super::generate_document_id(),
        text,
        metadata,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use openjiuwen_common::error::ErrorKind;

    #[test]
    fn html_to_text_strips_scripts_styles_comments_and_tags() {
        let html = "<html><head><title>T</title><style>body { color: red; }</style>\
                    <script type=\"text/javascript\">var x = '<p>no</p>';</script></head>\
                    <body><!-- hidden --><h1>Hello</h1><p>world  &amp;  friends</p></body></html>";
        assert_eq!(html_to_text(html), "T Hello world & friends");
    }

    #[test]
    fn html_to_text_decodes_named_and_numeric_entities() {
        assert_eq!(
            html_to_text("<p>a&nbsp;b &mdash; c&hellip; &copy; &#65;&#x42; &unknown; &amp</p>"),
            "a b \u{2014} c\u{2026} \u{00a9} AB &unknown; &amp",
        );
        assert_eq!(html_to_text("&lt;tag&gt; &quot;q&quot;"), "<tag> \"q\"");
    }

    #[test]
    fn html_to_text_handles_unterminated_script_and_empty_input() {
        assert_eq!(html_to_text("<div>x</div><script>never closed"), "x");
        assert_eq!(html_to_text(""), "");
        assert_eq!(html_to_text("<!-- only a comment -->"), "");
    }

    #[test]
    fn extract_title_prefers_open_graph_metadata() {
        let html = "<meta property=\"og:title\" content=\"OG &amp; Title\"><title>Plain</title>";
        assert_eq!(extract_title(html), Some("OG & Title".to_owned()));

        let named = "<meta name='og:title' content='Single quoted'><title>Plain</title>";
        assert_eq!(extract_title(named), Some("Single quoted".to_owned()));

        let meta_only = "<title>Plain  Title</title>";
        assert_eq!(extract_title(meta_only), Some("Plain Title".to_owned()));

        // An empty og:title falls back to <title>, Python parity.
        let empty_og = "<meta property=\"og:title\" content=\"\"><title>Fallback</title>";
        assert_eq!(extract_title(empty_og), Some("Fallback".to_owned()));

        assert_eq!(extract_title("<p>no title</p>"), None);
        assert_eq!(extract_title("<title></title>"), Some(String::new()));
    }

    #[tokio::test]
    async fn html_loader_builds_one_document_with_web_page_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("page.html");
        std::fs::write(
            &path,
            "<html><head><title>Docs</title></head><body><p>Hello <b>there</b></p></body></html>",
        )
        .unwrap();

        let documents = HtmlLoader::new()
            .load(path.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(documents.len(), 1);
        assert_eq!(documents[0].text, "Docs Hello there");
        assert_eq!(
            documents[0].metadata.get("title"),
            Some(&Value::String("Docs".to_owned()))
        );
        assert_eq!(
            documents[0].metadata.get("source_type"),
            Some(&Value::String("web_page".to_owned()))
        );
    }

    #[tokio::test]
    async fn html_loader_rejects_pages_without_text() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.htm");
        std::fs::write(&path, "<html><script>x</script></html>").unwrap();

        let error = HtmlLoader::new()
            .load(path.to_str().unwrap())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("No text content found in HTML"));
    }

    #[tokio::test]
    async fn load_bytes_matches_the_path_result() {
        let source = DocumentBytes::new(
            "page.html",
            b"<html><head><title>Docs</title></head><body><p>Hello <b>there</b></p></body></html>"
                .to_vec(),
        );
        let documents = HtmlLoader::new()
            .load_bytes(&source, &LoadOptions::new())
            .await
            .unwrap();
        assert_eq!(documents.len(), 1);
        assert_eq!(documents[0].text, "Docs Hello there");
        assert_eq!(documents[0].id.len(), 36);
        assert_eq!(
            documents[0].metadata.get("title"),
            Some(&Value::String("Docs".to_owned()))
        );
        assert_eq!(
            documents[0].metadata.get("source_type"),
            Some(&Value::String("web_page".to_owned()))
        );
        assert_eq!(
            documents[0].metadata.get("doc_id").and_then(Value::as_str),
            Some(documents[0].id.as_str())
        );
    }

    #[tokio::test]
    async fn load_bytes_applies_doc_id_and_title() {
        let source = DocumentBytes::new(
            "page.html",
            b"<html><head><title>Docs</title></head><body>body</body></html>".to_vec(),
        );
        let options = LoadOptions::new().with_doc_id("doc-3").with_title("Custom");
        let documents = HtmlLoader::new()
            .load_bytes(&source, &options)
            .await
            .unwrap();
        assert_eq!(documents[0].id, "doc-3");
        assert_eq!(
            documents[0].metadata.get("title"),
            Some(&Value::String("Custom".to_owned()))
        );
        assert_eq!(
            documents[0].metadata.get("doc_id"),
            Some(&Value::String("doc-3".to_owned()))
        );
    }

    #[tokio::test]
    async fn load_bytes_rejects_pages_without_text() {
        let source = DocumentBytes::new("empty.htm", b"<html><script>x</script></html>".to_vec());
        let error = HtmlLoader::new()
            .load_bytes(&source, &LoadOptions::new())
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(
            error
                .to_string()
                .contains("No text content found in HTML: empty.htm"),
            "unexpected message: {error}"
        );
    }

    #[test]
    fn loader_supports_html_extensions_only() {
        let loader = HtmlLoader::new();
        assert!(loader.supports("page.html"));
        assert!(loader.supports("page.HTM"));
        assert!(!loader.supports("page.txt"));
    }
}
