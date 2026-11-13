//! In-crate HTML-to-text extraction and URL normalisation for the web tools.
//!
//! Ports the text-extraction half of `openjiuwen/harness/tools/web/fetch_webpage.py`
//! (`_extract_main_text_from_html`) without `BeautifulSoup`: a small streaming tokenizer drops
//! markup, decodes HTML entities and keeps block structure as newlines. No external HTML
//! parser is linked, so the crate stays offline-testable and dependency-light.
//!
//! [`normalize_url`] is the in-crate replacement for `urllib.parse.urljoin`/`urlsplit`
//! (this crate does not depend on the `url` crate): it resolves relative references, strips
//! fragments, lowercases the scheme and host and rejects non-http(s) targets.

use std::borrow::Cow;

/// The title and main text extracted from an HTML document.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExtractedHtml {
    /// The trimmed contents of the first `<title>` element, when the document has one.
    pub title: Option<String>,
    /// The document text with markup and metadata noise removed, whitespace collapsed.
    pub text: String,
}

/// Elements whose contents are never rendered as text.
///
/// Their bodies are skipped up to the matching end tag, so script source and stylesheet
/// rules never leak into the extracted text.
const RAW_TEXT_ELEMENTS: &[&str] = &[
    "script", "style", "noscript", "template", "svg", "canvas", "iframe",
];

/// Elements that start and end a visual block, and therefore emit a line break.
const BLOCK_ELEMENTS: &[&str] = &[
    "address",
    "article",
    "aside",
    "blockquote",
    "body",
    "caption",
    "dd",
    "details",
    "dialog",
    "div",
    "dl",
    "dt",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "html",
    "li",
    "main",
    "nav",
    "ol",
    "option",
    "p",
    "section",
    "summary",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "tr",
    "ul",
];

/// Extracts the title and the main text of an HTML document.
///
/// `<script>`, `<style>`, `<noscript>`, `<template>`, `<svg>`, `<canvas>` and `<iframe>`
/// bodies are dropped whole; comments, CDATA sections, doctypes and processing instructions
/// are dropped as markup; `<title>` is captured separately; block-level elements emit a line
/// break (at most one blank line); runs of horizontal whitespace collapse to a single space
/// and the result is trimmed. `<pre>` content keeps its own newlines and indentation.
///
/// The extractor never panics: malformed input (an unclosed tag, a stray `<`, an unterminated
/// script) degrades to literal text instead of failing.
#[must_use]
pub fn extract(html: &str) -> ExtractedHtml {
    let normalized = normalize_newlines(html);
    let mut state = ExtractState::default();
    state.run(&normalized);
    ExtractedHtml {
        title: state
            .title
            .map(|title| title.trim().to_owned())
            .filter(|title| !title.is_empty()),
        text: state.out.trim().to_owned(),
    }
}

/// Resolves `href` against `base` and normalizes the result.
///
/// A relative or root-relative `href` is resolved against `base`; a protocol-relative
/// `//host/path` inherits the base scheme. The returned URL has a lowercase scheme and host,
/// no fragment, no dot segments and a non-empty path. Returns `None` when the reference is
/// empty, fragment-only, malformed or uses a scheme other than `http`/`https`.
///
/// Deviation: user info is preserved verbatim and percent-encoded dot segments (`%2e%2e`)
/// are not decoded, so they are treated as ordinary path segments.
#[must_use]
pub fn normalize_url(base: &str, href: &str) -> Option<String> {
    let href = href.trim();
    if href.is_empty() || href.starts_with('#') {
        return None;
    }
    let href = strip_fragment(href);
    if href.is_empty() {
        return None;
    }
    if url_scheme(href).is_some() {
        return normalize_absolute(href);
    }
    let base = strip_fragment(base.trim());
    let base = split_absolute(base)?;
    if href.starts_with("//") {
        return normalize_absolute(&format!("{}:{}", base.scheme, href));
    }
    let (path, query) = split_query(href);
    let resolved_path = if path.is_empty() {
        // A query-only reference keeps the base path.
        query?;
        base.path.to_owned()
    } else if path.starts_with('/') {
        path.to_owned()
    } else {
        let directory = match base.path.rsplit_once('/') {
            Some((directory, _)) => directory,
            None => "",
        };
        format!("{directory}/{path}")
    };
    let mut url = format!(
        "{}://{}{}",
        base.scheme.to_ascii_lowercase(),
        normalize_authority(base.authority)?,
        normalize_path(&resolved_path)
    );
    if let Some(query) = query {
        if !query.is_empty() {
            url.push('?');
            url.push_str(query);
        }
    }
    Some(url)
}

/// Collapses every whitespace run in `text` to one space and trims the ends.
#[must_use]
pub fn collapse_whitespace(text: &str) -> String {
    let mut collapsed = String::with_capacity(text.len());
    let mut pending_space = false;
    for character in text.chars() {
        if character.is_whitespace() {
            pending_space = !collapsed.is_empty();
        } else {
            if pending_space {
                collapsed.push(' ');
                pending_space = false;
            }
            collapsed.push(character);
        }
    }
    collapsed
}

/// Decodes the named and numeric HTML entities of `text`.
///
/// Unknown or malformed entities are kept verbatim so no text is lost.
#[must_use]
pub fn decode_entities(text: &str) -> String {
    if !text.contains('&') {
        return text.to_owned();
    }
    let mut decoded = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(position) = rest.find('&') {
        decoded.push_str(&rest[..position]);
        let candidate = &rest[position..];
        if let Some((entity, consumed)) = split_entity(candidate) {
            decoded.push(entity);
            rest = &candidate[consumed..];
        } else {
            decoded.push('&');
            rest = &candidate[1..];
        }
    }
    decoded.push_str(rest);
    decoded
}

/// Splits one `&...;` entity off `candidate`, returning the character and the bytes consumed.
fn split_entity(candidate: &str) -> Option<(char, usize)> {
    let mut end = None;
    for (offset, character) in candidate.char_indices().take(34) {
        if character == ';' {
            end = Some(offset);
            break;
        }
    }
    let end = end?;
    if end <= 1 {
        return None;
    }
    let name = &candidate[1..end];
    let decoded = if let Some(number) = name.strip_prefix('#') {
        decode_numeric_entity(number)?
    } else {
        named_entity(name)?
    };
    Some((decoded, end + 1))
}

/// Decodes `#123` / `#x1f600` entity bodies.
fn decode_numeric_entity(number: &str) -> Option<char> {
    let code = match number
        .strip_prefix('x')
        .or_else(|| number.strip_prefix('X'))
    {
        Some(hex) => u32::from_str_radix(hex, 16).ok()?,
        None => number.parse::<u32>().ok()?,
    };
    if code == 0 {
        return None;
    }
    char::from_u32(code)
}

/// Decodes a named entity body (ASCII case-insensitive).
fn named_entity(name: &str) -> Option<char> {
    let lower = name.to_ascii_lowercase();
    let character = match lower.as_str() {
        "amp" => '&',
        "lt" => '<',
        "gt" => '>',
        "quot" => '"',
        "apos" => '\'',
        "nbsp" | "ensp" | "emsp" | "thinsp" => ' ',
        "shy" => '\u{00ad}',
        "copy" => '\u{00a9}',
        "reg" => '\u{00ae}',
        "trade" => '\u{2122}',
        "deg" => '\u{00b0}',
        "plusmn" => '\u{00b1}',
        "times" => '\u{00d7}',
        "divide" => '\u{00f7}',
        "frac12" => '\u{00bd}',
        "frac14" => '\u{00bc}',
        "frac34" => '\u{00be}',
        "sup2" => '\u{00b2}',
        "sup3" => '\u{00b3}',
        "micro" => '\u{00b5}',
        "middot" => '\u{00b7}',
        "bull" => '\u{2022}',
        "ndash" => '\u{2013}',
        "mdash" => '\u{2014}',
        "hellip" => '\u{2026}',
        "lsquo" => '\u{2018}',
        "rsquo" => '\u{2019}',
        "ldquo" => '\u{201c}',
        "rdquo" => '\u{201d}',
        "laquo" => '\u{00ab}',
        "raquo" => '\u{00bb}',
        "lsaquo" => '\u{2039}',
        "rsaquo" => '\u{203a}',
        "dagger" => '\u{2020}',
        "permil" => '\u{2030}',
        "euro" => '\u{20ac}',
        "pound" => '\u{00a3}',
        "yen" => '\u{00a5}',
        "cent" => '\u{00a2}',
        "sect" => '\u{00a7}',
        "para" => '\u{00b6}',
        "larr" => '\u{2190}',
        "rarr" => '\u{2192}',
        "uarr" => '\u{2191}',
        "darr" => '\u{2193}',
        "harr" => '\u{2194}',
        "le" => '\u{2264}',
        "ge" => '\u{2265}',
        "ne" => '\u{2260}',
        "asymp" => '\u{2248}',
        "infin" => '\u{221e}',
        "minus" => '\u{2212}',
        _ => return None,
    };
    Some(character)
}

/// Replaces `\r\n` and lone `\r` with `\n`.
fn normalize_newlines(html: &str) -> Cow<'_, str> {
    if html.contains('\r') {
        Cow::Owned(html.replace("\r\n", "\n").replace('\r', "\n"))
    } else {
        Cow::Borrowed(html)
    }
}

/// Incremental extractor state.
#[derive(Debug, Default)]
struct ExtractState {
    /// The first captured title.
    title: Option<String>,
    /// Text accumulated while `<title>` is open.
    title_buf: String,
    /// Rendered text.
    out: String,
    /// Pending newlines (capped at two) before the next text run.
    pending_breaks: usize,
    /// Whether a space is owed before the next text run.
    pending_space: bool,
    /// Nesting depth of `<pre>`; the body keeps its own whitespace while non-zero.
    pre_depth: usize,
    /// Whether text currently belongs to `<title>` rather than the body.
    in_title: bool,
}

impl ExtractState {
    /// Consumes the whole document.
    fn run(&mut self, html: &str) {
        let mut rest = html;
        while let Some(position) = rest.find('<') {
            if position > 0 {
                self.push_text(&rest[..position]);
                rest = &rest[position..];
            }
            if let Some(consumed) = skip_markup(rest) {
                rest = &rest[consumed..];
                continue;
            }
            // A stray '<' is literal text, and a tag may swallow a raw-text body.
            match self.start_markup(rest) {
                Ok(consumed) => rest = &rest[consumed..],
                Err(raw_element) => match find_raw_end(rest, raw_element) {
                    Some(consumed) => rest = &rest[consumed..],
                    None => return,
                },
            }
        }
        self.push_text(rest);
    }

    /// Parses one tag. `Ok` is the bytes consumed; `Err` names a raw-text element whose body
    /// must be skipped up to its end tag. Also returns the consumed length of a bare `<`.
    fn start_markup(&mut self, rest: &str) -> Result<usize, &'static str> {
        let Some(end) = find_tag_end(rest) else {
            // An unclosed tag: treat the rest as text.
            self.push_text(rest);
            return Ok(rest.len());
        };
        let tag = &rest[1..end];
        if tag.starts_with(['!', '?']) || tag.is_empty() {
            return Ok(end + 1);
        }
        match self.handle_tag(tag) {
            Some(raw_element) => Err(raw_element),
            None => Ok(end + 1),
        }
    }

    /// Applies one tag to the extraction state, returning the raw-text element name when the
    /// tag opens a body that must be skipped.
    fn handle_tag(&mut self, tag: &str) -> Option<&'static str> {
        let tag = tag.trim();
        let (closing, body) = match tag.strip_prefix('/') {
            Some(body) => (true, body),
            None => (false, tag),
        };
        let name = tag_name(body);
        if name.is_empty() {
            return None;
        }
        let raw_element = if closing {
            None
        } else {
            RAW_TEXT_ELEMENTS
                .iter()
                .copied()
                .find(|candidate| *candidate == name)
        };
        if let Some(raw_element) = raw_element {
            return Some(raw_element);
        }
        if name == "title" {
            self.handle_title(closing);
            return None;
        }
        if name == "pre" {
            if closing {
                self.pre_depth = self.pre_depth.saturating_sub(1);
            } else {
                self.pre_depth += 1;
            }
            self.break_lines(1);
            return None;
        }
        if !closing && (name == "br" || name == "hr") {
            self.break_lines(1);
            return None;
        }
        if BLOCK_ELEMENTS.contains(&name.as_str()) {
            self.break_lines(1);
        }
        None
    }

    /// Opens or closes `<title>`, publishing the first non-empty title.
    fn handle_title(&mut self, closing: bool) {
        if closing {
            if self.in_title {
                self.in_title = false;
                let title = collapse_whitespace(&self.title_buf);
                if !title.is_empty() && self.title.is_none() {
                    self.title = Some(title);
                }
                self.title_buf.clear();
            }
            return;
        }
        if !self.in_title {
            self.in_title = true;
            self.title_buf.clear();
        }
    }

    /// Appends text, decoding entities and normalizing whitespace.
    fn push_text(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        let decoded = decode_entities(text);
        if self.in_title {
            self.title_buf.push_str(&decoded);
            return;
        }
        if self.pre_depth > 0 {
            if decoded.chars().all(char::is_whitespace) {
                if !self.out.is_empty() {
                    self.flush_breaks();
                    self.out.push_str(&decoded);
                }
                return;
            }
            self.flush_breaks();
            self.out.push_str(&decoded);
            return;
        }
        let collapsed = collapse_whitespace(&decoded);
        if collapsed.is_empty() {
            if !self.out.is_empty() && !self.out.ends_with(['\n', ' ']) {
                self.pending_space = true;
            }
            return;
        }
        let leading = decoded.chars().next().is_some_and(char::is_whitespace);
        let trailing = decoded.chars().last().is_some_and(char::is_whitespace);
        if self.pending_breaks > 0 {
            self.flush_breaks();
        } else if (leading || self.pending_space)
            && !self.out.is_empty()
            && !self.out.ends_with(['\n', ' '])
        {
            self.out.push(' ');
        }
        self.pending_space = false;
        self.out.push_str(&collapsed);
        self.pending_space = trailing;
    }

    /// Requests `count` newlines before the next text run.
    fn break_lines(&mut self, count: usize) {
        if self.in_title {
            return;
        }
        self.pending_breaks = self.pending_breaks.max(count);
        self.pending_space = false;
    }

    /// Emits the pending newlines (at most two) and clears the flags.
    fn flush_breaks(&mut self) {
        self.pending_space = false;
        if self.pending_breaks == 0 {
            return;
        }
        while self.out.ends_with(' ') {
            self.out.pop();
        }
        if !self.out.is_empty() {
            let existing = self.out.chars().rev().take_while(|c| *c == '\n').count();
            for _ in existing..self.pending_breaks.min(2) {
                self.out.push('\n');
            }
        }
        self.pending_breaks = 0;
    }
}

/// Skips comments, CDATA sections, doctypes and processing instructions.
fn skip_markup(rest: &str) -> Option<usize> {
    if let Some(body) = rest.strip_prefix("<!--") {
        return Some(match body.find("-->") {
            Some(end) => 4 + end + 3,
            None => rest.len(),
        });
    }
    if let Some(body) = rest.strip_prefix("<![CDATA[") {
        return Some(match body.find("]]>") {
            Some(end) => 9 + end + 3,
            None => rest.len(),
        });
    }
    if rest.starts_with("<!") || rest.starts_with("<?") {
        return Some(match rest.find('>') {
            Some(end) => end + 1,
            None => rest.len(),
        });
    }
    None
}

/// Returns the lowercased element name at the start of `body`.
fn tag_name(body: &str) -> String {
    body.trim_start()
        .chars()
        .take_while(|character| {
            character.is_ascii_alphanumeric() || *character == ':' || *character == '-'
        })
        .map(|character| character.to_ascii_lowercase())
        .collect()
}

/// Returns the byte index of the `>` closing the tag that starts at `tag[0] == '<'`.
///
/// Quoted attribute values may contain `/` and `>`.
fn find_tag_end(tag: &str) -> Option<usize> {
    let mut quote: Option<char> = None;
    for (index, character) in tag.char_indices().skip(1) {
        if let Some(active) = quote {
            if character == active {
                quote = None;
            }
            continue;
        }
        match character {
            '"' | '\'' => quote = Some(character),
            '>' => return Some(index),
            _ => {}
        }
    }
    None
}

/// Returns the byte offset just past the `</name>` end tag, if the input has one.
fn find_raw_end(input: &str, name: &str) -> Option<usize> {
    let bytes = input.as_bytes();
    let mut offset = 0;
    while let Some(position) = input[offset..].find("</") {
        let start = offset + position;
        let after = start + 2;
        let name_end = after + name.len();
        if name_end <= bytes.len() && bytes[after..name_end].eq_ignore_ascii_case(name.as_bytes()) {
            return input[start..].find('>').map(|end| start + end + 1);
        }
        offset = start + 2;
    }
    None
}

/// Returns the URI scheme of `url`, or `None` when the reference is relative or schemeless.
///
/// The scheme must start with an ASCII letter and continue with letters, digits, `+`, `-` or
/// `.`, so a colon inside a path (`a/b:c`) is not mistaken for one.
#[must_use]
pub fn url_scheme(url: &str) -> Option<&str> {
    let colon = url.find(':')?;
    let scheme = &url[..colon];
    let mut characters = scheme.chars();
    if !characters.next()?.is_ascii_alphabetic() {
        return None;
    }
    if !characters
        .all(|character| character.is_ascii_alphanumeric() || matches!(character, '+' | '-' | '.'))
    {
        return None;
    }
    Some(scheme)
}

/// The parts of an absolute `http`/`https` URL.
#[derive(Debug, Clone, Copy)]
struct UrlParts<'a> {
    /// The scheme, as written.
    scheme: &'a str,
    /// The authority (`host`, optionally with user info and port).
    authority: &'a str,
    /// The path, possibly empty.
    path: &'a str,
    /// The query, without the leading `?`.
    query: Option<&'a str>,
}

/// Splits an absolute URL into its parts, rejecting a missing or empty authority.
fn split_absolute(url: &str) -> Option<UrlParts<'_>> {
    let scheme_end = url.find("://")?;
    let scheme = &url[..scheme_end];
    if scheme.is_empty()
        || !scheme.chars().next()?.is_ascii_alphabetic()
        || !scheme.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '+' | '-' | '.')
        })
    {
        return None;
    }
    let rest = &url[scheme_end + 3..];
    let authority_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let authority = &rest[..authority_end];
    if authority.is_empty() {
        return None;
    }
    let rest = strip_fragment(&rest[authority_end..]);
    let (path, query) = split_query(rest);
    Some(UrlParts {
        scheme,
        authority,
        path,
        query,
    })
}

/// Normalizes a fully absolute URL, rejecting non-http(s) schemes.
fn normalize_absolute(url: &str) -> Option<String> {
    let parts = split_absolute(url)?;
    let scheme = parts.scheme.to_ascii_lowercase();
    if scheme != "http" && scheme != "https" {
        return None;
    }
    let authority = normalize_authority(parts.authority)?;
    let mut normalized = format!("{scheme}://{authority}{}", normalize_path(parts.path));
    if let Some(query) = parts.query {
        if !query.is_empty() {
            normalized.push('?');
            normalized.push_str(query);
        }
    }
    Some(normalized)
}

/// Lowercases the host and port of `authority`, keeping any user info verbatim.
fn normalize_authority(authority: &str) -> Option<String> {
    let authority = authority.trim();
    if authority.is_empty() || authority.chars().any(char::is_whitespace) {
        return None;
    }
    let normalized = match authority.rsplit_once('@') {
        Some((user_info, host)) => format!("{user_info}@{}", host.to_ascii_lowercase()),
        None => authority.to_ascii_lowercase(),
    };
    let host = normalized
        .rsplit_once('@')
        .map_or(normalized.as_str(), |(_, host)| host);
    if host.is_empty() {
        return None;
    }
    Some(normalized)
}

/// Normalizes a URL path: resolves `.`/`..`, collapses duplicate slashes, keeps a trailing
/// slash and always returns a path starting with `/`.
fn normalize_path(path: &str) -> String {
    let trailing_slash = path.len() > 1 && path.ends_with('/');
    let mut segments: Vec<&str> = Vec::new();
    for segment in path.split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                segments.pop();
            }
            other => segments.push(other),
        }
    }
    let mut normalized = String::from("/");
    normalized.push_str(&segments.join("/"));
    if trailing_slash && !normalized.ends_with('/') {
        normalized.push('/');
    }
    normalized
}

/// Removes everything from the first `#` on.
fn strip_fragment(url: &str) -> &str {
    match url.find('#') {
        Some(position) => &url[..position],
        None => url,
    }
}

/// Splits `url` into its path and optional query.
fn split_query(url: &str) -> (&str, Option<&str>) {
    match url.find('?') {
        Some(position) => (&url[..position], Some(&url[position + 1..])),
        None => (url, None),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn drops_scripts_styles_and_comments() {
        let extracted = extract(
            "<html><head><style>body{color:red}</style></head><body>\
             <script>alert('x')</script><!-- hidden --><p>visible</p></body></html>",
        );
        assert_eq!(extracted.text, "visible");
        assert_eq!(extracted.title, None);
    }

    #[test]
    fn captures_the_first_title_only() {
        let extracted = extract(
            "<html><head><title> Hello &amp; World </title></head>\
             <body><title>second</title><p>body</p></body></html>",
        );
        assert_eq!(extracted.title.as_deref(), Some("Hello & World"));
        assert_eq!(extracted.text, "body");
    }

    #[test]
    fn block_elements_emit_newlines() {
        let extracted =
            extract("<h1>One</h1><p>Two</p><div>Three</div><ul><li>A</li><li>B</li></ul>");
        assert_eq!(extracted.text, "One\nTwo\nThree\nA\nB");
    }

    #[test]
    fn collapses_horizontal_whitespace() {
        let extracted = extract("<p>a   \n\t  b</p><p> c </p>");
        assert_eq!(extracted.text, "a b\nc");
    }

    #[test]
    fn preserves_preformatted_content() {
        let extracted = extract("<p>code:</p><pre>line 1\n    indented\nline 3</pre><p>after</p>");
        assert_eq!(extracted.text, "code:\nline 1\n    indented\nline 3\nafter");
    }

    #[test]
    fn decodes_named_and_numeric_entities() {
        assert_eq!(
            decode_entities("&lt;a&gt; &amp; &quot;b&quot;"),
            "<a> & \"b\""
        );
        assert_eq!(decode_entities("&#65;&#x42;&#x1F600;"), "AB\u{1f600}");
        assert_eq!(decode_entities("&nbsp;x"), " x");
        assert_eq!(decode_entities("a &unknown; b"), "a &unknown; b");
        assert_eq!(decode_entities("a & b"), "a & b");
        assert_eq!(decode_entities("trailing &"), "trailing &");
        assert_eq!(decode_entities("&#0;"), "&#0;");
    }

    #[test]
    fn malformed_markup_degrades_to_text() {
        assert_eq!(extract("a < b").text, "a < b");
        assert_eq!(extract("<p>unclosed").text, "unclosed");
        assert_eq!(extract("<p>one<p>two").text, "one\ntwo");
        assert_eq!(extract("").text, "");
        assert_eq!(extract("plain text").text, "plain text");
    }

    #[test]
    fn skips_an_unterminated_script() {
        let extracted = extract("<body><p>before</p><script>never closed");
        assert_eq!(extracted.text, "before");
    }

    #[test]
    fn ignores_doctypes_cdata_and_processing_instructions() {
        let extracted = extract(
            "<!DOCTYPE html><?xml version=\"1.0\"?><![CDATA[ignored]]><body><p>kept</p></body>",
        );
        assert_eq!(extracted.text, "kept");
    }

    #[test]
    fn attributes_may_contain_angle_brackets() {
        let extracted = extract("<div title=\"a > b\"><p>inside</p></div>");
        assert_eq!(extracted.text, "inside");
    }

    #[test]
    fn normalizes_absolute_and_relative_references() {
        let base = "https://Example.COM/a/b?x=1";
        assert_eq!(
            normalize_url(base, "../c").as_deref(),
            Some("https://example.com/c")
        );
        assert_eq!(
            normalize_url(base, "/d#frag").as_deref(),
            Some("https://example.com/d")
        );
        assert_eq!(
            normalize_url(base, "//CDN.Example.org/x").as_deref(),
            Some("https://cdn.example.org/x")
        );
        assert_eq!(
            normalize_url(base, "sibling.html").as_deref(),
            Some("https://example.com/a/sibling.html")
        );
        assert_eq!(
            normalize_url(base, "?q=2").as_deref(),
            Some("https://example.com/a/b?q=2")
        );
        assert_eq!(
            normalize_url("https://e.com:8443/x", "y").as_deref(),
            Some("https://e.com:8443/y")
        );
        assert_eq!(
            normalize_url("https://e.com", "page").as_deref(),
            Some("https://e.com/page")
        );
        assert_eq!(
            normalize_url("HTTPS://Example.com", "x").as_deref(),
            Some("https://example.com/x")
        );
    }

    #[test]
    fn rejects_unsupported_references() {
        let base = "https://example.com/a";
        assert_eq!(normalize_url(base, ""), None);
        assert_eq!(normalize_url(base, "#top"), None);
        assert_eq!(normalize_url(base, "mailto:a@b.example"), None);
        assert_eq!(normalize_url(base, "javascript:void(0)"), None);
        assert_eq!(normalize_url(base, "file:///etc/passwd"), None);
        assert_eq!(normalize_url("not a url", "x"), None);
        assert_eq!(normalize_url(base, "http://"), None);
    }
}
