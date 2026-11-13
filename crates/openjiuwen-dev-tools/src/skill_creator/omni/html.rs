//! Rule-based HTML extraction for the `skill_omni_creation` scrape stage.
//!
//! Python scrapes with `BeautifulSoup`; this module recovers the same information with a
//! tolerant hand-written scanner, so the pipeline stays dependency-light, offline-testable
//! and deterministic. [`HtmlDocument::parse`] never fails: every malformed construct
//! degrades to a best-effort document.
//!
//! # Extraction rules
//!
//! - `<script>`, `<style>`, `<noscript>` and `<svg>` bodies plus `<!-- -->` comments are
//!   stripped before anything is read.
//! - `<title>` fills [`HtmlDocument::title`] and `<meta name="description">` fills
//!   [`HtmlDocument::description`]; attribute order is irrelevant.
//! - `<h1>`…`<h6>` fill [`HtmlDocument::headings`] in document order.
//! - `<a href>` fills [`HtmlDocument::links`] with the raw, unresolved `href` text.
//! - `<img>` fills [`HtmlDocument::images`] from `src`, falling back to `data-src` for
//!   lazy-loaded images.
//! - `<pre>` and `<code>` fill [`HtmlDocument::code_blocks`]; a `<code>` nested inside a
//!   `<pre>` is not emitted twice.
//! - [`HtmlDocument::text`] is the tag-free document: block boundaries become single
//!   spaces and every whitespace run collapses to one space.
//!
//! Tag and attribute names are ASCII case-insensitive, attribute values may be
//! double-quoted, single-quoted or unquoted, and attribute values never leak into the text.
//!
//! # Entities
//!
//! Entities are decoded exactly once for this set: `&amp;`, `&lt;`, `&gt;`, `&quot;`,
//! `&#39;`, `&apos;`, `&nbsp;`, `&mdash;`, `&ndash;`, `&hellip;`, `&copy;`, plus decimal
//! `&#NN;` and hexadecimal `&#xHH;` numeric references. Unknown names and invalid or
//! out-of-range numeric references stay literal.
//!
//! # Recovery policy
//!
//! The scanner never panics and never loops: a stray `<` is literal text, an attribute with
//! an unterminated quote consumes to the end of the input, a tag without `>` ends at the end
//! of the input, and raw-text bodies (`script`, `style`, …) without their closing tag are
//! dropped up to the end of the input. An element without its closing tag contributes no
//! entry to its collection, while its text still appears in [`HtmlDocument::text`].
//!
//! Relative references are not resolved here; callers resolve them with [`resolve_url`].

use serde::{Deserialize, Serialize};

/// A heading (`<h1>`…`<h6>`) in document order.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct Heading {
    /// Heading level: `1` for `<h1>` through `6` for `<h6>`.
    pub level: u8,
    /// Whitespace-collapsed heading text with inner tags removed and entities decoded.
    pub text: String,
}

/// An anchor (`<a href>`) in document order.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct Link {
    /// The raw `href` value, trimmed but not resolved against the page URL.
    pub href: String,
    /// Whitespace-collapsed visible text; empty for image-only anchors.
    pub text: String,
}

/// An image (`<img>`) in document order.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct Image {
    /// The raw `src` (or `data-src`) value, trimmed but not resolved.
    pub src: String,
    /// Whitespace-collapsed `alt` text, empty when the tag carries no `alt`.
    pub alt: String,
}

/// The information the scrape stage needs from one HTML page.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct HtmlDocument {
    /// Contents of the first `<title>`, whitespace-collapsed and entity-decoded.
    pub title: String,
    /// Contents of the first `<meta name="description">`, collapsed and decoded.
    pub description: String,
    /// Headings in document order.
    pub headings: Vec<Heading>,
    /// Anchors in document order.
    pub links: Vec<Link>,
    /// Images in document order.
    pub images: Vec<Image>,
    /// `<pre>` and `<code>` bodies in document order, entities decoded, whitespace kept.
    pub code_blocks: Vec<String>,
    /// The tag-free document text with whitespace collapsed.
    pub text: String,
}

impl HtmlDocument {
    /// Extracts a best-effort document from `html`.
    ///
    /// `base_url` is accepted for pipeline symmetry only: links and images stay raw, and
    /// callers resolve them with [`resolve_url`] when they need absolute targets. The
    /// function never fails, panics or hangs, whatever the markup looks like.
    pub fn parse(html: &str, _base_url: &str) -> HtmlDocument {
        let cleaned = strip_noise(html);
        let tokens = tokenize(&cleaned);
        let mut document = HtmlDocument {
            text: extract_text(&tokens),
            ..HtmlDocument::default()
        };
        walk(&tokens, &mut document);
        document
    }
}

/// Walks the token stream once, filling every collection in document order.
fn walk(tokens: &[Token<'_>], document: &mut HtmlDocument) {
    let mut index = 0;
    while index < tokens.len() {
        index = match &tokens[index] {
            Token::Tag(tag) if !tag.closing => match tag.name.as_str() {
                "meta" => {
                    take_description(tag, document);
                    index + 1
                }
                "a" => {
                    take_link(tokens, index, tag, document);
                    index + 1
                }
                "img" => {
                    take_image(tag, document);
                    index + 1
                }
                name @ ("title" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6" | "pre" | "code") => {
                    take_inner(tokens, index, name, document)
                }
                _ => index + 1,
            },
            _ => index + 1,
        };
    }
}

/// Captures the first non-empty `<meta name="description">` content.
fn take_description(tag: &Tag, document: &mut HtmlDocument) {
    if document.description.is_empty() {
        if let Some(content) = meta_description(tag) {
            let text = normalize(content);
            if !text.is_empty() {
                document.description = text;
            }
        }
    }
}

/// Captures the body of `<title>`, an `<h1>`…`<h6>`, a `<pre>` or a `<code>`.
///
/// Returns the next token index, skipping past the close tag. A `<code>` nested in a `<pre>`
/// is never visited, so it cannot be emitted twice.
fn take_inner(
    tokens: &[Token<'_>],
    index: usize,
    name: &str,
    document: &mut HtmlDocument,
) -> usize {
    let Some(close) = find_close(tokens, index, name) else {
        return index + 1;
    };
    let raw = inner_text(tokens, index, close);
    match name {
        "title" => {
            let text = normalize(&raw);
            if document.title.is_empty() && !text.is_empty() {
                document.title = text;
            }
        }
        "pre" | "code" => {
            let content = decode_entities(&raw);
            if !content.trim().is_empty() {
                document.code_blocks.push(content);
            }
        }
        _ => {
            let text = normalize(&raw);
            if !text.is_empty() {
                if let Some(level) = heading_level(name) {
                    document.headings.push(Heading { level, text });
                }
            }
        }
    }
    close + 1
}

/// Captures one `<a href>` anchor, keeping anchors whose visible text is empty.
fn take_link(tokens: &[Token<'_>], index: usize, tag: &Tag, document: &mut HtmlDocument) {
    let Some(href) = tag.attr("href").map(str::trim).filter(|s| !s.is_empty()) else {
        return;
    };
    let text = find_close(tokens, index, "a").map_or_else(String::new, |close| {
        normalize(&inner_text(tokens, index, close))
    });
    document.links.push(Link {
        href: href.to_string(),
        text,
    });
}

/// Captures one `<img>`, preferring `src` and falling back to `data-src`.
fn take_image(tag: &Tag, document: &mut HtmlDocument) {
    let src = tag
        .attr("src")
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .or_else(|| {
            tag.attr("data-src")
                .map(str::trim)
                .filter(|s| !s.is_empty())
        });
    let Some(src) = src else {
        return;
    };
    let alt = tag.attr("alt").map_or_else(String::new, normalize);
    document.images.push(Image {
        src: src.to_string(),
        alt,
    });
}

/// Resolves `href` against `base`, returning an absolute `http`/`https` URL.
///
/// The `href` is trimmed. Empty references, `#`-fragment-only references and every
/// non-`http(s)` scheme (`mailto:`, `javascript:`, `data:`, `tel:`, …) yield `None`, as does a
/// base that is not an absolute `http(s)` URL. Absolute targets lose their fragment and keep
/// their query; `//host/path` inherits the base scheme; `/abs` and relative references are
/// resolved against the base path with `.`/`..` segments normalized (a `..` above the root is
/// dropped). Characters outside the URL structure are passed through unchanged.
pub fn resolve_url(base: &str, href: &str) -> Option<String> {
    let href = href.trim();
    if href.is_empty() || href.starts_with('#') {
        return None;
    }
    let (scheme, authority, base_path) = split_base(base)?;
    if href.starts_with("//") {
        return Some(format!("{scheme}:{}", strip_fragment(href)));
    }
    if let Some(target_scheme) = scheme_of(href) {
        if target_scheme.eq_ignore_ascii_case("http") || target_scheme.eq_ignore_ascii_case("https")
        {
            return Some(strip_fragment(href).to_string());
        }
        return None;
    }
    let href = strip_fragment(href);
    let (href_path, query) = match href.find('?') {
        Some(index) => (&href[..index], &href[index..]),
        None => (href, ""),
    };
    let path = if href_path.starts_with('/') {
        normalize_path(href_path)
    } else if href_path.is_empty() {
        // A query-only reference keeps the base path.
        normalize_path(base_path)
    } else {
        let directory = match base_path.rfind('/') {
            Some(index) => &base_path[..=index],
            None => "/",
        };
        normalize_path(&format!("{directory}{href_path}"))
    };
    Some(format!("{scheme}://{authority}{path}{query}"))
}

/// Splits `base` into scheme, authority and path; `None` unless it is absolute `http(s)`.
fn split_base(base: &str) -> Option<(&str, &str, &str)> {
    let scheme = scheme_of(base)?;
    if !scheme.eq_ignore_ascii_case("http") && !scheme.eq_ignore_ascii_case("https") {
        return None;
    }
    let rest = base.get(scheme.len() + 1..)?.strip_prefix("//")?;
    let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let authority = &rest[..end];
    if authority.is_empty() {
        return None;
    }
    let tail = strip_fragment(&rest[end..]);
    let path = tail.split('?').next().unwrap_or("");
    Some((scheme, authority, path))
}

/// Returns the scheme of `url` when it starts with a syntactically valid scheme.
fn scheme_of(url: &str) -> Option<&str> {
    let end = url.find(':')?;
    let scheme = &url[..end];
    let mut characters = scheme.chars();
    if !characters.next().is_some_and(|c| c.is_ascii_alphabetic()) {
        return None;
    }
    if !characters.all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '-' || c == '.') {
        return None;
    }
    Some(scheme)
}

/// Drops everything from the first `#` on.
fn strip_fragment(url: &str) -> &str {
    match url.find('#') {
        Some(index) => &url[..index],
        None => url,
    }
}

/// Normalizes `.`/`..` segments in a path, dropping any `..` above the root.
fn normalize_path(path: &str) -> String {
    let trailing = path.ends_with('/') || path.ends_with("/.") || path.ends_with("/..");
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
    if trailing && !segments.is_empty() {
        normalized.push('/');
    }
    normalized
}

/// `<meta name="description">` content, when the tag is a description meta tag.
fn meta_description(tag: &Tag) -> Option<&str> {
    let name = tag.attr("name")?;
    if !name.trim().eq_ignore_ascii_case("description") {
        return None;
    }
    tag.attr("content").map(str::trim).filter(|s| !s.is_empty())
}

/// Maps an `h1`…`h6` element name to its numeric level.
fn heading_level(name: &str) -> Option<u8> {
    let mut characters = name.chars();
    if characters.next() != Some('h') {
        return None;
    }
    match characters.next() {
        Some(digit @ '1'..='6') if characters.next().is_none() => Some(digit as u8 - b'0'),
        _ => None,
    }
}

/// Decodes entities and collapses every whitespace run to one space, trimming the ends.
fn normalize(text: &str) -> String {
    let mut collapsed = String::with_capacity(text.len());
    let mut pending_space = false;
    for character in decode_entities(text).chars() {
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

/// Decodes the documented named and numeric entities exactly once.
fn decode_entities(text: &str) -> String {
    if !text.contains('&') {
        return text.to_string();
    }
    let mut decoded = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(position) = rest.find('&') {
        decoded.push_str(&rest[..position]);
        let candidate = &rest[position..];
        if let Some((character, consumed)) = split_entity(candidate) {
            decoded.push(character);
            rest = &candidate[consumed..];
        } else {
            decoded.push('&');
            rest = &candidate[1..];
        }
    }
    decoded.push_str(rest);
    decoded
}

/// Splits one `&...;` entity off `candidate`, returning the character and bytes consumed.
fn split_entity(candidate: &str) -> Option<(char, usize)> {
    let end = candidate
        .char_indices()
        .take(34)
        .find(|&(_, character)| character == ';')
        .map(|(offset, _)| offset)?;
    if end <= 1 {
        return None;
    }
    let name = &candidate[1..end];
    let decoded = match name.strip_prefix('#') {
        Some(number) => numeric_entity(number)?,
        None => named_entity(name)?,
    };
    Some((decoded, end + 1))
}

/// Decodes `#123` / `#x1f600` entity bodies, rejecting invalid and out-of-range code points.
fn numeric_entity(number: &str) -> Option<char> {
    let code = if let Some(hexadecimal) = number
        .strip_prefix('x')
        .or_else(|| number.strip_prefix('X'))
    {
        if hexadecimal.is_empty() || !hexadecimal.bytes().all(|b| b.is_ascii_hexdigit()) {
            return None;
        }
        u32::from_str_radix(hexadecimal, 16).ok()?
    } else {
        if number.is_empty() || !number.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        number.parse::<u32>().ok()?
    };
    if code == 0 {
        return None;
    }
    char::from_u32(code)
}

/// Decodes one named entity body from the documented set.
fn named_entity(name: &str) -> Option<char> {
    let character = match name.to_ascii_lowercase().as_str() {
        "amp" => '&',
        "lt" => '<',
        "gt" => '>',
        "quot" => '"',
        "apos" => '\'',
        "nbsp" => '\u{a0}',
        "mdash" => '\u{2014}',
        "ndash" => '\u{2013}',
        "hellip" => '\u{2026}',
        "copy" => '\u{a9}',
        _ => return None,
    };
    Some(character)
}

/// Removes comments, then the bodies of `<script>`, `<style>`, `<noscript>` and `<svg>`.
fn strip_noise(html: &str) -> String {
    let mut cleaned = strip_comments(html);
    for element in ["script", "style", "noscript", "svg"] {
        cleaned = strip_element(&cleaned, element);
    }
    cleaned
}

/// Removes every `<!-- … -->` comment; an unterminated comment consumes the rest.
fn strip_comments(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut position = 0;
    while let Some(relative) = input[position..].find("<!--") {
        let start = position + relative;
        output.push_str(&input[position..start]);
        let Some(end) = input[start + 4..].find("-->") else {
            return output;
        };
        position = start + 4 + end + 3;
    }
    output.push_str(&input[position..]);
    output
}

/// Removes every `<tag …>…</tag>` region (ASCII case-insensitive), including empty bodies.
fn strip_element(input: &str, name: &str) -> String {
    let opening = format!("<{name}");
    let closing = format!("</{name}");
    let mut output = String::with_capacity(input.len());
    let mut position = 0;
    let mut search = 0;
    while let Some(relative) = find_ascii_case_insensitive(&input[search..], &opening) {
        let start = search + relative;
        let after_name = start + opening.len();
        let is_tag = match input.as_bytes().get(after_name) {
            None => true,
            Some(&byte) => byte == b'>' || byte == b'/' || byte.is_ascii_whitespace(),
        };
        if !is_tag {
            search = after_name;
            continue;
        }
        let Some(open_end) = find_byte(input, b'>', after_name) else {
            output.push_str(&input[position..start]);
            return output;
        };
        output.push_str(&input[position..start]);
        if input.as_bytes().get(open_end - 1) == Some(&b'/') {
            // Self-closing: drop the tag alone.
            position = open_end + 1;
            search = position;
            continue;
        }
        let content_start = open_end + 1;
        let Some(close_relative) = find_ascii_case_insensitive(&input[content_start..], &closing)
        else {
            return output;
        };
        let close_start = content_start + close_relative;
        let Some(close_end) = find_byte(input, b'>', close_start) else {
            return output;
        };
        position = close_end + 1;
        search = position;
    }
    output.push_str(&input[position..]);
    output
}

/// Finds the first ASCII case-insensitive occurrence of `needle` in `haystack`.
fn find_ascii_case_insensitive(haystack: &str, needle: &str) -> Option<usize> {
    let bytes = haystack.as_bytes();
    let needle_bytes = needle.as_bytes();
    if needle_bytes.is_empty() || bytes.len() < needle_bytes.len() {
        return None;
    }
    (0..=bytes.len() - needle_bytes.len())
        .find(|&index| bytes[index..index + needle_bytes.len()].eq_ignore_ascii_case(needle_bytes))
}

/// Position after the next `>`, or the end of the input when the tag is unterminated.
fn skip_to_gt(html: &str, from: usize) -> usize {
    find_byte(html, b'>', from).map_or(html.len(), |position| position + 1)
}

/// Finds the first `byte` at or after `from`.
fn find_byte(text: &str, byte: u8, from: usize) -> Option<usize> {
    text.as_bytes()
        .get(from..)?
        .iter()
        .position(|&candidate| candidate == byte)
        .map(|offset| offset + from)
}

/// One lexical element of the cleaned document.
#[derive(Debug)]
enum Token<'a> {
    /// A raw text run.
    Text(&'a str),
    /// A start tag, end tag or skipped declaration.
    Tag(Tag),
}

/// A parsed tag with lowercased name and attribute names.
#[derive(Debug)]
struct Tag {
    /// Lowercased element name; empty for skipped declarations and stray `</`.
    name: String,
    /// Attribute name/value pairs in source order, values entity-decoded.
    attrs: Vec<(String, String)>,
    /// Whether this is an end tag.
    closing: bool,
}

impl Tag {
    /// A tag that contributes nothing (declaration, stray `</`).
    fn skipped() -> Tag {
        Tag {
            name: String::new(),
            attrs: Vec::new(),
            closing: true,
        }
    }

    /// The first value of the attribute named `name` (already lowercased).
    fn attr(&self, name: &str) -> Option<&str> {
        self.attrs
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }
}

/// Splits the cleaned document into text runs and tags.
fn tokenize(html: &str) -> Vec<Token<'_>> {
    let mut tokens = Vec::new();
    let mut index = 0;
    let mut text_start = 0;
    while index < html.len() {
        if html.as_bytes()[index] != b'<' {
            index += 1;
            continue;
        }
        if let Some((tag, end)) = parse_tag(html, index) {
            if text_start < index {
                tokens.push(Token::Text(&html[text_start..index]));
            }
            tokens.push(Token::Tag(tag));
            index = end;
            text_start = end;
        } else {
            // A stray `<` is literal text.
            index += 1;
        }
    }
    if text_start < html.len() {
        tokens.push(Token::Text(&html[text_start..]));
    }
    tokens
}

/// Parses the tag starting at the `<` at `start`, returning it and the position after it.
///
/// Returns `None` when the `<` does not start a tag, leaving it to the caller as literal text.
fn parse_tag(html: &str, start: usize) -> Option<(Tag, usize)> {
    let bytes = html.as_bytes();
    let length = bytes.len();
    let mut index = start + 1;
    if index >= length {
        return None;
    }
    let mut closing = false;
    match bytes[index] {
        b'/' => {
            closing = true;
            index += 1;
        }
        b'!' | b'?' => return Some((Tag::skipped(), skip_to_gt(html, index))),
        _ => {}
    }
    if index >= length {
        return Some((Tag::skipped(), length));
    }
    let name_start = index;
    while index < length {
        let byte = bytes[index];
        if byte.is_ascii_alphanumeric() || byte == b'-' || byte == b':' || byte == b'_' {
            index += 1;
        } else {
            break;
        }
    }
    if index == name_start || !bytes[name_start].is_ascii_alphabetic() {
        if closing {
            return Some((Tag::skipped(), skip_to_gt(html, index)));
        }
        return None;
    }
    let name = html[name_start..index].to_ascii_lowercase();
    if closing {
        return Some((
            Tag {
                name,
                attrs: Vec::new(),
                closing,
            },
            skip_to_gt(html, index),
        ));
    }
    let mut attrs = Vec::new();
    let end = parse_attributes(html, index, &mut attrs);
    Some((
        Tag {
            name,
            attrs,
            closing,
        },
        end,
    ))
}

/// Parses the attribute list of a start tag, returning the position after the tag.
///
/// Values may be double-quoted, single-quoted or unquoted. An unterminated quoted value
/// consumes to the end of the input, and a tag without `>` ends at the end of the input.
fn parse_attributes(html: &str, mut index: usize, attrs: &mut Vec<(String, String)>) -> usize {
    let bytes = html.as_bytes();
    let length = bytes.len();
    let mut end = length;
    loop {
        while index < length && bytes[index].is_ascii_whitespace() {
            index += 1;
        }
        if index >= length {
            break;
        }
        match bytes[index] {
            b'>' => {
                index += 1;
                end = index;
                break;
            }
            b'/' => {
                index += 1;
            }
            _ => {
                let attr_start = index;
                while index < length {
                    let byte = bytes[index];
                    if byte.is_ascii_whitespace() || byte == b'=' || byte == b'>' || byte == b'/' {
                        break;
                    }
                    index += 1;
                }
                let attr_name = html[attr_start..index].to_ascii_lowercase();
                while index < length && bytes[index].is_ascii_whitespace() {
                    index += 1;
                }
                let value = parse_attribute_value(html, &mut index);
                if !attr_name.is_empty() {
                    attrs.push((attr_name, decode_entities(&value)));
                }
            }
        }
    }
    end
}

/// Parses one optional `=value` attribute value, advancing past it.
fn parse_attribute_value(html: &str, index: &mut usize) -> String {
    let bytes = html.as_bytes();
    let length = bytes.len();
    if *index >= length || bytes[*index] != b'=' {
        return String::new();
    }
    *index += 1;
    while *index < length && bytes[*index].is_ascii_whitespace() {
        *index += 1;
    }
    if *index >= length {
        return String::new();
    }
    let quote = bytes[*index];
    if quote == b'"' || quote == b'\'' {
        *index += 1;
        let value_start = *index;
        while *index < length && bytes[*index] != quote {
            *index += 1;
        }
        let value = html[value_start..*index].to_string();
        if *index < length {
            *index += 1;
        }
        return value;
    }
    let value_start = *index;
    while *index < length && !bytes[*index].is_ascii_whitespace() && bytes[*index] != b'>' {
        *index += 1;
    }
    html[value_start..*index].to_string()
}

/// Index of the end tag matching the start tag at `open`, counting nested same-name tags.
fn find_close(tokens: &[Token<'_>], open: usize, name: &str) -> Option<usize> {
    let mut depth = 0_usize;
    for (index, token) in tokens.iter().enumerate().skip(open) {
        if let Token::Tag(tag) = token {
            if tag.name == name {
                if tag.closing {
                    if depth == 1 {
                        return Some(index);
                    }
                    depth = depth.saturating_sub(1);
                } else {
                    depth += 1;
                }
            }
        }
    }
    None
}

/// Concatenates the raw text runs between two token positions.
fn inner_text(tokens: &[Token<'_>], open: usize, close: usize) -> String {
    let mut text = String::new();
    for token in &tokens[open + 1..close] {
        if let Token::Text(run) = token {
            text.push_str(run);
        }
    }
    text
}

/// Space-separated names of the block-level elements whose boundaries become one space.
const BLOCK_ELEMENTS: &str = " address article aside blockquote br dd div dl dt figcaption \
    figure footer form h1 h2 h3 h4 h5 h6 header hr li main nav ol p pre section table tbody \
    td tfoot th thead tr ul ";

/// Whether `name` is one of the [`BLOCK_ELEMENTS`].
fn is_block(name: &str) -> bool {
    BLOCK_ELEMENTS.split(' ').any(|candidate| candidate == name)
}

/// Renders the tag-free document text with block boundaries as single spaces.
fn extract_text(tokens: &[Token<'_>]) -> String {
    let mut raw = String::new();
    for token in tokens {
        match token {
            Token::Text(run) => raw.push_str(run),
            Token::Tag(tag) => {
                if is_block(&tag.name) {
                    raw.push(' ');
                }
            }
        }
    }
    normalize(&raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: &str = r#"<!doctype html>
<html><head>
<title>  My   Page </title>
<meta name="description" content="A &amp; B page">
</head>
<body>
<h1>Welcome</h1>
<h2>Features &amp; More</h2>
<a href="/docs">Docs</a>
<a href="https://example.com/x?y=1#frag">External</a>
<img src="/img/a.png" alt="Alpha">
<img alt="Beta" data-src="/img/b.png">
<pre><code>let x = 1;
fn main() {}</code></pre>
<p>Inline <code>cargo test</code> runs.</p>
</body></html>
"#;

    fn heading_pairs(document: &HtmlDocument) -> Vec<(u8, &str)> {
        document
            .headings
            .iter()
            .map(|h| (h.level, h.text.as_str()))
            .collect()
    }

    fn link_pairs(document: &HtmlDocument) -> Vec<(&str, &str)> {
        document
            .links
            .iter()
            .map(|l| (l.href.as_str(), l.text.as_str()))
            .collect()
    }

    fn image_pairs(document: &HtmlDocument) -> Vec<(&str, &str)> {
        document
            .images
            .iter()
            .map(|i| (i.src.as_str(), i.alt.as_str()))
            .collect()
    }

    fn check(base: &str, href: &str, expected: Option<&str>) {
        assert_eq!(
            resolve_url(base, href).as_deref(),
            expected,
            "{base} + {href}"
        );
    }

    #[test]
    fn parse_extracts_every_collection() {
        let document = HtmlDocument::parse(PAGE, "https://example.com");
        assert_eq!(document.title, "My Page");
        assert_eq!(document.description, "A & B page");
        assert_eq!(
            heading_pairs(&document),
            vec![(1, "Welcome"), (2, "Features & More")]
        );
        assert_eq!(
            link_pairs(&document),
            vec![
                ("/docs", "Docs"),
                ("https://example.com/x?y=1#frag", "External")
            ]
        );
        assert_eq!(
            image_pairs(&document),
            vec![("/img/a.png", "Alpha"), ("/img/b.png", "Beta")]
        );
        assert_eq!(
            document.code_blocks,
            vec!["let x = 1;\nfn main() {}", "cargo test"]
        );
        assert!(document.text.contains("Welcome"));
        assert!(document.text.contains("Inline cargo test runs."));
        assert!(!document.text.contains('<'));
    }

    #[test]
    fn parse_decodes_entities_exactly_once() {
        let html = "<p>&amp; &lt; &gt; &quot; &#39; &apos; &nbsp; &mdash; &ndash; \
                    &hellip; &copy; &#65; &#x42; &amp;lt; &unknown; &#xZZ; &#999999999;</p>";
        let text = HtmlDocument::parse(html, "").text;
        assert!(text.starts_with("& < > \" ' ' "), "{text}");
        assert!(text.contains('\u{2014}') && text.contains('\u{2013}'));
        assert!(text.contains('\u{2026}') && text.contains('\u{a9}') && text.contains("A B"));
        for literal in ["&lt;", "&unknown;", "&#xZZ;", "&#999999999;"] {
            assert!(text.contains(literal), "{literal} missing from {text}");
        }
        assert!(!text.contains("&amp;lt;"));
    }

    #[test]
    fn parse_strips_raw_text_elements_and_comments() {
        let html = r#"<html><head><style>body { color: red; }</style>
<script>var secret = "leak";</script></head><body><!-- hidden comment -->
<noscript>enable js</noscript><svg><path d="M0 0"/></svg>
<h1>Visible</h1><a href="/x">Link</a><pre>keep</pre></body></html>"#;
        let document = HtmlDocument::parse(html, "");
        for leaked in [
            "color",
            "secret",
            "leak",
            "hidden comment",
            "enable js",
            "M0 0",
        ] {
            assert!(!document.text.contains(leaked), "{leaked} leaked into text");
        }
        assert_eq!(heading_pairs(&document), vec![(1, "Visible")]);
        assert_eq!(link_pairs(&document), vec![("/x", "Link")]);
        assert_eq!(document.code_blocks, vec!["keep"]);
    }

    #[test]
    fn parse_is_attribute_order_independent() {
        let first = HtmlDocument::parse(
            r#"<meta name="description" content="First"><img src="/a.png" alt="A"><a href="/x">X</a>"#,
            "",
        );
        let second = HtmlDocument::parse(
            r#"<meta content="First" name="description"><img alt="A" src="/a.png"><a target="" href="/x">X</a>"#,
            "",
        );
        assert_eq!(first.description, "First");
        assert_eq!(first.description, second.description);
        assert_eq!(first.images, second.images);
        assert_eq!(first.links, second.links);
    }

    #[test]
    fn parse_survives_unterminated_markup() {
        for html in ["<p>before</p><script>var x = 1;", "<a href=>Click</a>"] {
            let document = HtmlDocument::parse(html, "");
            assert_eq!(document.links.len(), 0, "{html}");
        }
        assert_eq!(
            HtmlDocument::parse("<p>before</p><script>var x = 1;", "").text,
            "before"
        );
        assert_eq!(HtmlDocument::parse("<a href=>Click</a>", "").text, "Click");
        let unquoted = HtmlDocument::parse("<a href=unquoted>Bare</a>", "");
        assert_eq!(link_pairs(&unquoted), vec![("unquoted", "Bare")]);
        let document = HtmlDocument::parse("<h1>One</h1><h2>Two", "");
        assert_eq!(heading_pairs(&document), vec![(1, "One")]);
        assert!(document.text.contains("Two"));
        assert_eq!(
            image_pairs(&HtmlDocument::parse("<img src='x>", "")),
            vec![("x>", "")]
        );
    }

    #[test]
    fn parse_handles_empty_and_plain_text() {
        assert_eq!(
            HtmlDocument::parse("", "https://example.com"),
            HtmlDocument::default()
        );
        let document = HtmlDocument::parse("Just some plain text.", "");
        assert_eq!(document.text, "Just some plain text.");
        let collection_lengths = (
            document.title.len(),
            document.description.len(),
            document.headings.len(),
            document.links.len(),
            document.images.len(),
            document.code_blocks.len(),
        );
        assert_eq!(collection_lengths, (0, 0, 0, 0, 0, 0));
    }

    #[test]
    fn text_is_collapsed_and_free_of_markup() {
        let html = r#"<div class="wrapper" data-id="42">
        <p>Hello
           <b>bold</b>   world</p>
        <br><p>Second</p></div>"#;
        let document = HtmlDocument::parse(html, "");
        assert_eq!(document.text, "Hello bold world Second");
        for leaked in ["wrapper", "42", "div", "class"] {
            assert!(!document.text.contains(leaked), "{leaked} leaked into text");
        }
    }

    #[test]
    fn pre_wins_over_nested_code() {
        let html = "<pre><code>inner</code></pre><code>standalone</code>";
        assert_eq!(
            HtmlDocument::parse(html, "").code_blocks,
            vec!["inner", "standalone"]
        );
    }

    #[test]
    fn parse_is_deterministic() {
        let first = HtmlDocument::parse(PAGE, "https://example.com");
        assert_eq!(first, HtmlDocument::parse(PAGE, "https://example.com"));
        let hrefs: Vec<&str> = first.links.iter().map(|link| link.href.as_str()).collect();
        assert_eq!(hrefs, vec!["/docs", "https://example.com/x?y=1#frag"]);
    }

    #[test]
    fn html_document_round_trips_json() {
        let document = HtmlDocument::parse(PAGE, "https://example.com");
        let json = serde_json::to_string(&document).unwrap();
        let restored: HtmlDocument = serde_json::from_str(&json).unwrap();
        assert_eq!(document, restored);
        let partial: HtmlDocument = serde_json::from_str(r#"{"title":"T","unknown":1}"#).unwrap();
        assert_eq!(partial.title, "T");
        assert_eq!(partial.headings.len(), 0);
    }

    #[test]
    fn resolve_url_matrix() {
        let base = "https://example.com/dir/sub/page.html?x=1";
        // Absolute and protocol-relative targets keep their query and lose their fragment.
        check(
            base,
            "https://other.org/x?q=1#f",
            Some("https://other.org/x?q=1"),
        );
        check(
            "http://example.com/a",
            "//cdn.example.com/lib.js",
            Some("http://cdn.example.com/lib.js"),
        );
        check(
            "https://example.com",
            "http://plain.org/",
            Some("http://plain.org/"),
        );
        // Root-relative and relative paths resolve against the base directory.
        check(base, "/root/x", Some("https://example.com/root/x"));
        check(
            base,
            "img/a.png",
            Some("https://example.com/dir/sub/img/a.png"),
        );
        check(base, "./x", Some("https://example.com/dir/sub/x"));
        check(base, "sub/", Some("https://example.com/dir/sub/sub/"));
        check(
            "https://example.com",
            "a/b",
            Some("https://example.com/a/b"),
        );
        // Dot segments are normalized and never climb above the root.
        check(base, "../a/b", Some("https://example.com/dir/a/b"));
        check(base, "../../../x", Some("https://example.com/x"));
        check(
            base,
            "../../../../../../../x",
            Some("https://example.com/x"),
        );
        check(base, "/a/./b/../c", Some("https://example.com/a/c"));
        // Queries survive resolution; fragments do not.
        check(
            base,
            "next?q=1&r=2#frag",
            Some("https://example.com/dir/sub/next?q=1&r=2"),
        );
        check(
            base,
            "https://example.com/x#frag",
            Some("https://example.com/x"),
        );
        check(
            base,
            "?only=query",
            Some("https://example.com/dir/sub/page.html?only=query"),
        );
        // Empty, fragment-only and non-http(s) references are rejected, as are non-http bases.
        for href in [
            "",
            "   ",
            "#frag",
            "#",
            "mailto:a@b.c",
            "javascript:void(0)",
            "data:text/plain,hi",
            "tel:+123",
            "ftp://example.com/x",
        ] {
            check(base, href, None);
        }
        check("ftp://example.com/x", "a", None);
        check("example.com/x", "a", None);
        check("", "a", None);
    }
}
