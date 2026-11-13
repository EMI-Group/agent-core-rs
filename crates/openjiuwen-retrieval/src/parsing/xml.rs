//! Tolerant XML parsing for the `OOXML` vocabularies.
//!
//! `DOCX` and `XLSX` parts are XML documents in a handful of namespaces
//! (`w:`, `r:`, `dc:`, or none at all). This module turns such a part into an
//! [`XmlDocument`] tree: elements, attributes, ordered children, and direct text.
//!
//! Scope, deliberately:
//!
//! - prolog/processing instructions, comments, `CDATA`, markup declarations
//!   (`<!DOCTYPE …>`), self-closing tags, nested elements, and attributes in
//!   single or double quotes;
//! - entity decoding for `&amp; &lt; &gt; &quot; &apos;`, `&#NN;`, and `&#xNN;`
//!   (unknown or malformed entities are kept verbatim);
//! - namespace prefixes preserved verbatim (`w:p` stays `w:p`), so callers can
//!   either match the qualified name or use [`local_name`];
//! - whitespace text preserved as written.
//!
//! There is no DTD, namespace resolution, or schema validation: undeclared
//! prefixes are simply part of the name. Structural damage — an unclosed
//! element, a mismatched end tag, an unterminated comment/`CDATA`/attribute, or
//! more than [`MAX_DEPTH`] nested elements — returns [`Error::Validation`]
//! naming the element, so a truncated part is reported instead of silently
//! producing half a document. Stray text outside the root element is rejected
//! for the same reason; stray comments and processing instructions are ignored.
//!
//! # Example
//!
//! ```
//! use openjiuwen_retrieval::parsing::xml::parse;
//!
//! let document = parse(r#"<?xml version="1.0"?><w:document><w:body>
//!     <w:p w:rsid="00"><w:t>A &amp; B</w:t></w:p>
//! </w:body></w:document>"#).unwrap();
//! let body = document.root().unwrap().find_first("body").unwrap();
//! let paragraph = body.find_first("w:p").unwrap();
//! assert_eq!(paragraph.find_first("w:t").unwrap().text, "A & B");
//! ```

use openjiuwen_common::error::{Error, Result};

/// Maximum element nesting depth accepted by [`parse`].
///
/// The parsers are recursive, so the limit is what keeps a hostile document from
/// exhausting the stack; real `OOXML` parts nest a few dozen levels at most.
pub const MAX_DEPTH: usize = 256;

/// Longest entity name/number sequence decoded between `&` and `;`.
const MAX_ENTITY_LEN: usize = 12;

/// One XML element: its name, attributes, child elements, and direct text.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct XmlElement {
    /// Element name as written, namespace prefix included (`w:p`).
    pub name: String,
    /// Attributes in document order; values are entity-decoded, prefixes kept.
    pub attributes: Vec<(String, String)>,
    /// Child elements in document order.
    pub children: Vec<XmlElement>,
    /// Direct text (entity-decoded, `CDATA` included), concatenated across child
    /// boundaries; leading and trailing whitespace is preserved.
    pub text: String,
}

impl XmlElement {
    /// Returns the first descendant element called `name`.
    ///
    /// A name without a prefix also matches by local name, so `find_first("p")`
    /// finds `<w:p>`; a prefixed name must match exactly. The element itself is
    /// not a candidate — use the parent for that.
    pub fn find_first(&self, name: &str) -> Option<&Self> {
        for child in &self.children {
            if name_matches(&child.name, name) {
                return Some(child);
            }
            if let Some(found) = child.find_first(name) {
                return Some(found);
            }
        }
        None
    }

    /// Returns every descendant element called `name`, in document order.
    ///
    /// Name matching follows [`XmlElement::find_first`].
    pub fn find_all(&self, name: &str) -> Vec<&Self> {
        let mut found = Vec::new();
        self.collect_all(name, &mut found);
        found
    }

    /// Returns the direct children called `name`.
    pub fn children_named(&self, name: &str) -> Vec<&Self> {
        self.children
            .iter()
            .filter(|child| name_matches(&child.name, name))
            .collect()
    }

    /// Returns the value of attribute `name` (entity-decoded).
    ///
    /// A name without a prefix also matches by local name (`attr("val")` finds
    /// `w:val`); a prefixed name must match exactly.
    pub fn attr(&self, name: &str) -> Option<&str> {
        self.attributes
            .iter()
            .find(|(key, _)| name_matches(key, name))
            .map(|(_, value)| value.as_str())
    }

    /// Returns the element name without its namespace prefix (`w:p` → `p`).
    pub fn local_name(&self) -> &str {
        local_name(&self.name)
    }

    /// Returns all descendant text, in document order.
    ///
    /// Direct text is concatenated first and the children's text follows, which
    /// is exact for `OOXML` (where an element carries either text or children,
    /// not both interleaved).
    pub fn text_content(&self) -> String {
        let mut text = self.text.clone();
        for child in &self.children {
            text.push_str(&child.text_content());
        }
        text
    }

    /// Recursively collects descendants called `name`.
    fn collect_all<'a>(&'a self, name: &str, found: &mut Vec<&'a Self>) {
        for child in &self.children {
            if name_matches(&child.name, name) {
                found.push(child);
            }
            child.collect_all(name, found);
        }
    }
}

/// A parsed XML document: an optional `<?xml …?>` prolog and its top-level
/// elements in document order.
///
/// Well-formed XML has a single root, which [`XmlDocument::root`] returns;
/// fragments with several roots are tolerated for callers that concatenate
/// parts.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct XmlDocument {
    /// Body of the first `<?xml …?>` declaration, trimmed, when present.
    pub prolog: Option<String>,
    /// Top-level elements in document order.
    pub roots: Vec<XmlElement>,
}

impl XmlDocument {
    /// Returns the first top-level element (the document root in well-formed XML).
    pub fn root(&self) -> Option<&XmlElement> {
        self.roots.first()
    }
}

/// Returns `name` without its namespace prefix (`w:p` → `p`, `p` → `p`).
pub fn local_name(name: &str) -> &str {
    name.rsplit_once(':').map_or(name, |(_, local)| local)
}

/// Returns `true` when `candidate` satisfies the requested name.
///
/// Prefixed requests match exactly; unprefixed requests also match by local name.
fn name_matches(candidate: &str, wanted: &str) -> bool {
    candidate == wanted || (!wanted.contains(':') && local_name(candidate) == wanted)
}

/// Parses an XML document (or fragment) into a tree.
///
/// # Errors
///
/// Returns [`Error::Validation`] for structurally invalid input: an unclosed
/// element, a mismatched end tag, an unterminated comment/`CDATA`/processing
/// instruction/attribute value, text outside the root element, or nesting deeper
/// than [`MAX_DEPTH`].
pub fn parse(input: &str) -> Result<XmlDocument> {
    // A leading byte-order mark is not part of the markup.
    let input = input.strip_prefix('\u{feff}').unwrap_or(input);
    let mut scanner = Scanner::new(input);
    let mut document = XmlDocument::default();
    loop {
        scanner.skip_whitespace();
        let Some(byte) = scanner.peek() else {
            break;
        };
        if byte != b'<' {
            let text = scanner.take_text();
            if !text.trim().is_empty() {
                return Err(scanner.error("text outside the root element"));
            }
        } else if scanner.starts_with("<?") {
            let body = scanner.take_processing_instruction()?;
            if document.prolog.is_none() {
                document.prolog = (!body.is_empty()).then_some(body);
            }
        } else if scanner.starts_with("<!--") {
            scanner.skip_comment()?;
        } else if scanner.starts_with("<![CDATA[") {
            let _ = scanner.take_cdata()?;
        } else if scanner.starts_with("<!") {
            scanner.skip_declaration()?;
        } else if scanner.starts_with("</") {
            return Err(scanner.error("unexpected end tag outside the root element"));
        } else {
            document.roots.push(scanner.parse_element(0)?);
        }
    }
    Ok(document)
}

/// Decodes the five named entities plus decimal and hexadecimal references.
///
/// Anything that is not a complete, known `&…;` sequence is kept verbatim, so
/// document text such as `a & b` or `&nbsp;` survives unchanged.
fn decode_entities(raw: &str) -> String {
    if !raw.contains('&') {
        return raw.to_owned();
    }
    let mut decoded = String::with_capacity(raw.len());
    let mut rest = raw;
    while let Some(index) = rest.find('&') {
        decoded.push_str(&rest[..index]);
        let after = &rest[index + 1..];
        let Some(end) = after.find(';').filter(|end| *end <= MAX_ENTITY_LEN) else {
            decoded.push('&');
            rest = after;
            continue;
        };
        let Some(character) = decode_entity(&after[..end]) else {
            decoded.push('&');
            rest = after;
            continue;
        };
        decoded.push(character);
        rest = &after[end + 1..];
    }
    decoded.push_str(rest);
    decoded
}

/// Resolves one entity body (the text between `&` and `;`).
fn decode_entity(entity: &str) -> Option<char> {
    match entity {
        "amp" => Some('&'),
        "lt" => Some('<'),
        "gt" => Some('>'),
        "quot" => Some('"'),
        "apos" => Some('\''),
        _ => {
            let numeric = entity.strip_prefix('#')?;
            let value = match numeric
                .strip_prefix('x')
                .or_else(|| numeric.strip_prefix('X'))
            {
                Some(hex) => u32::from_str_radix(hex, 16).ok()?,
                None => numeric.parse::<u32>().ok()?,
            };
            char::from_u32(value)
        }
    }
}

/// Byte scanner over the input; every `pos` it records sits on a `char`
/// boundary because it only ever stops at ASCII delimiters.
struct Scanner<'a> {
    /// The original input, used for zero-copy slices.
    input: &'a str,
    /// The same input as bytes, used for marker searches.
    bytes: &'a [u8],
    /// Current byte offset.
    pos: usize,
}

impl<'a> Scanner<'a> {
    /// Starts a scan at the beginning of `input`.
    fn new(input: &'a str) -> Self {
        Self {
            input,
            bytes: input.as_bytes(),
            pos: 0,
        }
    }

    /// Returns the byte at the cursor without consuming it.
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    /// Returns `true` when the input continues with `prefix`.
    fn starts_with(&self, prefix: &str) -> bool {
        self.bytes[self.pos..].starts_with(prefix.as_bytes())
    }

    /// Builds a validation error that reports the current offset.
    fn error(&self, message: impl std::fmt::Display) -> Error {
        Error::Validation(format!(
            "invalid XML at byte {}: {message}",
            self.pos.min(self.bytes.len())
        ))
    }

    /// Advances past ASCII whitespace.
    fn skip_whitespace(&mut self) {
        while let Some(byte) = self.peek() {
            if !byte.is_ascii_whitespace() {
                break;
            }
            self.pos += 1;
        }
    }

    /// Parses one element (and its subtree) starting at `<`.
    fn parse_element(&mut self, depth: usize) -> Result<XmlElement> {
        if depth >= MAX_DEPTH {
            return Err(self.error(format!("element nesting exceeds {MAX_DEPTH} levels")));
        }
        self.pos += 1; // '<'
        let name = self.take_name()?;
        let mut attributes = Vec::new();
        loop {
            self.skip_whitespace();
            match self.peek() {
                None => return Err(self.error(format!("unclosed start tag <{name}>"))),
                Some(b'>') => {
                    self.pos += 1;
                    break;
                }
                Some(b'/') => {
                    self.pos += 1;
                    if self.peek() != Some(b'>') {
                        return Err(self.error(format!("malformed self-closing tag <{name}/>")));
                    }
                    self.pos += 1;
                    return Ok(XmlElement {
                        name,
                        attributes,
                        children: Vec::new(),
                        text: String::new(),
                    });
                }
                Some(_) => {
                    let attribute = self.take_name()?;
                    self.skip_whitespace();
                    if self.peek() != Some(b'=') {
                        return Err(self.error(format!("attribute {attribute:?} has no value")));
                    }
                    self.pos += 1;
                    self.skip_whitespace();
                    let value = self.take_attribute_value()?;
                    attributes.push((attribute, value));
                }
            }
        }
        let mut element = XmlElement {
            name,
            attributes,
            children: Vec::new(),
            text: String::new(),
        };
        loop {
            if self.pos >= self.bytes.len() {
                return Err(self.error(format!("unclosed element <{}>", element.name)));
            }
            if self.starts_with("</") {
                self.pos += 2;
                let end_name = self.take_name()?;
                self.skip_whitespace();
                if self.peek() != Some(b'>') {
                    return Err(self.error(format!("malformed end tag </{end_name}>")));
                }
                self.pos += 1;
                if end_name != element.name {
                    return Err(self.error(format!(
                        "mismatched end tag </{end_name}> for <{}>",
                        element.name
                    )));
                }
                return Ok(element);
            }
            if self.starts_with("<!--") {
                self.skip_comment()?;
            } else if self.starts_with("<![CDATA[") {
                let cdata = self.take_cdata()?;
                element.text.push_str(&cdata);
            } else if self.starts_with("<!") {
                self.skip_declaration()?;
            } else if self.starts_with("<?") {
                let _ = self.take_processing_instruction()?;
            } else if self.peek() == Some(b'<') {
                let child = self.parse_element(depth + 1)?;
                element.children.push(child);
            } else {
                let raw = self.take_text();
                element.text.push_str(&decode_entities(raw));
            }
        }
    }

    /// Reads an element or attribute name up to the next delimiter.
    fn take_name(&mut self) -> Result<String> {
        let start = self.pos;
        while let Some(byte) = self.peek() {
            let ends_name = byte.is_ascii_whitespace()
                || matches!(byte, b'/' | b'>' | b'=' | b'<' | b'"' | b'\'');
            if ends_name {
                break;
            }
            self.pos += 1;
        }
        if start == self.pos {
            return Err(self.error("expected an element or attribute name"));
        }
        Ok(self.input[start..self.pos].to_owned())
    }

    /// Reads a quoted attribute value and decodes its entities.
    fn take_attribute_value(&mut self) -> Result<String> {
        let Some(quote) = self.peek() else {
            return Err(self.error("unterminated attribute value"));
        };
        if quote != b'"' && quote != b'\'' {
            return Err(self.error("attribute value must be quoted"));
        }
        self.pos += 1;
        let start = self.pos;
        while let Some(byte) = self.peek() {
            if byte == quote {
                let raw = &self.input[start..self.pos];
                self.pos += 1;
                return Ok(decode_entities(raw));
            }
            self.pos += 1;
        }
        Err(self.error("unterminated attribute value"))
    }

    /// Consumes text up to the next `<` (or the end of input).
    fn take_text(&mut self) -> &'a str {
        let start = self.pos;
        while let Some(byte) = self.peek() {
            if byte == b'<' {
                break;
            }
            self.pos += 1;
        }
        &self.input[start..self.pos]
    }

    /// Skips a `<!-- … -->` comment.
    fn skip_comment(&mut self) -> Result<()> {
        let start = self.pos + "<!--".len();
        let Some(end) = find(&self.bytes[start..], b"-->") else {
            return Err(self.error("unterminated comment"));
        };
        self.pos = start + end + 3;
        Ok(())
    }

    /// Reads a `<![CDATA[ … ]]>` section body.
    fn take_cdata(&mut self) -> Result<String> {
        let start = self.pos + "<![CDATA[".len();
        let Some(end) = find(&self.bytes[start..], b"]]>") else {
            return Err(self.error("unterminated CDATA section"));
        };
        let text = self.input[start..start + end].to_owned();
        self.pos = start + end + 3;
        Ok(text)
    }

    /// Reads a `<? … ?>` processing instruction body.
    fn take_processing_instruction(&mut self) -> Result<String> {
        let start = self.pos + "<?".len();
        let Some(end) = find(&self.bytes[start..], b"?>") else {
            return Err(self.error("unterminated processing instruction"));
        };
        let body = self.input[start..start + end].trim().to_owned();
        self.pos = start + end + 2;
        Ok(body)
    }

    /// Skips a `<! … >` markup declaration such as `<!DOCTYPE …>`.
    fn skip_declaration(&mut self) -> Result<()> {
        self.pos += 2; // "<!"
        let mut brackets = 0usize;
        let mut quote: Option<u8> = None;
        while let Some(byte) = self.peek() {
            self.pos += 1;
            match (quote, byte) {
                (Some(open), current) if current == open => quote = None,
                (None, b'"' | b'\'') => quote = Some(byte),
                (None, b'[') => brackets += 1,
                (None, b']') => brackets = brackets.saturating_sub(1),
                (None, b'>') if brackets == 0 => return Ok(()),
                (_, _) => {}
            }
        }
        Err(self.error("unterminated markup declaration"))
    }
}

/// Returns the offset of `needle` in `haystack`.
fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use openjiuwen_common::error::ErrorKind;

    use super::*;

    #[test]
    fn parses_nested_elements_attributes_and_ordered_children() {
        let document = parse(
            r#"<?xml version="1.0" encoding="UTF-8"?>
            <w:document xmlns:w="urn:word">
                <w:body>
                    <w:p w:rsid="00A" w:val='single'><w:t>hello</w:t></w:p>
                    <w:p/>
                </w:body>
            </w:document>"#,
        )
        .unwrap();
        assert_eq!(
            document.prolog.as_deref(),
            Some(r#"xml version="1.0" encoding="UTF-8""#)
        );
        let root = document.root().unwrap();
        assert_eq!(root.name, "w:document");
        assert_eq!(root.attr("xmlns:w"), Some("urn:word"));
        assert_eq!(root.local_name(), "document");
        let body = root.find_first("body").unwrap();
        assert_eq!(body.children.len(), 2);
        let first = body.children[0].clone();
        assert_eq!(first.name, "w:p");
        assert_eq!(first.attr("w:rsid"), Some("00A"));
        assert_eq!(first.attr("w:val"), Some("single"));
        assert_eq!(first.attr("rsid"), Some("00A"));
        assert_eq!(first.find_first("t").unwrap().text, "hello");
        assert_eq!(first.text_content(), "hello");
        assert_eq!(first.children_named("w:t").len(), 1);
        assert_eq!(first.children_named("tbl"), Vec::<&XmlElement>::new());
        assert_eq!(body.find_all("p").len(), 2);
        assert!(document.root().unwrap().find_first("missing").is_none());
    }

    #[test]
    fn decodes_named_decimal_and_hex_entities() {
        let document = parse(
            "<a>&amp;&lt;&gt;&quot;&apos;&#65;&#x42;&#x1F600;&nbsp;&unknown; & raw &#xZZ;</a>",
        )
        .unwrap();
        let element = document.root().unwrap();
        assert_eq!(
            element.text,
            "&<>\"'AB\u{1F600}&nbsp;&unknown; & raw &#xZZ;"
        );
        assert_eq!(decode_entities("no entities"), "no entities");
        assert_eq!(decode_entities("trailing &"), "trailing &");
    }

    #[test]
    fn handles_comments_cdata_self_closing_and_declarations() {
        let document = parse(
            r"<!DOCTYPE a [ <!ELEMENT a (#PCDATA)> ]>
            <!-- leading comment -->
            <a>before<!-- inner -->after<![CDATA[ <raw & text> ]]>tail<b/>end</a>
            <!-- trailing -->",
        )
        .unwrap();
        assert!(document.prolog.is_none());
        let element = document.root().unwrap();
        assert_eq!(element.text, "beforeafter <raw & text> tailend");
        assert_eq!(element.text_content(), "beforeafter <raw & text> tailend");
        assert_eq!(element.children.len(), 1);
        assert_eq!(element.children[0].name, "b");
        assert_eq!(element.children[0].children.len(), 0);
    }

    #[test]
    fn preserves_whitespace_and_keeps_multiple_roots() {
        let document = parse("<a>  x  y  </a><b>&#x9;z&#x9;</b>").unwrap();
        assert_eq!(document.roots.len(), 2);
        assert_eq!(document.roots[0].text, "  x  y  ");
        assert_eq!(document.roots[1].text, "\tz\t");
        assert_eq!(parse("\u{feff}<a/>").unwrap().roots.len(), 1);
        assert_eq!(parse("   \n\t  ").unwrap().roots, Vec::<XmlElement>::new());
    }

    #[test]
    fn structural_damage_is_reported_with_the_element_name() {
        let error = parse("<a><b></a>").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("mismatched end tag"), "{error}");
        assert!(error.to_string().contains("<b>"), "{error}");

        let error = parse("<a><b>text").unwrap_err();
        assert!(error.to_string().contains("unclosed element"), "{error}");

        let error = parse("<a").unwrap_err();
        assert!(error.to_string().contains("unclosed start tag"), "{error}");

        let error = parse("<a b=c/>").unwrap_err();
        assert!(error.to_string().contains("must be quoted"), "{error}");

        let error = parse("<a b/>").unwrap_err();
        assert!(error.to_string().contains("has no value"), "{error}");

        let error = parse("<a><!-- open").unwrap_err();
        assert!(
            error.to_string().contains("unterminated comment"),
            "{error}"
        );

        let error = parse("<a><![CDATA[ open").unwrap_err();
        assert!(error.to_string().contains("CDATA"), "{error}");

        let error = parse("<a><?pi").unwrap_err();
        assert!(
            error.to_string().contains("processing instruction"),
            "{error}"
        );

        let error = parse("<a/>text").unwrap_err();
        assert!(error.to_string().contains("text outside"), "{error}");

        let error = parse("</a>").unwrap_err();
        assert!(error.to_string().contains("unexpected end tag"), "{error}");

        let error = parse("<>").unwrap_err();
        assert!(error.to_string().contains("expected an element"), "{error}");
    }

    #[test]
    fn nesting_depth_is_bounded() {
        let depth = MAX_DEPTH + 1;
        let mut xml = "<a>".repeat(depth);
        xml.push_str(&"</a>".repeat(depth));
        let error = parse(&xml).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error.to_string().contains("nesting exceeds"), "{error}");

        // The limit itself is still accepted.
        let xml = format!(
            "{}{}",
            "<a>".repeat(MAX_DEPTH - 1),
            "</a>".repeat(MAX_DEPTH - 1)
        );
        let document = parse(&xml).unwrap();
        let mut element = document.root().unwrap();
        let mut levels = 1;
        while let Some(child) = element.children.first() {
            element = child;
            levels += 1;
        }
        assert_eq!(levels, MAX_DEPTH - 1);
    }

    #[test]
    fn local_name_and_matching_rules_are_documented_behavior() {
        assert_eq!(local_name("w:p"), "p");
        assert_eq!(local_name("p"), "p");
        assert_eq!(local_name("a:b:c"), "c");
        // Prefixed requests need an exact match.
        let document = parse("<root><x:a><b/></x:a></root>").unwrap();
        let root = document.root().unwrap();
        assert!(root.find_first("x:a").is_some());
        assert!(root.find_first("root").is_none());
        assert_eq!(root.find_first("a").unwrap().name, "x:a");
        assert_eq!(root.children_named("a").len(), 1);
        assert!(root.find_first("b").is_some());
    }
}
