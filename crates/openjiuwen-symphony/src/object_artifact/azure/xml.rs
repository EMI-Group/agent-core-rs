//! The tolerant XML scanner the Azure Blob listings are read with.
//!
//! The workspace ships no XML crate, and both listing documents
//! (`ListContainersResult`/`EnumerationResults`) are flat enough that a tag scan is smaller —
//! and cannot panic on malformed input. Attributes are ignored, self-closing and unclosed tags
//! are skipped, and each parsed text is unescaped.

/// Marker query value that ends a listing.
const NEXT_MARKER: &str = "NextMarker";

/// The next-page marker of a listing page, or `None` on the last page.
///
/// The marker is the unescaped `<NextMarker>` text. An empty marker (the last page), a marker
/// that repeats the one just followed, or a marker document without the element all end the
/// pagination, so a misbehaving service cannot make the client loop.
pub(super) fn next_marker(xml: &str, previous: Option<&str>) -> Option<String> {
    first_tag(xml, NEXT_MARKER)
        .filter(|marker| !marker.is_empty())
        .filter(|marker| previous != Some(marker.as_str()))
}

/// Returns the first unescaped `<tag>…</tag>` text of a document.
pub(super) fn first_tag(xml: &str, tag: &str) -> Option<String> {
    tag_texts(xml, tag).into_iter().next()
}

/// Returns the unescaped text of every `<tag>…</tag>` occurrence, in order.
///
/// The scan is tolerant: attributes are ignored, self-closing and unclosed tags are skipped, and
/// malformed input yields whatever was parsed so far instead of panicking.
pub(super) fn tag_texts(xml: &str, tag: &str) -> Vec<String> {
    let open = format!("<{tag}");
    let close = format!("</{tag}>");
    let mut texts = Vec::new();
    let mut rest = xml;
    while let Some(start) = rest.find(&open) {
        let after_name = &rest[start + open.len()..];
        let Some(end_of_name) = after_name.find('>') else {
            break;
        };
        let boundary = after_name.chars().next();
        let is_this_element =
            matches!(boundary, Some('>' | '/')) || boundary.is_some_and(char::is_whitespace);
        let inner = &after_name[end_of_name + 1..];
        if !is_this_element {
            // A longer tag name (`<Blobs>`): keep scanning after its `>`.
            rest = inner;
            continue;
        }
        match inner.find(&close) {
            Some(end) => {
                texts.push(unescape(inner[..end].trim()));
                rest = &inner[end + close.len()..];
            }
            None => rest = inner,
        }
    }
    texts
}

/// Replaces the five XML entities the Blob service emits, leaving unknown entities untouched.
pub(super) fn unescape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(position) = rest.find('&') {
        out.push_str(&rest[..position]);
        let tail = &rest[position..];
        let Some(semicolon) = tail.find(';') else {
            out.push_str(tail);
            return out;
        };
        let entity = &tail[1..semicolon];
        let replacement = match entity {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            _ => None,
        };
        if let Some(character) = replacement {
            out.push(character);
            rest = &tail[semicolon + 1..];
        } else {
            out.push('&');
            rest = &tail[1..];
        }
    }
    out.push_str(rest);
    out
}
